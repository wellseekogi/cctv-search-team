import json
import logging
import shutil
import sqlite3
import tempfile
import threading
from contextlib import closing
from pathlib import Path

from embedding.contracts import file_sha256, read_jsonl, within
from rag.engine import FrameRecommender
from .live_input import process_segment


class LiveStore:
    def __init__(self, root, config, encoder):
        self.root = Path(root).resolve()
        for folder in ("uploads", "inbox", "indexes"):
            (self.root / folder).mkdir(exist_ok=True, parents=True)
        self.database = self.root / "events.sqlite3"
        self.config, self.encoder = config, encoder
        self.encoder_lock = threading.Lock()
        self.catalog = {}
        self.b_t8 = {}
        self.wake, self.stop = threading.Event(), threading.Event()
        with closing(self.connect()) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, name TEXT NOT NULL,
                state TEXT NOT NULL, events INTEGER DEFAULT 0, vectors INTEGER DEFAULT 0,
                error TEXT DEFAULT '')""")
            db.execute("UPDATE jobs SET state='queued' WHERE state='processing'")
            db.execute("UPDATE jobs SET state='archived' WHERE kind='demo' AND id='demo-v1'")
            for job in db.execute("SELECT * FROM jobs WHERE state='ready'"):
                self.load(job["id"], job["kind"])
        self.worker = threading.Thread(target=self.run, daemon=True)
        self.worker.start()

    def connect(self):
        db = sqlite3.connect(self.database, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def load(self, key, kind):
        folder = self.root / "indexes" / key
        engine = FrameRecommender(folder / "index", self.encoder)
        media = {m["video_id"]: (within(folder / "input", m["relative_path"]), m["available_end_ms"],
                                 m.get("display_name", m["video_id"]))
                 for m in read_jsonl(folder / "manifest.jsonl")}
        if kind == "b_t8":
            self.b_t8 = {**self.b_t8, key: (engine, media)}
        else:
            self.catalog = {**self.catalog, key: (engine, media)}

    def submit(self, kind, key, name):
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT state FROM jobs WHERE id=?", (key,)).fetchone()
            if row is None or row["state"] == "failed":
                pending = db.execute("SELECT count(*) FROM jobs WHERE state IN ('queued','processing')").fetchone()[0]
                if pending >= 32:
                    raise OverflowError("대기열이 가득 찼습니다. 잠시 뒤 다시 보내 주세요.")
                db.execute("""INSERT INTO jobs(id, kind, name, state) VALUES(?,?,?,'queued')
                           ON CONFLICT(id) DO UPDATE SET state='queued', error=''""", (key, kind, name))
        self.wake.set()
        return key

    def status(self):
        with closing(self.connect()) as db:
            return [dict(row) for row in db.execute(
                "SELECT * FROM jobs WHERE state != 'archived' ORDER BY rowid DESC LIMIT 32")]

    def process(self, job):
        key, kind = job["id"], job["kind"]
        final = self.root / "indexes" / key
        if not final.exists():
            with tempfile.TemporaryDirectory(dir=final.parent) as temp:
                stage = Path(temp) / "segment"
                stage.mkdir()
                source = (self.root / "uploads" / key if kind == "clip"
                          else within(self.root / "inbox", job["name"]))
                with self.encoder_lock:
                    counts = process_segment(kind, key, job["name"], source, stage, self.config, self.encoder)
                if kind == "clip":
                    # Use a hard link when possible so replay survives queue cleanup.
                    target = stage / "input" / source.name
                    try:
                        target.hardlink_to(source)
                    except OSError:
                        shutil.copyfile(source, target)
                elif kind in ("handoff", "b_t8"):
                    for row in read_jsonl(stage / "manifest.jsonl"):
                        relative = row["relative_path"]
                        target = within(stage / "input", relative)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(within(source, relative), target)
                        if file_sha256(target) != row["file_sha256"]:
                            raise ValueError("인계 영상이 처리 도중 변경됐습니다.")
                (stage / "counts.json").write_text(json.dumps(counts), encoding="utf-8")
                stage.rename(final)
        counts = json.loads((final / "counts.json").read_text(encoding="utf-8"))
        self.load(key, kind)
        return counts

    def run(self):
        while not self.stop.is_set():
            self.wake.clear()
            if self.stop.is_set():
                return
            with closing(self.connect()) as db, db:
                job = db.execute("SELECT * FROM jobs WHERE state='queued' ORDER BY rowid LIMIT 1").fetchone()
                if job:
                    db.execute("UPDATE jobs SET state='processing' WHERE id=?", (job["id"],))
            if not job:
                self.wake.wait()
                continue
            try:
                events, vectors = self.process(job)
                state, error = "ready", ""
            except Exception as exc:
                logging.exception("Event processing failed for %s", job["id"])
                state, error, events, vectors = "failed", str(exc)[:600], 0, 0
            with closing(self.connect()) as db, db:
                db.execute("UPDATE jobs SET state=?,error=?,events=?,vectors=? WHERE id=?",
                           (state, error, events, vectors, job["id"]))

    def close(self):
        self.stop.set()
        self.wake.set()
        self.worker.join()
        self.catalog = {}
        self.b_t8 = {}
