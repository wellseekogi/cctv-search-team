from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import time

import av
from fastapi.testclient import TestClient
from PIL import Image
import pytest

from embedding.contracts import file_sha256, normalized, write_jsonl
from team_search import live
from team_search.live_input import T8_SOURCE, clip_inputs
from tests.test_embedding import TestEncoder as FixtureEncoder


@pytest.fixture
def short_path():
    with TemporaryDirectory(dir=live.ROOT) as directory:
        yield Path(directory)


def test_live_versions_use_separate_indexes_for_the_same_video(monkeypatch, short_path):
    chat_calls = []

    class Encoder(FixtureEncoder):
        def encode_text(self, texts):
            return normalized([[1] * self.dim for _ in texts], self.dim)

    class Chat:
        ready = True

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def chat(self, *_):
            chat_calls.append(True)
            return "검색 후보를 확인하세요."

    monkeypatch.setattr(live, "get_config", lambda _: {"crop_margin_ratio": 0.1, "batch_size_cpu": 2})
    monkeypatch.setattr(live, "create_encoder", lambda _: Encoder())
    monkeypatch.setattr(live, "ensure_assets", lambda *_: {"server_path": None, "model_path": None})
    monkeypatch.setattr(live, "LocalLLM", Chat)
    args = SimpleNamespace(store=short_path / "live", llm_port=0)

    clip = short_path / "source.mp4"
    with av.open(str(clip), "w") as output:
        stream = output.add_stream("libx264", rate=5)
        stream.width = stream.height = 64
        for color in ("black", "white", "red", "blue"):
            for _ in range(10):
                for packet in stream.encode(av.VideoFrame.from_image(Image.new("RGB", (64, 64), color))):
                    output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    key = file_sha256(clip)
    manifest, frames = clip_inputs(clip, key, clip.name)

    def wait_for(client, group, job_id):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            job = next(row for row in client.get("/api/status").json()[group] if row["id"] == job_id)
            if job["state"] in ("ready", "failed"):
                assert job["state"] == "ready", job["error"]
                return
            time.sleep(0.05)
        raise AssertionError(f"{job_id} did not finish")

    with TestClient(live.create_app(args), base_url="http://127.0.0.1") as client:
        assert 'search_versions = true' in client.get("/ui/config.js").text
        assert client.post("/api/events/clip", content=clip.read_bytes()).status_code == 202
        wait_for(client, "jobs", key)
        query = {"text": "a person", "video_ids": [key]}
        assert client.post("/search", json=dict(query, search_version="b_t8")).status_code == 409

        inbox = Path(args.store) / "inbox" / "t8-test"
        inbox.mkdir()
        (inbox / clip.name).write_bytes(clip.read_bytes())
        write_jsonl(inbox / "manifest.jsonl", manifest)
        write_jsonl(inbox / "frames.jsonl",
                    [dict(f, detector_status="ran", source_offset_ms=0, time_source="pts") for f in frames])
        write_jsonl(inbox / "tracks.jsonl", [dict(schema_version="cctv-3.0", source="pred", video_id=key,
                    track_source_id=T8_SOURCE, local_track_id="t1", track_uid=f"{T8_SOURCE}:{key}:t1",
                    frame_index=f["frame_index"], timestamp_ms=f["timestamp_ms"], class_name="person",
                    bbox=[8, 8, 56, 56], confidence=0.9, quality={"area_ratio": 0.5, "blur_var": 50,
                    "border_touch": False}) for f in frames])
        write_jsonl(inbox / "video_status.jsonl", [{"video_id": key, "video_status": "completed",
                     "available_end_ms": manifest[0]["available_end_ms"]}])
        job = client.post("/api/events/handoff/t8-test?version=b_t8")
        assert job.status_code == 202
        wait_for(client, "b_t8_jobs", job.json()["id"])
        assert len(client.get("/api/status").json()["jobs"]) == 1

        versions = set()
        for version in ("original", "b_t8", "pixel"):
            representation = "object" if version == "b_t8" else "scene"
            response = client.post("/search", json=dict(query, search_version=version, representation=representation))
            assert response.status_code == 200, response.text
            result = response.json()
            versions.add(result["index_version"])
            assert result["total_candidates"] > 0
            first = result["results"][0]
            assert first["video_id"] == key and first["representation"] == representation
            assert client.get(first["thumbnail_url"]).status_code == 200
            assert client.get(first["video_url"], headers={"Range": "bytes=0-511"}).status_code == 206
            if version == "pixel":
                assert first["end_ms"] - first["start_ms"] == 4000
        assert len(versions) == 3
        assert len(chat_calls) == 3
        for version in ("original", "b_t8", "pixel"):
            response = client.post("/search?retrieval_only=true", json=dict(query, search_version=version))
            assert response.status_code == 200 and response.json()["results"]
            assert "answer" not in response.json()
        assert len(chat_calls) == 3
