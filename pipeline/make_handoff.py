import hashlib, json, shutil, sys
from pathlib import Path

camera, src, video = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])

rows = [json.loads(l) for l in open("data/meva/manifest.jsonl", encoding="utf-8")]
f = next(r for r in rows if r["camera_id"] == camera)

first = json.loads(open(src / "frames.jsonl", encoding="utf-8").readline())
print("frames 첫 줄:", first)
b_id = first["video_id"]

sha = hashlib.sha256()
with open(video, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
        sha.update(chunk)
if sha.hexdigest() != f["file_sha256"]:
    sys.exit(f"SHA-256 불일치: {sha.hexdigest()}")

if f["split"] != "dev":
    print(f"주의: 원래 split={f['split']}. 로컬 데모용 사본에서만 dev로 바꿉니다.")

out = Path("runs/live/inbox") / camera.lower()
out.mkdir(parents=True, exist_ok=True)
for name in ("frames.jsonl", "tracks.jsonl"):
    shutil.copyfile(src / name, out / name)
shutil.copyfile(video, out / video.name)

manifest = dict(f, video_id=b_id, relative_path=video.name, split="dev")
(out / "manifest.jsonl").write_text(json.dumps(manifest, ensure_ascii=False) + "\n", encoding="utf-8")
status = dict(video_id=b_id, available_end_ms=f["available_end_ms"], video_status="completed")
(out / "video_status.jsonl").write_text(json.dumps(status) + "\n", encoding="utf-8")
print("완료:", out)