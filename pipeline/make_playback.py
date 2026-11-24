"""run의 원본 AVI를 브라우저가 재생할 수 있는 H.264 MP4로 변환해 재생용으로 연결한다.
사용: python make_playback.py <run_id> [변환할_초]
"""
import json
import sys
from fractions import Fraction
from pathlib import Path

import av

run_id = sys.argv[1]
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else None
FPS, WIDTH = 30, 960

if "libx264" not in av.codecs_available:
    sys.exit("이 PyAV에는 libx264가 없습니다. 이 메시지를 알려주세요.")

folder = Path("runs") / run_id / "indexes" / run_id
manifest = folder / "manifest.jsonl"
if not manifest.is_file():
    sys.exit(f"manifest가 없습니다: {manifest}")
rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]


def transcode(src, dst, width, height, max_frames):
    with av.open(str(src)) as inp, av.open(str(dst), "w", options={"movflags": "faststart"}) as out:
        vin = inp.streams.video[0]
        vout = out.add_stream("libx264", rate=FPS)
        vout.width, vout.height, vout.pix_fmt = width, height, "yuv420p"
        vout.options = {"crf": "26", "preset": "veryfast"}
        count = 0
        for frame in inp.decode(vin):
            if max_frames is not None and count >= max_frames:
                break
            small = frame.reformat(width=width, height=height, format="yuv420p")
            small.pts, small.time_base = count, Fraction(1, FPS)
            for packet in vout.encode(small):
                out.mux(packet)
            count += 1
            if count % 300 == 0:
                print(f"  {count}프레임", flush=True)
        for packet in vout.encode():
            out.mux(packet)
    return count


backup = manifest.with_name("manifest.jsonl.orig")
if not backup.exists():
    backup.write_bytes(manifest.read_bytes())

for row in rows:
    relative = Path(row["relative_path"])
    if relative.suffix.lower() == ".mp4":
        print("이미 mp4:", relative)
        continue
    src = folder / "input" / relative
    dst = src.with_suffix(".mp4")
    height = round(WIDTH * row["height"] / row["width"] / 2) * 2
    limit = int(seconds * FPS) if seconds else None
    print(f"변환: {src.name} -> {dst.name} ({WIDTH}x{height}, {'전체' if limit is None else f'{limit}프레임'})")
    count = transcode(src, dst, WIDTH, height, limit)
    row["relative_path"] = relative.with_suffix(".mp4").as_posix()
    print(f"완료 {count}프레임, {dst.stat().st_size // 1024} KB")

manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
print("manifest 갱신:", manifest, "\n원본 보관:", backup)