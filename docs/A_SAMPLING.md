# A sampling — B/F integration v10

Python 3.11. A selects scene/person candidates from B frame/track observations. It does not run object detection, generate embeddings, or infer human actions.

## Install and test

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-motion.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
.\.venv\Scripts\python.exe -m sampling --input fixtures/event_trace --config configs/sampling.yaml --out runs/synthetic
```

Synthetic fixture: expected 9 events, 5 object candidates, 8 scene candidates. The vehicle class in this historical synthetic fixture tests generic logic; it is not a real vehicle evaluation. Use configs/person/ for current MEVA person work. Motion tests use synthetic pixels.

## Reproduce the B/F integration

B observations and F metadata are separate dependencies. This A branch includes only the six provisional B manifests from the v10 handoff under `metadata/baseline_T3/<camera>/manifest.PROVISIONAL.jsonl`. Obtain the approved B baseline_T3 per-camera input folders and F checkout separately. Existing v8/v10 handoff inputs have the required format:

```text
B-inputs/<camera>/frames.jsonl
B-inputs/<camera>/tracks.jsonl
B-inputs/<camera>/manifest.PROVISIONAL.jsonl   # or manifest.jsonl
F-checkout/data/meva/manifest.jsonl
F-checkout/data/meva/queries.jsonl
```

The B manifest must report original SHA256, width, height, source_offset_ms and video_id. It is used only to establish a one-to-one match with F metadata; actual duration, split and media path come from F. A raw G330 ZIP is not this six-camera baseline input.

```powershell
.\.venv\Scripts\python.exe tools/prepare_bf_v10.py --b-inputs "../A_handoff_v10_BF_GitHub/runs/meva_baseline_T3_v8/inputs" --f-data "../F-checkout/data/meva" --out runs/bf_reproduced
```

Both paths are explicit. No existing local runs/reviews folder is required by the checkout. B frames/tracks and F input JSONLs remain external. The six provisional manifests are versioned separately as described below. New output paths are required to prevent overwrites. F input hashes and B frame/track hashes are recorded in generated configurations.

## Provisional manifests included in this branch

In the integrated checkout, pass `--b-manifests metadata/baseline_T3` to
`tools/prepare_bf_v10.py` to read the submitted manifests directly alongside
the external B observations. The original B input folders stay intact.

The files under `metadata/baseline_T3/` are byte-for-byte copies from
`A_handoff_v10_BF_GitHub/runs/meva_baseline_T3_v8/inputs/` for G299, G340,
G341, G421, G506 and G638. Copy each file into the corresponding B camera
folder alongside `frames.jsonl` and `tracks.jsonl` before running the command
above. For example, from this checkout:

```powershell
$bInputs = "C:/path/to/B-inputs"
Get-ChildItem metadata/baseline_T3 -Directory | ForEach-Object {
    Copy-Item -LiteralPath (Join-Path $_.FullName "manifest.PROVISIONAL.jsonl") -Destination (Join-Path $bInputs $_.Name)
}
```

These are provisional observation envelopes, not authoritative media metadata:
`relative_path` is null, `split` is unassigned, and duration is a synthetic
bound for the supplied observations. Keep the provisional flags intact.
`prepare_bf_v10.py` obtains actual media metadata and split from F's
`data/meva/manifest.jsonl` by matching the reported SHA256. These manifests
alone do not include the B frames/tracks or original videos. G330 is not part
of this six-video v10 integration.

## ID and exclusion contracts

B video IDs have a meva_ prefix absent from F IDs. A joins manifests by reported SHA256 and checks dimensions, offset and frame/time range. B video_id and full track_uid remain unchanged. video_id_map.json maps to F identifiers. Use this table for F lookups; never strip prefixes or equate GT and predicted track IDs. The mapping does not verify source media bytes.

F image queries contain exclusion intervals. A converts their video IDs and excludes [start_ms,end_ms) from every selection policy. excluded_queries.jsonl is a class/answer exclusion log, not a temporal exclusion file. Outputs retain B actual timestamp_ms, frame_index and original xyxy pixel boxes.

Six baseline videos: dev G340/G341/G421/G506, eval G299/G638. G330 has no entry in this F manifest and remains a separate v9 validation example, not an evaluation gallery member.

## Output and limitations

results/<camera>/<policy>/ contains samples, events, configuration, summary, smoke scene/object pair and expected crop coordinates. Policies: Event, hybrid, uniform 1/2/5 seconds. Motion requires RGB and is pending for real videos.

B completion logs were not supplied: runs stay observed_partial/completion_pending, no VIDEO_END is manufactured. Real video bbox/PTS parity, C embedding, GT matching, and F/G independent acceptance remain pending. Zero file errors does not establish detection accuracy. The C static checker is pinned to 1a6800865e1392fbae294b20aa0fbd5356e89c07 and is not current remote runtime verification.

## B reader

Install requirements-b-reader.txt only for actual decoding. integrations/b_reader.py accepts CCTV_B_READER_CONFIG pointing to a local JSON containing b_reader_module, manifest, vision_config and data_root (paths relative to that JSON). Use the aliased manifest.B_ALIASED.jsonl produced by this tool and B src/b_video_io.py. It verifies original file hashes before reading and normalizes rgb to image_rgb. Do not commit local settings.

## Dependency revisions used for v10

- B feature/b-litedetr-bytetrack: b6bcfe7a38199883f5409862caf4dfafdfa4e61a
- F feature/f-data-virat-meva: 326617d32c4a6809cf716ca1c5fe131568fdf9d1
- Repository: https://github.com/wellseekogi/cctv-search-team

This PR can be unit-tested independently. Real integration requires the external B observations and F metadata above. Neither develop merge nor real media acceptance is implied.
