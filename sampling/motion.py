"""Motion from consecutive candidate RGB frames, not from the last saved frame."""
import hashlib
import numpy as np
from PIL import Image

from .core import InputError, require
from .uniform import select_uniform

PREPROCESS = "pillow_rgb_bilinear_160x90_then_L_float32_v1"


def gray_frame(rgb, frame):
    require(isinstance(rgb,np.ndarray) and rgb.dtype == np.uint8,
            "frame_loader must return an RGB uint8 NumPy array")
    require(rgb.shape == (frame["height"],frame["width"],3), "RGB frame dimensions mismatch")
    return np.asarray(Image.fromarray(rgb).resize((160,90),Image.Resampling.BILINEAR).convert("L"),dtype=np.float32)


def select_motion(config, videos, frames, tracks, exclusions, run_id, frame_loader):
    preprocessing = config.get("motion",{}).get("preprocess",PREPROCESS)
    require(preprocessing == PREPROCESS, "Unsupported motion preprocessing")
    scores, chosen = [], []
    previous, previous_frame = None, None
    threshold = config["sampling"]["motion_threshold"]
    for f in sorted(frames,key=lambda x:(x["video_id"],x["timestamp_ms"],x["frame_index"])):
        if previous_frame is not None and previous_frame["video_id"] != f["video_id"]:
            previous,previous_frame=None,None
        row=dict(schema_version="cctv-3.0",source_run_id=run_id,video_id=f["video_id"],frame_index=f["frame_index"],
                 timestamp_ms=f["timestamp_ms"],previous_timestamp_ms=previous_frame["timestamp_ms"] if previous_frame else None,
                 motion_score=None,threshold=threshold,selected_by_motion=False,preprocess=PREPROCESS)
        if f["decode_status"] == "failed":
            row["status"]="DECODE_FAILED"
            scores.append(row)
            previous,previous_frame=None,None
            continue
        # Missing/corrupt images are errors, not fabricated zero-motion observations.
        rgb=frame_loader(f)
        current=gray_frame(rgb,f)
        row["rgb_sha256"]=hashlib.sha256(rgb.tobytes()).hexdigest()
        score=None if previous is None else float(np.mean(np.abs(current-previous),dtype=np.float32))/255.0
        selected=previous is None or score >= threshold
        row.update(motion_score=score,selected_by_motion=selected,
                   status="FIRST_VALID" if previous is None else "THRESHOLD_MET" if selected else "BELOW_THRESHOLD")
        scores.append(row)
        if selected:
            chosen.append(f)
        # Always update, including excluded and non-selected frames.
        previous,previous_frame=current,f
    events,samples,summary=select_uniform(config,videos,frames,tracks,exclusions,run_id,selected_frames=chosen)
    summary.pop("fixed_step_ms",None)
    summary.update(motion_threshold=threshold,motion_preprocess=PREPROCESS,motion_scores=scores,
                   motion_candidate_count=len(chosen),failure_rule="reset comparison after known decode failure")
    return events,samples,summary
