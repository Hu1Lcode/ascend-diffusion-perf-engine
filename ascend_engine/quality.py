"""Fast three-sample video comparison against an immutable baseline run."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path


PSNR = re.compile(r"average:([\d.]+|inf)")
SSIM = re.compile(r"All:([\d.]+)")


def _read_run(run_dir: Path) -> dict:
    path = run_dir / "outputs" / "benchmark.json"
    data = json.loads(path.read_text())
    if data.get("status") != "completed":
        raise ValueError(f"run is not completed: {path}")
    return data


def _probe(path: Path) -> dict:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_read_frames,avg_frame_rate", "-of", "json", str(path)],
        text=True, capture_output=True, check=True,
    )
    rows = json.loads(proc.stdout).get("streams", [])
    if len(rows) != 1:
        raise ValueError(f"expected one video stream: {path}")
    if not rows[0].get("nb_read_frames") or rows[0]["nb_read_frames"] == "N/A":
        raise ValueError(f"could not count video frames: {path}")
    return rows[0]


def _filter_score(reference: Path, candidate: Path, kind: str) -> float:
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "info", "-i", str(reference),
         "-i", str(candidate), "-lavfi", kind, "-f", "null", "-"],
        text=True, capture_output=True, check=True,
    )
    match = (PSNR if kind == "psnr" else SSIM).search(proc.stderr)
    if match is None:
        raise ValueError(f"ffmpeg {kind} output had no score")
    value = match.group(1)
    return float("inf") if value == "inf" else float(value)


def compare_runs(baseline_dir: Path, candidate_dir: Path,
                 min_psnr: float = 20.0, min_ssim: float = 0.8) -> dict:
    """Compare each sample; caller must arrange a separate visual review."""
    baseline, candidate = _read_run(baseline_dir), _read_run(candidate_dir)
    first, second = baseline["run_spec"], candidate["run_spec"]
    for key in ("task", "model", "model_fingerprint", "samples_sha256",
                "entrypoint_sha256", "runtime", "ascend_runtime"):
        if first[key] != second[key]:
            raise ValueError(f"incomparable runs: {key} differs")
    if first["samples"] != second["samples"]:
        raise ValueError("incomparable runs: sample contents or I2V image hashes differ")
    if first["settings"] != second["settings"]:
        # Features are captured separately, so all workload settings must match.
        left = {k: v for k, v in first["settings"].items() if k not in {"cache_backend", "vae_strategy"}}
        right = {k: v for k, v in second["settings"].items() if k not in {"cache_backend", "vae_strategy"}}
        if left != right:
            raise ValueError("incomparable runs: workload settings differ")
    a = {row["sample_id"]: row for row in baseline["runs"]}
    b = {row["sample_id"]: row for row in candidate["runs"]}
    if set(a) != set(b):
        raise ValueError("incomparable runs: sample IDs differ")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        result = {"status": "blocked", "reason": "ffmpeg_and_ffprobe_required", "samples": []}
    else:
        rows = []
        for sid in sorted(a):
            reference = Path(a[sid]["video"])
            trial = Path(b[sid]["video"])
            probe_a, probe_b = _probe(reference), _probe(trial)
            if (probe_a.get("width"), probe_a.get("height"), probe_a.get("nb_read_frames"),
                probe_a.get("avg_frame_rate")) != (
                probe_b.get("width"), probe_b.get("height"), probe_b.get("nb_read_frames"),
                probe_b.get("avg_frame_rate")
            ):
                rows.append({"sample_id": sid, "status": "failed_shape_or_frame_count",
                             "baseline": probe_a, "candidate": probe_b})
                continue
            psnr, ssim = _filter_score(reference, trial, "psnr"), _filter_score(reference, trial, "ssim")
            rows.append({"sample_id": sid, "psnr_db": "inf" if psnr == float("inf") else psnr,
                         "ssim": ssim,
                         "status": "numeric_pass" if psnr >= min_psnr and ssim >= min_ssim else "numeric_fail"})
        result = {
            "status": "numeric_pass_pending_visual_review"
            if rows and all(row["status"] == "numeric_pass" for row in rows)
            else "numeric_fail",
            "samples": rows,
            "thresholds": {"min_psnr_db": min_psnr, "min_ssim": min_ssim},
            "note": "Heuristic smoke gate; final promotion requires visual review of 3–5 videos.",
        }
    result["baseline_run"] = str(baseline_dir)
    result["candidate_run"] = str(candidate_dir)
    (candidate_dir / "outputs" / "quality.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
