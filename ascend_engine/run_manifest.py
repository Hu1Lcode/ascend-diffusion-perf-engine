"""Execute one Sol-Engine run bundle through pinned vLLM-Omni entrypoints."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from ascend_engine.features import feature_plan
from ascend_engine.vllm_adapter import parse_metrics, render_entrypoint, runtime_env


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} must be set")
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_fingerprint(model: Path) -> str:
    """Cheap change detector; avoid reading multi-gigabyte checkpoint contents."""
    files = sorted(path for path in model.rglob("*") if path.is_file())
    records = [(str(path.relative_to(model)), path.stat().st_size, path.stat().st_mtime_ns)
               for path in files]
    return hashlib.sha256(json.dumps(records).encode()).hexdigest()


def _ascend_preflight(python_bin: str, require_mindiesd: bool) -> dict:
    code = (
        "import importlib.util,json,torch,torch_npu; "
        "assert hasattr(torch,'npu') and torch.npu.is_available(), 'NPU unavailable'; "
        "print(json.dumps({'torch':torch.__version__,"
        "'torch_npu':torch_npu.__version__,"
        "'npu':torch.npu.get_device_name(0),"
        "'mindiesd':importlib.util.find_spec('mindiesd') is not None}))"
    )
    proc = subprocess.run([python_bin, "-c", code], text=True, capture_output=True)
    if proc.returncode:
        raise RuntimeError(f"Ascend torch_npu preflight failed: {proc.stderr[-1200:]}")
    info = json.loads(proc.stdout.strip().splitlines()[-1])
    if require_mindiesd and not info["mindiesd"]:
        raise RuntimeError("attention_flash requires MindIE-SD (mindiesd) in the selected Python environment")
    return info


def _samples(path: Path, task: str, limit: int, repo_root: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if data.get("task") != task or data.get("version") != 1:
        raise ValueError(f"unexpected sample suite task or version: {path}")
    rows = data.get("samples")
    if not isinstance(rows, list) or len(rows) < 3 or not 3 <= limit <= min(5, len(rows)):
        raise ValueError("sample suite must contain at least 3 samples; sample limit must be 3–5")
    result = []
    ids = set()
    for row in rows[:limit]:
        if not all(k in row for k in ("id", "seed", "prompt")) or row["id"] in ids:
            raise ValueError(f"invalid or duplicate sample: {row}")
        ids.add(row["id"])
        copied = dict(row)
        if task == "i2v":
            raw = Path(str(row.get("image", "")))
            image = (path.parent / raw).resolve()
            if not image.is_file() or not image.is_relative_to(repo_root):
                raise ValueError(f"invalid I2V image: {image}")
            copied["image"] = str(image)
            copied["image_sha256"] = _sha(image)
        result.append(copied)
    return result


def _command(script: Path, model: Path, task: str, sample: dict, output: Path,
             settings: dict[str, str], profile: bool) -> list[str]:
    cmd = [
        os.environ.get("PYTHON_BIN") or sys.executable, str(script),
        "--model", str(model), "--prompt", sample["prompt"],
        "--negative-prompt", sample.get("negative_prompt", ""),
        "--seed", str(sample["seed"]),
        "--height", settings["height"], "--width", settings["width"],
        "--num-frames", settings["frames"],
        "--num-inference-steps", settings["steps"],
        "--guidance-scale", settings["guidance"],
        "--fps", settings["fps"], "--output", str(output),
    ]
    if task == "i2v":
        cmd.extend(["--image", sample["image"]])
    if settings.get("cache_backend"):
        cmd.extend(["--cache-backend", settings["cache_backend"]])
        if settings["cache_backend"] == "cache_dit":
            cmd.append("--enable-cache-dit-summary")
    if settings.get("vae_strategy"):
        cmd.append("--vae-use-" + settings["vae_strategy"])
    if profile:
        cmd.append("--enable-diffusion-pipeline-profiler")
    return cmd


def _run_one(cmd: list[str], log_path: Path, env: dict[str, str], cwd: Path,
             timeout_s: int) -> dict:
    started = time.perf_counter()
    try:
        with log_path.open("w") as out:
            proc = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT,
                                  env=env, cwd=cwd, timeout=timeout_s, check=False)
        status = "completed" if proc.returncode == 0 else "failed"
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        status, returncode = "timeout", None
    log = log_path.read_text(errors="replace")
    return {
        "status": status, "returncode": returncode,
        "process_s_including_load": round(time.perf_counter() - started, 4),
        "log": str(log_path), **parse_metrics(log),
    }


def main() -> int:
    repo_root = Path(_required_env("AUTOVIDEO_REPO_ROOT")).resolve()
    output = Path(_required_env("OUT_DIR")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    vllm_root = Path(_required_env("VLLM_OMNI_ROOT")).expanduser().resolve()
    model = Path(_required_env("MODEL_PATH")).expanduser().resolve()
    task = os.environ.get("ASCEND_TASK", "t2v").lower()
    sample_rel = _required_env("ASCEND_SAMPLES")
    sample_path = (repo_root / sample_rel).resolve()
    if not sample_path.is_file() or not sample_path.is_relative_to(repo_root):
        raise ValueError(f"sample manifest must be a repository file: {sample_path}")
    limit = int(os.environ.get("ASCEND_SAMPLE_LIMIT", "3"))
    samples = _samples(sample_path, task, limit, repo_root)
    feature_names = [part.strip() for part in os.environ.get("ASCEND_FEATURES", "").split(",") if part.strip()]
    feature_env, ordered_features = feature_plan(
        feature_names, os.environ.get("ASCEND_MODEL_ID", "Wan2.2-TI2V-5B"))
    settings = {
        "width": _required_env("ASCEND_WIDTH"),
        "height": _required_env("ASCEND_HEIGHT"),
        "frames": _required_env("ASCEND_FRAMES"),
        "steps": _required_env("ASCEND_STEPS"),
        "fps": _required_env("ASCEND_FPS"),
        "guidance": _required_env("ASCEND_GUIDANCE"),
        "cache_backend": feature_env.get("ASCEND_CACHE_BACKEND", ""),
        "vae_strategy": feature_env.get("ASCEND_VAE_STRATEGY", ""),
    }
    script = render_entrypoint(vllm_root, task, output / "entrypoint.py")
    snapshot = {
        "task": task, "runtime": str(vllm_root), "model": str(model),
        "samples_sha256": _sha(sample_path), "samples": samples,
        "settings": settings, "features": ordered_features,
        "entrypoint_sha256": _sha(script),
        "timing_scope": "warm_generation_plus_input_media_and_output_media",
        "profile_is_separate": True,
    }
    if os.environ.get("ASCEND_DRY_RUN") == "1":
        _write_json(output / "run_spec.json", snapshot)
        _write_json(output / "benchmark.json", {"status": "planned", "run_spec": snapshot})
        print(f"Prepared {task} run with {len(samples)} samples; no inference executed")
        return 0
    if not model.is_dir():
        raise ValueError(f"local model weights directory does not exist: {model}")
    snapshot["model_fingerprint"] = _model_fingerprint(model)
    snapshot["ascend_runtime"] = _ascend_preflight(
        os.environ.get("PYTHON_BIN") or sys.executable,
        require_mindiesd="attention_flash" in ordered_features,
    )
    _write_json(output / "run_spec.json", snapshot)
    if shutil.which("ffprobe") is None:
        print("Warning: ffprobe unavailable; video frame validation is limited")
    timeout = int(os.environ.get("ASCEND_TIMEOUT_S", "3600"))
    runs = []
    env = runtime_env(vllm_root, feature_env, warmup=1)
    for sample in samples:
        sample_dir = output / sample["id"]
        sample_dir.mkdir(exist_ok=True)
        video = sample_dir / "output.mp4"
        cmd = _command(script, model, task, sample, video, settings, profile=False)
        print(f"Running {task} sample {sample['id']} ({sample['seed']})", flush=True)
        result = _run_one(cmd, sample_dir / "run.log", env, vllm_root, timeout)
        result.update({"sample_id": sample["id"], "video": str(video),
                       "command": cmd, "seed": sample["seed"]})
        if result["status"] == "completed" and (not video.is_file() or video.stat().st_size == 0):
            result["status"] = "failed_missing_video"
        runs.append(result)
        _write_json(output / "benchmark.json", {"status": "running", "runs": runs, "run_spec": snapshot})
        if (result["status"] != "completed" or result["generation_plus_media_s"] is None
                or (task == "i2v" and result["input_media_decode_preprocess_s"] is None)):
            _write_json(output / "benchmark.json", {"status": "failed", "runs": runs, "run_spec": snapshot})
            raise RuntimeError(f"sample {sample['id']} failed or lacked required timing; see {result['log']}")
    values = [row["generation_plus_media_s"] for row in runs]
    benchmark = {
        "status": "completed", "total_s": statistics.median(values),
        "timing_scope": snapshot["timing_scope"], "sample_count": len(runs),
        "runs": runs, "run_spec": snapshot,
    }
    # Profile separately so synchronize-heavy stage wrappers do not taint timing.
    if os.environ.get("ASCEND_PROFILE", "1") == "1":
        sample = samples[0]
        diag_dir = output / "diagnostic"
        diag_dir.mkdir(exist_ok=True)
        cmd = _command(script, model, task, sample, diag_dir / "output.mp4", settings, profile=True)
        diagnostic = _run_one(cmd, diag_dir / "run.log", runtime_env(vllm_root, feature_env, 0), vllm_root, timeout)
        benchmark["diagnostic"] = diagnostic
    _write_json(output / "benchmark.json", benchmark)
    print(f"Warm generation plus media median: {benchmark['total_s']:.4f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
