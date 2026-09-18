"""A pinned, reproducible bridge to vLLM-Omni 0.28.0 example entrypoints."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path

from ascend_engine import VLLM_OMNI_COMMIT


SOURCE_HASHES = {
    "t2v": "b3dae9dccd1356acde98e2ef5d6759033b68e2276e0c253cc81ba8319c62959d",
    "i2v": "5b3260fd1390f6a2043feef53efc7e198445fe8e3b4bf3a0ab0f5af2220600f9",
}
EXAMPLES = {
    "t2v": "examples/offline_inference/text_to_video/text_to_video.py",
    "i2v": "examples/offline_inference/image_to_video/image_to_video.py",
}
GENERATION = re.compile(r"Total generation time:\s*([\d.]+) seconds")
MEDIA = re.compile(r"Video postprocess and encode time:\s*([\d.]+) seconds")
INPUT_MEDIA = re.compile(r"Input image decode and preprocess time:\s*([\d.]+) seconds")
STAGE = re.compile(r"\[DiffusionPipelineProfiler\]\s+(.+?) took ([\d.]+)s")


def git_revision(path: Path) -> str:
    proc = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=True,
    )
    return proc.stdout.strip()


def verify_source(vllm_root: Path, task: str) -> Path:
    if task not in EXAMPLES:
        raise ValueError(f"unsupported task: {task}")
    if not vllm_root.is_dir():
        raise ValueError(f"vLLM-Omni source directory does not exist: {vllm_root}")
    revision = git_revision(vllm_root)
    if revision != VLLM_OMNI_COMMIT:
        raise ValueError(f"vLLM-Omni must be v0.28.0 ({VLLM_OMNI_COMMIT}); got {revision}")
    source = vllm_root / EXAMPLES[task]
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != SOURCE_HASHES[task]:
        raise ValueError(f"example entrypoint differs from pinned v0.28.0 source: {source}")
    return source


def render_entrypoint(vllm_root: Path, task: str, output: Path) -> Path:
    """Copy and instrument the pinned example; leave the user's checkout clean."""
    source = verify_source(vllm_root, task)
    body = source.read_text()
    if "import os\n" not in body:
        body = body.replace("import time\n", "import os\nimport time\n", 1)
    body = body.replace("import os\n", "import os\nimport random\n", 1)
    seed_anchor = "    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed)\n"
    if body.count(seed_anchor) != 1:
        raise ValueError("random seed anchor changed")
    body = body.replace(
        seed_anchor,
        "    random.seed(args.seed)\n"
        "    np.random.seed(args.seed % (2 ** 32))\n"
        "    torch.manual_seed(args.seed)\n"
        + seed_anchor,
        1,
    )
    begin = "    generation_start = time.perf_counter()\n"
    if body.count(begin) != 1:
        raise ValueError("generation timing anchor changed")
    warmup = (
        "    for _ in range(int(os.environ.get('ASCEND_WARMUP_PASSES', '1'))):\n"
        "        sampling_params.generator = torch.Generator(\n"
        "            device=current_omni_platform.device_type).manual_seed(args.seed)\n"
        "        omni.generate(prompt_dict, sampling_params)\n"
        "    sampling_params.generator = torch.Generator(\n"
        "        device=current_omni_platform.device_type).manual_seed(args.seed)\n"
    )
    body = body.replace(begin, warmup + begin, 1)
    if task == "i2v":
        image_open = '    image = PIL.Image.open(args.image).convert("RGB") if args.image else None\n'
        image_done = "    # Configure cache based on backend type\n"
        if body.count(image_open) != 1 or body.count(image_done) != 1:
            raise ValueError("I2V input media timing anchors changed")
        body = body.replace(image_open, "    input_media_start = time.perf_counter()\n" + image_open, 1)
        body = body.replace(
            image_done,
            "    print(f'Input image decode and preprocess time: {time.perf_counter() - input_media_start:.4f} seconds')\n"
            + image_done,
            1,
        )
    timer = "    generation_time = generation_end - generation_start\n"
    if body.count(timer) != 1:
        raise ValueError("generation duration anchor changed")
    body = body.replace(timer, timer + "    media_start = time.perf_counter()\n", 1)
    saved = '    print(f"Saved generated video to {output_path}")\n'
    if body.count(saved) != 1:
        raise ValueError("video export anchor changed")
    body = body.replace(
        saved,
        saved + "    print(f'Video postprocess and encode time: {time.perf_counter() - media_start:.4f} seconds')\n",
        1,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(body)
    return output


def parse_metrics(log: str) -> dict:
    generation = GENERATION.search(log)
    media = MEDIA.search(log)
    input_media = INPUT_MEDIA.search(log)
    stages: dict[str, float] = {}
    for name, duration in STAGE.findall(log):
        stages[name] = stages.get(name, 0.0) + float(duration)
    return {
        "generation_s": float(generation.group(1)) if generation else None,
        "media_postprocess_encode_s": float(media.group(1)) if media else None,
        "input_media_decode_preprocess_s": float(input_media.group(1)) if input_media else None,
        "generation_plus_media_s": (
            float(generation.group(1)) + float(media.group(1))
            + (float(input_media.group(1)) if input_media else 0.0)
            if generation and media else None
        ),
        "stage_durations_s": stages,
    }


def runtime_env(vllm_root: Path, feature_env: dict[str, str], warmup: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(feature_env)
    env["ASCEND_WARMUP_PASSES"] = str(warmup)
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONPATH"] = str(vllm_root) + os.pathsep + env.get("PYTHONPATH", "")
    return env
