"""Sequential, bounded feature experiments using Sol-Engine run bundles."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from ascend_engine.features import FEATURES, feature_plan
from ascend_engine.quality import compare_runs


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = re.compile(r"^run_dir\s*:\s*(.+)$", re.MULTILINE)


def _launch(config: Path, vllm_root: Path, model_path: Path, python_bin: Path,
            features: list[str], sample_limit: int, profile: bool) -> tuple[Path | None, str | None]:
    command = [
        str(python_bin), str(ROOT / "scripts/run.py"), str(config),
        "--set", f"VLLM_OMNI_ROOT={vllm_root}",
        "--set", f"MODEL_PATH={model_path}",
        "--set", f"PYTHON_BIN={python_bin}",
        "--set", f"ASCEND_FEATURES={','.join(features)}",
        "--set", f"ASCEND_SAMPLE_LIMIT={sample_limit}",
        "--set", f"ASCEND_PROFILE={int(profile)}",
    ]
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    match = RUN_DIR.search(proc.stdout)
    run_dir = Path(match.group(1)).resolve() if match else None
    failure = None if proc.returncode == 0 else (proc.stderr or proc.stdout)[-3000:]
    return run_dir, failure


def _benchmark(run_dir: Path) -> dict:
    return json.loads((run_dir / "outputs/benchmark.json").read_text())


def _speedup(before: float | None, after: float | None) -> float | None:
    return round(before / after, 4) if before and after else None


STAGE_LABELS = ("text_encode", "dit", "vae_encode", "vae_decode",
                "input_media_decode_preprocess", "output_media_postprocess_encode")


def _stages(bench: dict) -> dict[str, float | None]:
    result: dict[str, float | None] = {name: None for name in STAGE_LABELS}
    for name, seconds in bench.get("diagnostic", {}).get("stage_durations_s", {}).items():
        lower = name.lower()
        if "text_encoder.forward" in lower or "tokenizer.forward" in lower:
            label = "text_encode"
        elif ".diffuse" in lower or ".denoise_step" in lower:
            label = "dit"
        elif ".vae.encode" in lower:
            label = "vae_encode"
        elif ".vae.decode" in lower:
            label = "vae_decode"
        else:
            continue
        result[label] = round((result[label] or 0.0) + seconds, 4)
    runs = bench.get("runs", [])
    for label, field in (("input_media_decode_preprocess", "input_media_decode_preprocess_s"),
                         ("output_media_postprocess_encode", "media_postprocess_encode_s")):
        values = [row[field] for row in runs if row.get(field) is not None]
        if values:
            result[label] = round(statistics.median(values), 4)
    return result


def _write_report(path: Path, rows: list[dict], baseline: Path | None, task: str) -> None:
    lines = [
        f"# Ascend {task.upper()} optimization experiments", "",
        f"Baseline: `{baseline}`" if baseline else "Baseline: unavailable", "",
        "Timing: median warm generation plus input/output media processing across fixed samples.",
        "Quality: fast PSNR/SSIM smoke check; visual review is required before delivery.", "",
        "| Step | Added feature | Before (s) | After (s) | Incremental | vs. baseline | Quality | Search decision |",
        "|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for i, row in enumerate(rows):
        fmt = lambda x: "—" if x is None else f"{x:.4f}"
        ratio = lambda x: "—" if x is None else f"{x:.3f}×"
        lines.append(
            f"| {i} | {row['feature']} | {fmt(row.get('before_s'))} | "
            f"{fmt(row.get('after_s'))} | {ratio(row.get('incremental_speedup'))} | "
            f"{ratio(row.get('baseline_speedup'))} | {row.get('quality_status', '—')} | "
            f"{row['decision']} |"
        )
    lines += ["", "## Experiment details", ""]
    for row in rows:
        lines += [
            f"### {row['feature']}", "",
            f"- Run: `{row.get('run_dir')}`",
            f"- Features in this run: `{', '.join(row.get('features', [])) or 'baseline'}`",
            f"- Parent: `{row.get('parent') or 'none'}`",
            f"- Result: {row['decision']}; {row.get('reason', '')}", "",
        ]
        if row.get("stages_after"):
            lines += ["| Stage | Before (s) | After (s) |", "|---|---:|---:|"]
            for label in STAGE_LABELS:
                before = (row.get("stages_before") or {}).get(label)
                after = row["stages_after"].get(label)
                lines.append(f"| {label} | {before if before is not None else '—'} | "
                             f"{after if after is not None else '—'} |")
            lines.append("")
    lines += [
        "## Interpretation", "",
        "Speedups compare measured run bundles only. A failed or blocked quality gate never becomes a delivered configuration.",
        "The reported search winner is provisional until feature activation and side-by-side video review are recorded.",
        "Missing stage measurements are shown in each run's `outputs/benchmark.json`; they are not treated as zero.", "",
    ]
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("t2v", "i2v"), required=True)
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--feature", choices=sorted(FEATURES), action="append", default=[])
    parser.add_argument("--sample-limit", type=int, default=3)
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--baseline-run", type=Path)
    parser.add_argument("--min-psnr", type=float, default=20.0)
    parser.add_argument("--min-ssim", type=float, default=0.8)
    args = parser.parse_args(argv)
    if not 3 <= args.sample_limit <= 5:
        parser.error("sample limit must be 3–5")
    if len(set(args.feature)) != len(args.feature):
        parser.error("each feature should be listed once")
    config = ROOT / f"config/ascend_wan22/baseline_{args.task}.toml"
    run_root = ROOT / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + f"-ascend-{args.task}-sweep")
    run_root.mkdir(parents=True, exist_ok=False)
    rows: list[dict] = []
    if args.baseline_run:
        baseline = args.baseline_run.resolve()
        try:
            base_result = _benchmark(baseline)
            if base_result.get("status") != "completed" or base_result["run_spec"]["features"]:
                raise ValueError("baseline run is incomplete or has features enabled")
            if base_result["run_spec"]["task"] != args.task:
                raise ValueError("baseline task differs")
        except (OSError, KeyError, ValueError) as exc:
            parser.error(str(exc))
    else:
        baseline, error = _launch(config, args.vllm_root.resolve(), args.model_path.resolve(),
                                  args.python_bin.resolve(), [], args.sample_limit, not args.skip_profile)
        if error or baseline is None:
            rows.append({"feature": "baseline", "run_dir": str(baseline) if baseline else None,
                         "decision": "failed", "reason": error or "no run directory"})
            _write_report(run_root / "report.md", rows, baseline, args.task)
            return 1
    base_seconds = _benchmark(baseline)["total_s"]
    rows.append({"feature": "baseline", "run_dir": str(baseline), "after_s": base_seconds,
                 "baseline_speedup": 1.0, "decision": "reference", "features": [],
                 "stages_after": _stages(_benchmark(baseline))})
    accepted, parent = [], baseline
    for name in args.feature:
        candidate_features = accepted + [name]
        row = {"feature": name, "features": candidate_features[:], "parent": str(parent),
               "before_s": _benchmark(parent)["total_s"],
               "stages_before": _stages(_benchmark(parent))}
        try:
            feature_plan(candidate_features, "Wan2.2-TI2V-5B")
            run_dir, error = _launch(config, args.vllm_root.resolve(), args.model_path.resolve(),
                                     args.python_bin.resolve(), candidate_features, args.sample_limit,
                                     not args.skip_profile)
            row["run_dir"] = str(run_dir) if run_dir else None
            if error or run_dir is None:
                raise RuntimeError(error or "no run directory")
            bench = _benchmark(run_dir)
            if bench.get("status") != "completed":
                raise RuntimeError(f"candidate run status: {bench.get('status')}")
            row["after_s"] = bench["total_s"]
            row["stages_after"] = _stages(bench)
            row["incremental_speedup"] = _speedup(row["before_s"], row["after_s"])
            row["baseline_speedup"] = _speedup(base_seconds, row["after_s"])
            quality = compare_runs(baseline, run_dir, args.min_psnr, args.min_ssim)
            row["quality_status"] = quality["status"]
            if quality["status"] == "numeric_pass_pending_visual_review" and row["after_s"] < row["before_s"]:
                accepted, parent = candidate_features, run_dir
                row.update(decision="provisional_keep", reason="faster and numeric smoke gate passed")
            else:
                row.update(decision="discard", reason="quality blocked/failed or no measured speed gain")
        except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
            row.update(decision="failed", reason=str(exc))
        rows.append(row)
        _write_report(run_root / "report.md", rows, baseline, args.task)
        (run_root / "experiments.json").write_text(json.dumps(rows, indent=2) + "\n")
    result = {"baseline": str(baseline), "provisional_best": str(parent),
              "features": accepted, "visual_review_required": bool(accepted), "experiments": rows}
    (run_root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    _write_report(run_root / "report.md", rows, baseline, args.task)
    print(run_root / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
