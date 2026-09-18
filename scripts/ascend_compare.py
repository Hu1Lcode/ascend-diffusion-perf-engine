#!/usr/bin/env python3
"""Run the quick quality gate for two Sol-Engine run directories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ascend_engine.quality import compare_runs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--min-psnr", type=float, default=20.0)
    parser.add_argument("--min-ssim", type=float, default=0.8)
    args = parser.parse_args()
    result = compare_runs(args.baseline.resolve(), args.candidate.resolve(),
                          args.min_psnr, args.min_ssim)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "numeric_pass_pending_visual_review" else 1


if __name__ == "__main__":
    raise SystemExit(main())
