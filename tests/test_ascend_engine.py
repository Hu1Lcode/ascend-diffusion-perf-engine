"""CPU-only contract checks for the Ascend vLLM-Omni adapter."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ascend_engine.features import feature_plan
from ascend_engine.quality import compare_runs
from ascend_engine.run_manifest import _command, _samples
from ascend_engine.sweep import _stages
from ascend_engine.vllm_adapter import parse_metrics, render_entrypoint


ROOT = Path(__file__).resolve().parents[1]


class FeatureTests(unittest.TestCase):
    def test_composition_rejects_two_attention_backends(self):
        with self.assertRaisesRegex(ValueError, "attention_backend"):
            feature_plan(["attention_sdpa", "attention_flash"], "Wan2.2-TI2V-5B")

    def test_composition_rejects_two_vae_strategies(self):
        with self.assertRaisesRegex(ValueError, "vae_execution"):
            feature_plan(["vae_tiling", "vae_slicing"], "Wan2.2-TI2V-5B")

    def test_cache_limited_to_supported_model_family(self):
        with self.assertRaisesRegex(ValueError, "Wan2.2"):
            feature_plan(["cache_dit"], "other-model")


class WorkloadTests(unittest.TestCase):
    def test_pinned_t2v_and_i2v_entrypoints_can_be_instrumented(self):
        vllm = ROOT.parent / "vllm-omni-v0.28.0"
        if not vllm.is_dir():
            self.skipTest("pinned vLLM-Omni source is not checked out next to this repo")
        with tempfile.TemporaryDirectory() as tmp:
            for task in ("t2v", "i2v"):
                output = Path(tmp) / f"{task}.py"
                render_entrypoint(vllm, task, output)
                compile(output.read_text(), str(output), "exec")
                self.assertIn("random.seed(args.seed)", output.read_text())
                if task == "i2v":
                    self.assertIn("Input image decode and preprocess time:", output.read_text())

    def test_frozen_i2v_inputs_have_hashes(self):
        manifest = ROOT / "evals/ascend_samples/i2v.json"
        samples = _samples(manifest, "i2v", 3, ROOT)
        self.assertEqual(len(samples), 3)
        self.assertTrue(all(len(row["image_sha256"]) == 64 for row in samples))

    def test_command_pins_seed_input_and_flags(self):
        sample = {"id": "case", "seed": 7, "prompt": "motion", "image": "/x/input.png"}
        settings = {"height": "480", "width": "832", "frames": "81", "steps": "50",
                    "fps": "16", "guidance": "5.0", "cache_backend": "cache_dit",
                    "vae_strategy": "tiling"}
        command = _command(Path("example.py"), Path("/model"), "i2v", sample,
                           Path("/out.mp4"), settings, False)
        self.assertEqual(command[command.index("--seed") + 1], "7")
        self.assertEqual(command[command.index("--image") + 1], "/x/input.png")
        self.assertIn("--enable-cache-dit-summary", command)
        self.assertIn("--vae-use-tiling", command)

    def test_missing_media_time_cannot_become_end_to_end_speedup(self):
        parsed = parse_metrics("Total generation time: 3.2000 seconds\n")
        self.assertEqual(parsed["generation_s"], 3.2)
        self.assertIsNone(parsed["generation_plus_media_s"])

    def test_stage_profile_parser_aggregates_repeated_steps(self):
        log = ("[DiffusionPipelineProfiler] Wan.diffuse took 1.5s\n"
               "[DiffusionPipelineProfiler] Wan.diffuse took 2.0s\n"
               "Video postprocess and encode time: 0.8 seconds\n"
               "Total generation time: 4.0 seconds\n")
        parsed = parse_metrics(log)
        self.assertEqual(parsed["stage_durations_s"]["Wan.diffuse"], 3.5)
        self.assertEqual(parsed["generation_plus_media_s"], 4.8)

    def test_i2v_total_includes_input_preprocess(self):
        parsed = parse_metrics("Input image decode and preprocess time: 0.3 seconds\n"
                               "Total generation time: 4.0 seconds\n"
                               "Video postprocess and encode time: 0.8 seconds\n")
        self.assertEqual(parsed["generation_plus_media_s"], 5.1)

    def test_stage_summary_keeps_missing_stages_unknown(self):
        bench = {"runs": [{"media_postprocess_encode_s": 0.8}],
                 "diagnostic": {"stage_durations_s": {"Wan.diffuse": 2.0}}}
        stages = _stages(bench)
        self.assertEqual(stages["dit"], 2.0)
        self.assertEqual(stages["output_media_postprocess_encode"], 0.8)
        self.assertIsNone(stages["vae_decode"])

    def test_quality_refuses_different_sample_suites(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / "base", root / "trial"]
            for path, digest in zip(paths, ("a", "b")):
                (path / "outputs").mkdir(parents=True)
                (path / "outputs/benchmark.json").write_text(json.dumps({
                    "status": "completed", "runs": [],
                    "run_spec": {"task": "t2v", "model": "/model", "samples_sha256": digest,
                                 "model_fingerprint": "same", "settings": {},
                                 "entrypoint_sha256": "same", "runtime": "/vllm",
                                 "ascend_runtime": {}, "samples": []},
                }))
            with self.assertRaisesRegex(ValueError, "samples_sha256"):
                compare_runs(*paths)


if __name__ == "__main__":
    unittest.main()
