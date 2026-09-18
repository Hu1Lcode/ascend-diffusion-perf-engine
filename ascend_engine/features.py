"""vLLM-Omni 0.28.0 features expressed as Sol-Engine transforms.

Only flags consumed by the checked-in v0.28.0 example entrypoints belong here.
The transform plans an external runtime invocation; it does not claim that
setting an environment variable implements a new kernel.
"""

from __future__ import annotations

from techniques import Capability, ModelSpec, ModelTransform, Seam, compose


class _FlagTransform(ModelTransform):
    key: str
    value: str

    def set_env(self, ctx) -> None:
        ctx.env[self.key] = self.value


class CacheDiT(_FlagTransform):
    name = "cache_dit"
    key, value = "ASCEND_CACHE_BACKEND", "cache_dit"
    writes = frozenset({Seam.STEP_OUTPUT})
    required_capabilities = frozenset({Capability.HAS_DENOISE_STEPS})


class VaeTiling(_FlagTransform):
    name = "vae_tiling"
    key, value = "ASCEND_VAE_STRATEGY", "tiling"
    writes = frozenset({Seam.VAE_EXECUTION})
    required_capabilities = frozenset({Capability.HAS_VAE})


class VaeSlicing(_FlagTransform):
    name = "vae_slicing"
    key, value = "ASCEND_VAE_STRATEGY", "slicing"
    writes = frozenset({Seam.VAE_EXECUTION})
    required_capabilities = frozenset({Capability.HAS_VAE})


class AttentionSDPA(_FlagTransform):
    name = "attention_sdpa"
    key, value = "DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA"
    writes = frozenset({Seam.ATTENTION_BACKEND})
    required_capabilities = frozenset({Capability.HAS_ATTENTION_BACKEND_SWITCH})


class AttentionFlash(_FlagTransform):
    name = "attention_flash"
    key, value = "DIFFUSION_ATTENTION_BACKEND", "FLASH_ATTN"
    writes = frozenset({Seam.ATTENTION_BACKEND})
    required_capabilities = frozenset({Capability.HAS_ATTENTION_BACKEND_SWITCH})


FEATURES = {cls.name: cls for cls in (
    CacheDiT, VaeTiling, VaeSlicing, AttentionSDPA, AttentionFlash,
)}


def feature_plan(names: list[str], model: str) -> tuple[dict[str, str], list[str]]:
    """Type/conflict-check a recipe and render the external runtime settings."""
    if len(names) != len(set(names)):
        raise ValueError("feature names must be unique")
    unknown = set(names) - FEATURES.keys()
    if unknown:
        raise ValueError(f"unsupported features: {sorted(unknown)}")
    if "cache_dit" in names and "wan2.2" not in model.lower():
        raise ValueError("cache_dit recipe is currently limited to Wan2.2")
    if "attention_flash" in names:
        # v0.28.0's NPU platform only selects this backend when MindIE-SD works.
        # The runtime import is checked separately on the NPU host.
        pass
    spec = ModelSpec(
        name=model,
        capabilities=frozenset({Capability.HAS_DENOISE_STEPS,
                                Capability.HAS_VAE,
                                Capability.HAS_ATTENTION_BACKEND_SWITCH}),
    )
    plan = compose([FEATURES[name]() for name in names], spec)
    env: dict[str, str] = {}
    plan.apply_transforms(None, env=env)
    return env, [item.name for item in plan.transforms]
