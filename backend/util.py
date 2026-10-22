"""Pure helpers, no torch/fastapi imports (so they stay unit-testable)."""

from __future__ import annotations

# MOSS-Image wants multiples of 16 (see DiffSynth docs: "must be a multiple of 16").
DIM_STEP = 16

# 1M px is the recommended sweet spot for 1 image. 8GB VRAM laptop: do not exceed
# MAX_PIXELS_HARD. Guide says the model extrapolates up to 4M, we are nowhere near that.
DEFAULT_MAX_PIXELS = 1_048_576          # 1024x1024
MAX_PIXELS_HARD = 1_572_864             # 1280x1228-ish, absolute ceiling
MIN_PIXELS = 262_144                    # 512x512


def _snap(value: float, step: int = DIM_STEP) -> int:
    # plain nearest-multiple; int() avoids Python's banker's rounding on .5
    return max(step, int((value + step / 2) // step) * step)


def clamp_geometry(width: int, height: int, max_pixels: int = DEFAULT_MAX_PIXELS) -> tuple[int, int]:
    """Return (width, height) snapped to multiples of 16 and inside the pixel budget.

    Aspect ratio is preserved if the budget is exceeded. Raises ValueError on nonsense.
    """
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    budget = max(MIN_PIXELS, min(int(max_pixels), MAX_PIXELS_HARD))
    width, height = _snap(width), _snap(height)

    pixels = width * height
    if pixels > budget:
        scale = (budget / pixels) ** 0.5
        width, height = _snap(width * scale), _snap(height * scale)
        # snapping can round back over the budget; walk down until it fits
        while width * height > budget and width > DIM_STEP and height > DIM_STEP:
            if width >= height:
                width -= DIM_STEP
            else:
                height -= DIM_STEP

    if width * height < MIN_PIXELS:
        raise ValueError(f"geometry too small; need at least {MIN_PIXELS} pixels ({MIN_PIXELS**0.5:.0f}^2)")
    return width, height


def resolve_sampling(
    steps: int,
    cfg: float,
    turbo: bool = False,
    turbo_steps: int = 8,
    turbo_cfg: float = 1.0,
) -> tuple[int, float]:
    """Clamp steps/CFG when the OPD 8-step distillation LoRA is loaded.

    A step-distilled model is trained at few steps with no classifier-free guidance; running it
    at 25 steps / CFG 3 does not just waste time, it produces blown-out garbage. The official
    recipe is `num_inference_steps=8, cfg_scale=1.0`, so that is the ceiling here.
    """
    steps = int(steps)
    cfg = float(cfg)
    if turbo:
        steps = min(steps, int(turbo_steps))
        cfg = min(cfg, float(turbo_cfg))
    return steps, cfg


def snap_seed(seed) -> int:
    """None -> random 32-bit; anything else coerced to int."""
    if seed is None or seed == "":
        import random
        return random.randint(0, 2**31 - 1)
    return int(seed) % (2**31)
