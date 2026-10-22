"""Pipeline construction for MOSS-Image-SFT under VRAM management.

Three offload strategies, all taken from DiffSynth's own docs
(docs/en/Pipeline_Usage/VRAM_management.md + examples/moss_image/model_inference_low_vram):

  disk   - transformer + text encoder lazy-loaded from the .safetensors on disk.
           Official low-VRAM example for MOSS-Image-SFT. Uses almost no system RAM,
           needs a fast SSD. START HERE on an 8GB card / 16GB RAM laptop.
  hybrid - transformer parked in system RAM as fp8 (~5GB), text encoder read from disk.
           Faster steps than `disk`, ~7-9GB RAM used. Good on a 16GB box.
  fp8    - everything in system RAM as fp8. Fastest of the three, but the 9B text
           encoder alone is ~8.4GB fp8, so this wants 24GB+ of RAM. Not for 16GB.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # MOSS_FAKE=1 dev mode, or a fresh venv before torch is installed
    torch = None  # type: ignore[assignment]

MODEL_ID = os.environ.get("MOSS_MODEL_ID", "openmoss/MOSS-Image-SFT")
MODE = os.environ.get("MOSS_MODE", "disk").lower()

# --- optional 8-step acceleration -------------------------------------------------
# DiffSynth publishes an OPD step-distilled LoRA for MOSS-Image-SFT (50 steps -> 8).
# Official recipe: pipe.load_lora(pipe.dit, "...model-8steps.safetensors") + num_inference_steps=8,
# cfg_scale=1.0. See examples/moss_image/model_training/special/opd/.
TURBO = os.environ.get("MOSS_TURBO", "0") not in ("0", "false", "False", "")
TURBO_STEPS = int(os.environ.get("MOSS_TURBO_STEPS", "8"))
TURBO_CFG = float(os.environ.get("MOSS_TURBO_CFG", "1.0"))
TURBO_LORA_ID = os.environ.get("MOSS_TURBO_LORA_ID", "DiffSynth-Studio/MOSS-Image-SFT-TurboLoRA")
TURBO_LORA_FILE = os.environ.get("MOSS_TURBO_LORA_FILE", "model-8steps.safetensors")

# set by build_pipeline(): only clamp steps/CFG when the adapter actually loaded
TURBO_ACTIVE = False
TURBO_ERROR: str | None = None

_FILES = {
    "dit": "transformer/diffusion_pytorch_model*.safetensors",
    "te": "text_encoder/model*.safetensors",
    "vae": "vae/diffusion_pytorch_model.safetensors",
    "tok": "tokenizer/",
}


def _disk_cfg() -> dict:
    return {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": "disk",
        "onload_device": "disk",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }


def _cpu_fp8_cfg() -> dict:
    return {
        "offload_dtype": torch.float8_e4m3fn,
        "offload_device": "cpu",
        "onload_dtype": torch.float8_e4m3fn,
        "onload_device": "cpu",
        "preparing_dtype": torch.float8_e4m3fn,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }


def _cpu_bf16_cfg() -> dict:
    return {
        "offload_dtype": torch.bfloat16,
        "offload_device": "cpu",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }


def mode_configs(mode: str = MODE) -> dict:
    if torch is None:
        raise RuntimeError(
            "torch is not installed -- only MOSS_FAKE=1 works without it. "
            "Install with: uv pip install torch --index-url https://download.pytorch.org/whl/cu124"
        )
    if mode == "disk":
        dit = te = _disk_cfg()
        vae_extra = {"computation_dtype": torch.float32}
    elif mode == "hybrid":
        dit, te = _cpu_fp8_cfg(), _disk_cfg()
        vae_extra = {"computation_dtype": torch.float32}
    elif mode == "fp8":
        dit = te = _cpu_fp8_cfg()
        vae_extra = {"computation_dtype": torch.float32}
    elif mode == "ram":
        dit = te = _cpu_bf16_cfg()
        vae_extra = {"computation_dtype": torch.float32}
    else:
        raise ValueError(f"unknown MOSS_MODE={mode!r} (use disk | hybrid | fp8 | ram)")
    return {"dit": dit, "te": te, "vae_extra": vae_extra}


def vram_limit(reserve_gib: float | None = None) -> float:
    """Free-VRAM-aware limit; docs recommend 'slightly less than available'."""
    if reserve_gib is None:
        reserve_gib = float(os.environ.get("MOSS_VRAM_RESERVE", "0.7"))
    total = torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3)
    return max(0.0, total - reserve_gib)


def resolve_turbo_lora() -> str:
    """Local path to the 8-step OPD LoRA, downloading it from ModelScope on first use."""
    explicit = os.environ.get("MOSS_TURBO_LORA")
    if explicit:
        if not Path(explicit).is_file():
            raise FileNotFoundError(f"MOSS_TURBO_LORA={explicit!r} does not exist")
        return explicit

    base = Path(
        os.environ.get("MOSS_MODEL_DIR")
        or os.environ.get("DIFFSYNTH_MODEL_BASE_PATH")
        or "./models"
    )
    cached = base / TURBO_LORA_FILE
    if cached.is_file():
        return str(cached)

    print(f"[moss] downloading turbo LoRA {TURBO_LORA_ID}/{TURBO_LORA_FILE} (~621MB) ...", flush=True)
    from modelscope.hub.file_download import model_file_download

    path = model_file_download(TURBO_LORA_ID, TURBO_LORA_FILE, cache_dir=str(base / "modelscope"))
    return str(path)


def _fake_pipeline():
    """MOSS_FAKE=1: stand-in pipeline so the frontend can be developed on any machine.

    Same call signature, same progress callback, no torch and no weights. Draws a
    gradient with the prompt on it so you can tell which request produced which image.
    """
    import textwrap
    import time

    from PIL import Image, ImageDraw

    class FakePipe:
        def __call__(self, *, prompt, negative_prompt="", height=1024, width=1024, seed=None,
                     num_inference_steps=50, cfg_scale=3.0, sigma_shift=5.0,
                     max_sequence_length=4096, uncond_skip_layers=(22,), progress_bar_cmd=None):
            bar = progress_bar_cmd or (lambda it: it)
            for _ in bar(list(range(num_inference_steps))):
                time.sleep(float(os.environ.get("MOSS_FAKE_STEP_DELAY", "0.05")))
            img = Image.new("RGB", (width, height), (18, 22, 28))
            draw = ImageDraw.Draw(img)
            for y in range(height):  # cheap gradient
                t = y / max(height - 1, 1)
                draw.line([(0, y), (width, y)], fill=(int(20 + 70 * t), int(30 + 120 * t), int(60 + 140 * t)))
            text = f"FAKE PIPELINE\nseed {seed}  {width}x{height}  {num_inference_steps} steps\n\n" + \
                   "\n".join(textwrap.wrap(prompt, 46)[:18])
            draw.multiline_text((24, 24), text, fill=(240, 245, 255), spacing=6)
            return img

    return FakePipe()


def build_pipeline(mode: str = MODE, reserve_gib: float | None = None):
    if os.environ.get("MOSS_FAKE") == "1":
        print("[moss] MOSS_FAKE=1 -- serving a stand-in pipeline, no GPU used")
        return _fake_pipeline()

    from diffsynth.pipelines.moss_image import MOSSImagePipeline, ModelConfig

    cfgs = mode_configs(mode)
    # VAE stays fp32 for computation (official example: computation_dtype=torch.float32)
    vae_cfg = {**cfgs["dit"], **cfgs["vae_extra"]}

    pipe = MOSSImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id=MODEL_ID, origin_file_pattern=_FILES["dit"], **cfgs["dit"]),
            ModelConfig(model_id=MODEL_ID, origin_file_pattern=_FILES["te"], **cfgs["te"]),
            ModelConfig(model_id=MODEL_ID, origin_file_pattern=_FILES["vae"], **vae_cfg),
        ],
        tokenizer_config=ModelConfig(model_id=MODEL_ID, origin_file_pattern=_FILES["tok"]),
        vram_limit=vram_limit(reserve_gib),
    )

    if TURBO:
        global TURBO_ACTIVE, TURBO_ERROR
        try:
            lora = resolve_turbo_lora()
            pipe.load_lora(pipe.dit, lora)
            TURBO_ACTIVE = True
            print(f"[moss] TURBO on: 8-step LoRA loaded from {lora} "
                  f"(steps<={TURBO_STEPS}, cfg<={TURBO_CFG})")
        except Exception as exc:
            # A failed adapter must not take the whole server down: run at full steps instead.
            TURBO_ERROR = f"{type(exc).__name__}: {exc}"
            print(f"[moss] TURBO FAILED, falling back to full steps: {TURBO_ERROR}")
    return pipe
