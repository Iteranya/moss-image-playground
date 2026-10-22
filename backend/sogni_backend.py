"""Sogni Supernet as an alternative to running MOSS-Image-SFT on this machine.

Same job contract as the local pipeline (see app.py::_run_job): takes a GenerateRequest,
calls a `progress` callback per sampler step, returns a PIL image. That way the API surface,
the job queue, the progress reporting and the frontend need to know nothing about which
backend actually rendered the picture.

Two Sogni model presets exist and they are NOT interchangeable:

  moss_image_sft_dmd8_int8_convrot  8-step DMD distillation, ~1.25 Spark (~$0.006) @1024^2
  moss_image_sft_int8_convrot       full-step SFT, 50-100 steps required, ~8.9 Spark (~$0.044)

The DMD preset is pinned to exactly 8 steps / Euler / SGM Uniform and the negative prompt has no
effect. The full-step preset refuses to go below 50 steps -- a "25-step 50-step model" is just a
line item, the service quotes the same price, so we clamp rather than let the UI lie to you.
"""

from __future__ import annotations

import asyncio
import io
import os
import uuid

PRESET_8STEP = "moss_image_sft_dmd8_int8_convrot"
PRESET_FULL = "moss_image_sft_int8_convrot"

# Documented bounds: canvas 512-2048 per edge, multiples of 16 (moss-image model guide).
MIN_EDGE = 512
MAX_EDGE = 2048

_app_id: str | None = None


def api_key() -> str | None:
    return os.environ.get("SOGNI_API_KEY") or None


def enabled() -> bool:
    return bool(api_key())


def default_preset() -> str:
    """MOSS-Image 8-Step (DMD) is the default: same checkpoint, ~7x cheaper per image."""
    return os.environ.get("MOSS_SOGNI_PRESET") or PRESET_8STEP


def app_id() -> str:
    """Stable per installation. The SDK requires this to survive restarts -- never per-run UUIDs.

    Persisted next to the jobs so resuming a project after a restart still works.
    """
    global _app_id
    if _app_id:
        return _app_id
    path = os.environ.get("MOSS_SOGNI_APPID_FILE") or os.path.join(
        os.environ.get("MOSS_OUT_DIR", "./out"), "sogni_app_id.txt"
    )
    try:
        with open(path, encoding="utf-8") as fh:
            _app_id = fh.read().strip() or None
    except FileNotFoundError:
        _app_id = None
    if not _app_id:
        _app_id = f"moss-playground-{uuid.uuid4().hex[:12]}"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_app_id)
    return _app_id


def resolve_params(preset: str, width: int, height: int, steps: int, cfg: float) -> dict:
    """Sogni-side sampling rules, mirroring the model guide.

    8-step preset: exactly 8 steps, guidance is the 3-6 range, negative prompt does nothing.
    Full preset: 50 steps minimum. Canvas must be 512-2048 per edge and a multiple of 16.
    """
    width = max(MIN_EDGE, min(MAX_EDGE, int(width)))
    height = max(MIN_EDGE, min(MAX_EDGE, int(height)))
    width -= width % 16
    height -= height % 16

    if preset == PRESET_8STEP:
        steps = 8
    else:
        steps = max(50, min(100, int(steps)))
    guidance = max(3.0, min(6.0, float(cfg)))
    return {"width": width, "height": height, "steps": steps, "guidance": guidance}


def _project_argname() -> dict:
    return {}


async def estimate(preset: str, width: int, height: int, steps: int) -> dict:
    """Spark/USD/seconds quote for one image, from Sogni's own pricing endpoint."""
    from sogni_client import SogniClient

    key = api_key()
    if not key:
        raise RuntimeError("SOGNI_API_KEY is not set")
    resolved = resolve_params(preset, width, height, steps, 3.0)
    async with await SogniClient.create(api_key=key, app_id=app_id(), disable_socket=True) as sogni:
        return await sogni.projects.estimate_cost({
            "model": preset,
            "imageCount": 1,
            "previewCount": 0,
            "width": resolved["width"],
            "height": resolved["height"],
            "stepCount": resolved["steps"],
            "network": "fast",
            "tokenType": "spark",
        })


async def balance() -> dict:
    from sogni_client import SogniClient

    key = api_key()
    if not key:
        raise RuntimeError("SOGNI_API_KEY is not set")
    async with await SogniClient.create(api_key=key, app_id=app_id(), disable_socket=True) as sogni:
        return await sogni.account.account_balance()


def run_job(params, job: dict | None = None, progress=None):
    """Blocking entry point for the GPU/lock worker thread: render on Sogni, return a PIL image.

    `progress(step, total)` is called as the project advances so the existing job polling shows
    a real bar. Sogni reports a 0..1 fraction, which we map onto the step count the UI expects.
    """
    from PIL import Image

    key = api_key()
    if not key:
        raise RuntimeError("SOGNI_API_KEY is not set; use the local backend or add the key")

    preset = getattr(params, "sogni_preset", None) or default_preset()
    resolved = resolve_params(
        preset,
        params.width,
        params.height,
        int(params.num_inference_steps or 50),
        float(params.cfg_scale or 3.0),
    )
    if job is not None:
        # reflect what will ACTUALLY run, not what the UI asked for
        job["total_steps"] = resolved["steps"]
        job["width"], job["height"] = resolved["width"], resolved["height"]

    return asyncio.run(_generate(key, preset, params, resolved, progress, job))


async def _generate(key, preset, params, resolved, progress, job) -> "Image.Image":
    import httpx
    from PIL import Image, ImageDraw
    from sogni_client import SogniClient

    if os.environ.get("MOSS_SOGNI_DRY_RUN") == "1":
        # No network and no spend. Lets the UI path be exercised without touching Spark.
        total = resolved["steps"]
        for i in range(total + 1):
            if job is not None:
                job["progress"] = round(i / total, 3)
                job["step"] = i
            if progress:
                progress(i, total)
            await asyncio.sleep(0.05)
        img = Image.new("RGB", (resolved["width"], resolved["height"]), (20, 26, 34))
        draw = ImageDraw.Draw(img)
        note = (
            f"DRY RUN (MOSS_SOGNI_DRY_RUN=1)\n\npreset: {preset}\n"
            f"canvas: {resolved['width']}x{resolved['height']}\nsteps: {total}\n"
            f"guidance: {resolved['guidance']}\n\nprompt:\n"
            + "\n".join(params.prompt[i:i + 60] for i in range(0, min(len(params.prompt), 600), 60))
        )
        draw.multiline_text((28, 28), note, fill=(235, 240, 255), spacing=6)
        return img

    async with await SogniClient.create(
        api_key=key,
        app_id=app_id(),
        disable_socket=True,          # one-shot jobs: REST is enough, no socket to babysit
    ) as sogni:
        kwargs = {
            "type": "image",
            "model_id": preset,
            "positive_prompt": params.prompt,
            "number_of_media": 1,
            "width": resolved["width"],
            "height": resolved["height"],
            "steps": resolved["steps"],
            "guidance": resolved["guidance"],
            "token_type": "spark",
        }
        # The 8-step DMD preset ignores the negative prompt; the full preset takes it.
        if preset != PRESET_8STEP and params.negative_prompt:
            kwargs["negative_prompt"] = params.negative_prompt

        project = await sogni.projects.create(**kwargs)

        deadline_handled = False

        def on_progress(fraction: float) -> None:
            nonlocal deadline_handled
            if job is not None:
                job["progress"] = round(max(0.0, min(0.999, float(fraction))), 3)
                job["step"] = int(job["progress"] * job.get("total_steps", 1))
            if progress:
                progress(job["step"] if job else 0, job.get("total_steps", 0) if job else 0)

        try:
            project.on("progress", on_progress)
        except Exception:
            pass  # progress is a nicety; a render without a bar still finishes

        try:
            urls = await project.wait_for_completion()   # no client-side timeout on purpose
        except asyncio.CancelledError:
            await project.cancel()
            raise

        if job is not None:
            job["sogni_project_id"] = getattr(project, "id", None)

        if not urls:
            raise RuntimeError(f"Sogni project {getattr(project, 'id', '?')} returned no result URLs")

        # Results are presigned URLs that expire; pull the bytes now and hand back an image,
        # so the rest of the pipeline (save as PNG, serve from this process) is unchanged.
        async with httpx.AsyncClient(follow_redirects=True, timeout=120) as client:
            resp = await client.get(urls[0])
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
