"""MOSS-Image-SFT inference backend for an external frontend.

Design constraints from the 8GB VRAM / 16GB RAM laptop:
  * the pipeline is loaded ONCE at startup (minutes with disk offload) -- never per request
  * exactly one GPU worker: torch calls are serialised through a 1-slot executor
  * jobs are queued and polled, because a run takes minutes not seconds
  * progress is reported from inside the sampler loop, so the UI can show a real bar
  * geometry is clamped server-side so a frontend can't ask for 2K and OOM the card

Run:  uvicorn app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

import config
from util import DEFAULT_MAX_PIXELS, clamp_geometry, resolve_sampling, snap_seed

OUT_DIR = Path(os.environ.get("MOSS_OUT_DIR", "./out")).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)
PRELOAD = os.environ.get("MOSS_PRELOAD", "1") not in ("0", "false", "False")
KEEP_JOBS = int(os.environ.get("MOSS_KEEP_JOBS", "50"))

app = FastAPI(title="MOSS-Image-SFT local backend", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("MOSS_CORS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- global state -----------------------------------------------------------
_pipe = None
_load_error: str | None = None
_loading = False
_load_started = 0.0
_gpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
_gpu_lock: asyncio.Lock | None = None
_jobs: dict[str, dict[str, Any]] = {}


def _lock() -> asyncio.Lock:
    global _gpu_lock
    if _gpu_lock is None:
        _gpu_lock = asyncio.Lock()
    return _gpu_lock


def _load() -> None:
    """Blocking load; runs in a thread so /health stays responsive."""
    global _pipe, _load_error, _loading, _load_started
    _loading, _load_started = True, time.time()
    try:
        if os.environ.get("MOSS_FAKE") != "1":
            import torch  # noqa: F401  (fail here, not in a request)
        _pipe = config.build_pipeline()
        print(f"[moss] pipeline ready in {time.time() - _load_started:.0f}s (mode={config.MODE})")
    except Exception as exc:  # surfaced through /health
        _load_error = f"{type(exc).__name__}: {exc}"
        print(f"[moss] LOAD FAILED: {_load_error}")
    finally:
        _loading = False


@app.on_event("startup")
async def _startup() -> None:
    if PRELOAD:
        threading.Thread(target=_load, name="moss-load", daemon=True).start()


# --- schemas ---------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    # note: no "nsfw" here on purpose -- the local checkpoint is uncensored, and the upstream
    # negative suggestion would fight exactly the content a self-hosted box is for.
    negative_prompt: str = "worst quality, bad anatomy, watermark, jpeg artifact"
    width: int = 1024
    height: int = 1024
    # None = "whatever the loaded model wants": 8 steps / CFG 1.0 with turbo, else 50 / 3.0
    num_inference_steps: int | None = Field(None, ge=4, le=100)
    cfg_scale: float | None = Field(None, ge=1.0, le=8.0)
    sigma_shift: float = Field(5.0, ge=1.0, le=10.0)
    seed: int | None = None
    max_sequence_length: int = Field(4096, ge=64, le=4096)
    uncond_skip_layers: list[int] = [22]
    max_pixels: int = DEFAULT_MAX_PIXELS
    # which engine renders this job: "local" (this machine) or "sogni" (Supernet, needs a key)
    backend: str | None = None
    # Sogni model preset; ignored when backend == "local"
    sogni_preset: str | None = None


class JobAccepted(BaseModel):
    job_id: str
    status: str
    queue_position: int


# --- job helpers -----------------------------------------------------------
def _progress_cmd(job: dict[str, Any]):
    """tqdm-compatible wrapper over the sampler timesteps; also the cancellation hook."""

    def bar(iterable):
        total = len(iterable)
        job["total_steps"] = total
        for index, item in enumerate(iterable):
            if job.get("cancel"):
                raise RuntimeError("cancelled by user")
            job["step"] = index
            job["progress"] = round(index / max(total, 1), 3)
            yield item

    return bar


def _turbo() -> bool:
    """True only when the 8-step LoRA actually loaded -- otherwise keep full steps."""
    return bool(getattr(config, "TURBO", False)) and bool(getattr(config, "TURBO_ACTIVE", False))


def _resolve_sampling(params: GenerateRequest) -> GenerateRequest:
    """Fill in steps/CFG from the loaded model, then clamp if turbo is active.

    Turbo is a step-distilled model: 25 steps at CFG 3 through it gives burnt output, so the
    ceiling is enforced server-side rather than trusted to the caller.
    """
    turbo = _turbo()
    steps = params.num_inference_steps
    cfg = params.cfg_scale
    if steps is None:
        steps = int(getattr(config, "TURBO_STEPS", 8)) if turbo else 50
    if cfg is None:
        cfg = float(getattr(config, "TURBO_CFG", 1.0)) if turbo else 3.0
    steps, cfg = resolve_sampling(
        steps,
        cfg,
        turbo=turbo,
        turbo_steps=int(getattr(config, "TURBO_STEPS", 8)),
        turbo_cfg=float(getattr(config, "TURBO_CFG", 1.0)),
    )
    return params.model_copy(update={"num_inference_steps": steps, "cfg_scale": cfg})


def _new_job(params: GenerateRequest) -> dict[str, Any]:
    job: dict[str, Any] = {
        "job_id": uuid.uuid4().hex[:12],
        "status": "queued",
        "progress": 0.0,
        "step": 0,
        "total_steps": params.num_inference_steps,
        "seed": snap_seed(params.seed),
        "created": time.time(),
        "queue_position": sum(1 for j in _jobs.values() if j["status"] in ("queued", "running")) + 1,
        "cancel": False,
    }
    _jobs[job["job_id"]] = job
    _prune()
    return job


def _prompt_tokens(pipe, prompt: str) -> int | None:
    """Count tokens the text encoder will actually receive.

    The pipeline does `tokenizer(apply_chat_template(...), truncation=True, max_length=...)`, so a
    prompt longer than max_sequence_length is silently cut -- no error, no warning, the tail just
    disappears. Counting here is the only way to surface that, and the chat template wrapper has to
    be included or the count is off by its overhead.
    """
    tok = getattr(pipe, "tokenizer", None)
    if tok is None:
        return None
    try:
        formatted = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        ids = tok(formatted, truncation=False)["input_ids"]
        return len(ids)
    except Exception as exc:  # never let accounting break a generation
        print(f"[moss] token count unavailable: {type(exc).__name__}: {exc}")
        return None


def _select_backend(params: GenerateRequest) -> str:
    """Per-job engine. Explicit request wins; otherwise whatever this server is set up for.

    MOSS_BACKEND=sogni makes the Supernet the default, which also means the local 27GB of
    weights never have to be downloaded at all.
    """
    choice = (params.backend or os.environ.get("MOSS_BACKEND") or "local").lower()
    if choice not in ("local", "sogni"):
        raise HTTPException(422, f"backend must be 'local' or 'sogni', got {choice!r}")
    return choice


def _run_job_remote(job: dict[str, Any], params: GenerateRequest) -> None:
    """Same contract as _run_job, but the pixels are rendered on the Sogni Supernet."""
    import sogni_backend

    job.update(status="running", started=time.time(), engine="sogni")
    try:
        preset = params.sogni_preset or sogni_backend.default_preset()
        job["sogni_preset"] = preset
        image = sogni_backend.run_job(
            params,
            job=job,
            progress=lambda step, total: job.update(step=step, total_steps=total or job["total_steps"]),
        )
        if job.get("cancel"):
            raise RuntimeError("cancelled by user")
        path = OUT_DIR / f"{job['job_id']}.png"
        image.save(path)
        (OUT_DIR / f"{job['job_id']}.json").write_text(
            json.dumps(
                {
                    **params.model_dump(),
                    "seed": job["seed"],
                    "width": job["width"],
                    "height": job["height"],
                    "engine": "sogni",
                    "sogni_preset": preset,
                    "sogni_project_id": job.get("sogni_project_id"),
                    "elapsed_s": round(time.time() - job["started"], 1),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        job.update(status="done", file=str(path), progress=1.0, finished=time.time())
    except Exception as exc:
        cancelled = "cancelled" in str(exc).lower()
        job.update(
            status="cancelled" if cancelled else "error",
            error=None if cancelled else f"{type(exc).__name__}: {exc}",
            finished=time.time(),
        )
        print(f"[moss] sogni job {job['job_id']} {job['status']}: {job.get('error')}")


def _run_job(job: dict[str, Any], params: GenerateRequest) -> None:
    job.update(status="running", started=time.time(), engine="local")
    try:
        tokens = _prompt_tokens(_pipe, params.prompt)
        job["prompt_chars"] = len(params.prompt)
        job["prompt_tokens"] = tokens
        job["prompt_truncated"] = bool(tokens and tokens > params.max_sequence_length)
        if job["prompt_truncated"]:
            print(
                f"[moss] WARNING prompt is {tokens} tokens but max_sequence_length="
                f"{params.max_sequence_length}: the tail is silently dropped by the tokenizer "
                f"(with a layout JSON that means the LAST elements never reach the model)"
            )
        width, height = clamp_geometry(params.width, params.height, params.max_pixels)
        job["width"], job["height"] = width, height
        # _resolve_sampling() already filled these, this is just for the type checker
        steps = 50 if params.num_inference_steps is None else params.num_inference_steps
        cfg = 3.0 if params.cfg_scale is None else params.cfg_scale
        image = _pipe(
            prompt=params.prompt,
            negative_prompt=params.negative_prompt,
            height=height,
            width=width,
            seed=job["seed"],
            num_inference_steps=steps,
            cfg_scale=cfg,
            sigma_shift=params.sigma_shift,
            max_sequence_length=params.max_sequence_length,
            uncond_skip_layers=tuple(params.uncond_skip_layers),
            progress_bar_cmd=_progress_cmd(job),
        )
        path = OUT_DIR / f"{job['job_id']}.png"
        image.save(path)
        (OUT_DIR / f"{job['job_id']}.json").write_text(
            json.dumps(
                {
                    **params.model_dump(),
                    "seed": job["seed"],
                    "width": width,
                    "height": height,
                    "mode": config.MODE,
                    "model_id": config.MODEL_ID,
                    "elapsed_s": round(time.time() - job["started"], 1),
                    "prompt_chars": job.get("prompt_chars"),
                    "prompt_tokens": job.get("prompt_tokens"),
                    "prompt_truncated": job.get("prompt_truncated"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        job.update(status="done", file=str(path), progress=1.0, finished=time.time())
    except Exception as exc:
        cancelled = "cancelled" in str(exc).lower()
        job.update(
            status="cancelled" if cancelled else "error",
            error=None if cancelled else f"{type(exc).__name__}: {exc}",
            finished=time.time(),
        )
        print(f"[moss] job {job['job_id']} {job['status']}: {job.get('error')}")


async def _enqueue(job: dict[str, Any], params: GenerateRequest) -> None:
    loop = asyncio.get_running_loop()
    engine = job.get("engine_choice") or "local"
    async with _lock():  # one generation at a time, on an 8GB card this is law
        if engine == "sogni":
            import sogni_backend

            if not sogni_backend.enabled():
                job.update(status="error", error="SOGNI_API_KEY is not set")
                return
            job["queue_position"] = 1
            await loop.run_in_executor(_gpu_executor, _run_job_remote, job, params)
            return
        if _pipe is None:
            job.update(status="error", error=_load_error or "model not loaded")
            return
        job["queue_position"] = 1
        await loop.run_in_executor(_gpu_executor, _run_job, job, params)


# --- routes ----------------------------------------------------------------
def _backend_report() -> dict[str, Any]:
    """What can actually render right now. The UI uses this to show/hide the engine switch."""
    try:
        import sogni_backend

        sogni = {
            "configured": sogni_backend.enabled(),
            "default_preset": sogni_backend.default_preset(),
            "presets": {
                "8step": sogni_backend.PRESET_8STEP,
                "full": sogni_backend.PRESET_FULL,
            },
        }
    except Exception as exc:  # SDK missing: sogni simply isn't an option
        sogni = {"configured": False, "error": f"{type(exc).__name__}: {exc}"}
    default = (os.environ.get("MOSS_BACKEND") or "local").lower()
    return {
        "default": default,
        "local": {"ready": _pipe is not None, "error": _load_error, "mode": config.MODE},
        "sogni": sogni,
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    gpu = None
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info("cuda")
            gpu = {
                "name": torch.cuda.get_device_name(0),
                "vram_total_gib": round(total / 1024 ** 3, 2),
                "vram_free_gib": round(free / 1024 ** 3, 2),
            }
    except Exception:
        pass
    return {
        "ready": _pipe is not None,
        "loading": _loading,
        "load_seconds": round(time.time() - _load_started, 0) if _loading else None,
        "error": _load_error,
        "mode": config.MODE,
        "model_id": config.MODEL_ID,
        "backends": _backend_report(),
        "turbo": {
            "enabled": bool(getattr(config, "TURBO", False)),
            "active": _turbo(),
            "steps": getattr(config, "TURBO_STEPS", None),
            "cfg": getattr(config, "TURBO_CFG", None),
            "error": getattr(config, "TURBO_ERROR", None),
        },
        "gpu": gpu,
        "busy": _lock().locked(),
        "queued": sum(1 for j in _jobs.values() if j["status"] == "queued"),
    }


@app.post("/api/generate", response_model=JobAccepted)
async def generate(params: GenerateRequest) -> JobAccepted:
    engine = _select_backend(params)
    if engine == "local":
        if _pipe is None and _load_error:
            raise HTTPException(503, f"pipeline failed to load: {_load_error}")
        if _pipe is None and not _loading:
            raise HTTPException(503, "pipeline not loaded (started with MOSS_PRELOAD=0?)")
    else:
        import sogni_backend

        if not sogni_backend.enabled():
            raise HTTPException(
                503, "SOGNI_API_KEY is not set -- export it before starting the server"
            )

    if engine == "local":
        try:
            clamp_geometry(params.width, params.height, params.max_pixels)  # fail fast on bad geometry
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        params = _resolve_sampling(params)
    # sogni has its own documented bounds (512-2048, /16) applied in sogni_backend.resolve_params

    job = _new_job(params)
    job["engine_choice"] = engine
    asyncio.create_task(_enqueue(job, params))
    return JobAccepted(job_id=job["job_id"], status=job["status"], queue_position=job["queue_position"])


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str) -> dict[str, Any]:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    payload = {k: v for k, v in job.items() if k not in ("cancel", "started", "finished")}
    payload["elapsed_s"] = round(time.time() - (job.get("started") or job["created"]), 1)
    if job["status"] == "done":
        payload["image_url"] = f"/api/jobs/{job_id}/image"
        payload["metadata_url"] = f"/api/jobs/{job_id}/metadata"
    return payload


@app.get("/api/jobs/{job_id}/image")
async def job_image(job_id: str) -> FileResponse:
    job = _jobs.get(job_id)
    if job is None or job["status"] != "done":
        raise HTTPException(404, "no finished image for that job")
    return FileResponse(job["file"], media_type="image/png", filename=f"{job_id}.png")


@app.get("/api/jobs/{job_id}/metadata")
async def job_metadata(job_id: str) -> dict[str, Any]:
    path = OUT_DIR / f"{job_id}.json"
    if not path.exists():
        raise HTTPException(404, "no metadata for that job")
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/api/jobs/{job_id}/cancel")
async def cancel(job_id: str) -> dict[str, Any]:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    job["cancel"] = True
    if job["status"] == "queued":
        job["status"] = "cancelled"
    return {"job_id": job_id, "status": job["status"]}


@app.get("/api/sogni/account")
async def sogni_account() -> dict[str, Any]:
    """Live Spark balance + per-preset price for a 1024^2 render. Costs nothing to ask."""
    import sogni_backend

    if not sogni_backend.enabled():
        raise HTTPException(503, "SOGNI_API_KEY is not set")
    try:
        bal = await sogni_backend.balance()
    except Exception as exc:
        raise HTTPException(502, f"{type(exc).__name__}: {exc}") from exc

    spark = float(bal.get("spark", {}).get("net", 0) or 0)
    presets: dict[str, Any] = {}
    for name, preset in (("8step", sogni_backend.PRESET_8STEP), ("full", sogni_backend.PRESET_FULL)):
        try:
            est = await sogni_backend.estimate(preset, 1024, 1024, 8 if name == "8step" else 50)
            cost = float(est.get("spark", 0) or 0)
            presets[name] = {
                "model_id": preset,
                "spark": round(cost, 3),
                "usd": round(float(est.get("usd", 0) or 0), 4),
                "est_seconds": est.get("estimatedRenderSeconds"),
                "images_left": int(spark // cost) if cost else None,
            }
        except Exception as exc:
            presets[name] = {"model_id": preset, "error": f"{type(exc).__name__}: {exc}"}
    return {"spark": round(spark, 2), "balance": bal, "presets": presets}


@app.get("/api/jobs")
async def list_jobs() -> list[dict[str, Any]]:
    return [
        {"job_id": j["job_id"], "status": j["status"], "progress": j["progress"], "created": j["created"]}
        for j in sorted(_jobs.values(), key=lambda x: x["created"], reverse=True)
    ]


def _prune() -> None:
    finished = [j for j in _jobs.values() if j["status"] in ("done", "error", "cancelled")]
    if len(finished) <= KEEP_JOBS:
        return
    finished.sort(key=lambda j: j["created"])
    for job in finished[: len(finished) - KEEP_JOBS]:
        _jobs.pop(job["job_id"], None)


# --- OpenAI-shaped endpoint: the wire format index.html already speaks ---------
# index.html POSTs {model, prompt, n, size:"WxH", response_format, negative_prompt,
# num_inference_steps, guidance_scale, seed, output_watermark} and expects
# {data:[{url}], model, seed, created, usage:{output_width, output_height}}.
# Implementing that here means the playground needs no changes to talk to a local GPU.

_SIZE_RE = re.compile(r"^\s*(\d+)\s*[x\u00d7]\s*(\d+)\s*$", re.I)
COMPAT_MAX_PIXELS = int(os.environ.get("MOSS_COMPAT_MAX_PIXELS", DEFAULT_MAX_PIXELS))


class ImageGenRequest(BaseModel):
    model: str | None = "moss-image-sft-1.0"
    prompt: str = Field(..., min_length=1)
    n: int = Field(1, ge=1, le=4)
    size: str = "1024x1024"
    response_format: str = "url"
    negative_prompt: str | None = None
    # the playground always sends 25 / 3.0; with turbo these get clamped to 8 / 1.0
    num_inference_steps: int | None = Field(None, ge=4, le=100)
    guidance_scale: float | None = Field(None, ge=1.0, le=8.0)
    seed: int | None = None
    # accepted and ignored: the hosted API offers a visible watermark, this pipeline does not
    output_watermark: dict[str, Any] | None = None


@app.post("/v1/images/generations")
async def images_generations(req: ImageGenRequest, request: Request) -> Any:
    if _pipe is None:
        raise HTTPException(503, _load_error or "pipeline not loaded")

    match = _SIZE_RE.match(req.size or "")
    if not match:
        raise HTTPException(422, f"size must look like '1024x1024', got {req.size!r}")
    width, height = int(match.group(1)), int(match.group(2))

    base_url = str(request.base_url).rstrip("/")
    started = time.time()
    data: list[dict[str, str]] = []
    last: dict[str, Any] = {}

    for index in range(req.n):
        params = GenerateRequest(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt or "worst quality, bad anatomy, watermark, jpeg artifact",
            width=width,
            height=height,
            num_inference_steps=req.num_inference_steps,
            cfg_scale=req.guidance_scale,
            sigma_shift=5.0,
            max_sequence_length=4096,
            uncond_skip_layers=[22],
            seed=None if req.seed is None else req.seed + index,
            max_pixels=COMPAT_MAX_PIXELS,
        )
        params = _resolve_sampling(params)
        job = _new_job(params)
        await _enqueue(job, params)          # serialised by the GPU lock, blocks until done
        if job["status"] != "done":
            return JSONResponse(
                status_code=500,
                content={"error": {
                    "message": job.get("error") or job["status"],
                    "type": "generation_failed",
                    "code": job["status"],
                }},
            )
        last = job
        if req.response_format == "b64_json":
            data.append({"b64_json": base64.b64encode(Path(job["file"]).read_bytes()).decode()})
        else:
            data.append({"url": f"{base_url}/api/jobs/{job['job_id']}/image"})

    return {
        "created": int(last["created"]),
        "model": req.model or config.MODEL_ID,
        "seed": last["seed"],
        "data": data,
        "usage": {
            "output_width": last["width"],
            "output_height": last["height"],
            "num_inference_steps": last["total_steps"],
            "images": req.n,
            "elapsed_s": round(time.time() - started, 1),
            # prompt length as the text encoder saw it; prompt_truncated means the tail was cut
            "prompt_chars": last.get("prompt_chars"),
            "prompt_tokens": last.get("prompt_tokens"),
            "prompt_truncated": last.get("prompt_truncated"),
        },
    }


# --- serve the playground from the same origin (no CORS, no second process) ---
REPO_ROOT = Path(os.environ.get("MOSS_STATIC_DIR", str(Path(__file__).resolve().parents[1]))).resolve()
_ALLOWED_ROOT_FILES = {
    "index.html", "index-v1.html", "editor.html", "ui.png",
    "demo-layout.json", "ddlc-beach.json", "README.md",
}


@app.get("/", include_in_schema=False)
async def index_page():
    index = REPO_ROOT / "index.html"
    if not index.is_file():
        return {"service": "moss-image backend", "hint": "index.html not found; set MOSS_STATIC_DIR"}
    return FileResponse(index, media_type="text/html")


@app.get("/{filename}", include_in_schema=False)
async def root_file(filename: str):
    """Whitelist only: no dotfiles, no directories, no path traversal."""
    if filename not in _ALLOWED_ROOT_FILES:
        raise HTTPException(404, "not found")
    path = REPO_ROOT / filename
    if not path.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(path)
