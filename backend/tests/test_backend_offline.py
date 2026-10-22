"""Offline test of the API surface + job queue with a stubbed torch/diffsynth.

No GPU, no weights, no 27GB download: a fake pipeline emulates the sampler so the
queueing, progress, cancellation, geometry clamping and file delivery can be checked
on any machine.

    python tests/test_backend_offline.py
"""

from __future__ import annotations

import os
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["MOSS_PRELOAD"] = "0"
os.environ["MOSS_OUT_DIR"] = str(ROOT / "out_test")

# ---- stub torch ------------------------------------------------------------
fake_torch = types.ModuleType("torch")
fake_torch.bfloat16 = "bfloat16"
fake_torch.float32 = "float32"
fake_torch.float8_e4m3fn = "float8_e4m3fn"
fake_torch.no_grad = lambda: (lambda fn: fn)
fake_torch.cuda = types.SimpleNamespace(
    is_available=lambda: True,
    mem_get_info=lambda *a, **k: (6 * 1024 ** 3, 8 * 1024 ** 3),
    get_device_name=lambda *a, **k: "NVIDIA GeForce RTX 4060 Laptop GPU",
    reset_peak_memory_stats=lambda *a, **k: None,
    max_memory_allocated=lambda *a, **k: 5 * 1024 ** 3,
)
sys.modules["torch"] = fake_torch


# ---- stub config (build_pipeline returns a fake sampler) -------------------
class FakePipe:
    def __init__(self, step_delay=0.02):
        self.step_delay = step_delay

    def __call__(self, *, prompt, negative_prompt, height, width, seed, num_inference_steps,
                 cfg_scale, sigma_shift, max_sequence_length, uncond_skip_layers,
                 progress_bar_cmd):
        from PIL import Image

        timesteps = list(range(num_inference_steps))
        for _ in progress_bar_cmd(timesteps):   # exercises app._progress_cmd
            time.sleep(self.step_delay)
        assert width % 16 == 0 and height % 16 == 0, "app must clamp geometry"
        assert tuple(uncond_skip_layers) == (22,)
        return Image.new("RGB", (width, height), (30, 30, 40))


fake_config = types.ModuleType("config")
fake_config.MODE = "disk"
fake_config.MODEL_ID = "openmoss/MOSS-Image-SFT"
fake_config.build_pipeline = lambda *a, **k: FakePipe()
sys.modules["config"] = fake_config

import app as backend  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from util import clamp_geometry  # noqa: E402

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILED.append(name)


def main() -> int:
    # --- geometry unit checks ---
    check("1024 stays 1024", clamp_geometry(1024, 1024) == (1024, 1024))
    w, h = clamp_geometry(1920, 1080)
    check("1920x1080 clamped under 1MP and snapped", w * h <= 1_572_864 and w % 16 == 0 and h % 16 == 0,
          f"-> {w}x{h}")
    check("odd sizes snap to /16", clamp_geometry(1000, 1000) == (1008, 1008),
          str(clamp_geometry(1000, 1000)))
    try:
        clamp_geometry(64, 64)
        check("tiny geometry rejected", False)
    except ValueError:
        check("tiny geometry rejected", True)

    backend._pipe = FakePipe()          # pretend the model is loaded

    with TestClient(backend.app) as client:
        h_resp = client.get("/health")
        check("GET /health 200", h_resp.status_code == 200)
        body = h_resp.json()
        check("/health reports ready + gpu", body["ready"] and body["gpu"]["vram_total_gib"] == 8.0,
              str(body["gpu"]))

        # --- happy path ---
        t0 = time.time()
        r = client.post("/api/generate", json={
            "prompt": "a cat holding a sign that says hello",
            "width": 1024, "height": 1024, "num_inference_steps": 8, "seed": 7,
        })
        check("POST /api/generate 200", r.status_code == 200, r.text[:200])
        job_id = r.json()["job_id"]

        seen_progress = 0.0
        for _ in range(200):
            s = client.get(f"/api/jobs/{job_id}").json()
            seen_progress = max(seen_progress, s["progress"])
            if s["status"] in ("done", "error", "cancelled"):
                break
            time.sleep(0.05)
        check("job finished", s["status"] == "done", s.get("error") or "")
        check("progress reached 1.0", s["progress"] == 1.0, f"max seen {seen_progress}")
        check("seed honoured", s["seed"] == 7)
        check("elapsed reported", s["elapsed_s"] > 0, f"{s['elapsed_s']}s")

        img = client.get(f"/api/jobs/{job_id}/image")
        check("image served as png", img.status_code == 200 and img.content[:8] == b"\x89PNG\r\n\x1a\n",
              f"{len(img.content)} bytes")
        meta = client.get(f"/api/jobs/{job_id}/metadata").json()
        check("metadata sidecar written", meta["width"] == 1024 and meta["mode"] == "disk", str(meta["elapsed_s"]))

        # --- geometry clamp enforced by the API ---
        r = client.post("/api/generate", json={"prompt": "x", "width": 4096, "height": 4096, "num_inference_steps": 4})
        check("oversize request still accepted (server clamps)", r.status_code == 200)
        big_job = r.json()["job_id"]
        for _ in range(200):
            s = client.get(f"/api/jobs/{big_job}").json()
            if s["status"] in ("done", "error", "cancelled"):
                break
            time.sleep(0.05)
        check("clamped geometry applied", s["status"] == "done" and s["width"] * s["height"] <= 1_572_864,
              f"{s.get('width')}x{s.get('height')} {s.get('error') or ''}")

        # --- 422 on bad style input ---
        r = client.post("/api/generate", json={"prompt": "x", "num_inference_steps": 1})
        check("out-of-range steps -> 422", r.status_code == 422)

        # --- cancellation: enqueue behind a slow job, cancel while running ---
        r = client.post("/api/generate", json={"prompt": "slow", "num_inference_steps": 100})
        slow_id = r.json()["job_id"]
        time.sleep(0.3)
        c = client.post(f"/api/jobs/{slow_id}/cancel").json()
        check("cancel accepted", c["status"] in ("running", "cancelled"), c["status"])
        for _ in range(200):
            s = client.get(f"/api/jobs/{slow_id}").json()
            if s["status"] in ("done", "error", "cancelled"):
                break
            time.sleep(0.05)
        check("running job stops on cancel", s["status"] == "cancelled", s["status"])

        check("jobs list populated", len(client.get("/api/jobs").json()) >= 3)
        check("unknown job -> 404", client.get("/api/jobs/deadbeef").status_code == 404)

        # --- the OpenAI-shaped endpoint index.html actually calls ---
        payload = {
            "model": "moss-image-sft-1.0",
            "prompt": "a hand-painted sign that reads hello",
            "n": 1,
            "response_format": "url",
            "negative_prompt": "worst quality",
            "size": "1776x1168",          # the playground's default: 2MP, must be clamped
            "num_inference_steps": 25,
            "guidance_scale": 3,
            "seed": 1234,
            "output_watermark": {"enabled": False},
        }
        r = client.post("/v1/images/generations", json=payload)
        check("compat: POST /v1/images/generations 200", r.status_code == 200, r.text[:300])
        body = r.json()
        url = ((body.get("data") or [{}])[0] or {}).get("url", "")
        check("compat: data[0].url is absolute", url.startswith("http://testserver/api/jobs/"), url)
        usage = body.get("usage", {})
        w, h = usage.get("output_width", 0), usage.get("output_height", 0)
        check("compat: 1776x1168 clamped to <=1MP, /16",
              w * h <= 1_048_576 and w % 16 == 0 and h % 16 == 0, f"{w}x{h}")
        check("compat: echoes model + seed + created",
              body.get("model") == "moss-image-sft-1.0" and body.get("seed") == 1234 and body.get("created"),
              f"model={body.get('model')} seed={body.get('seed')}")
        img = client.get(url.replace("http://testserver", ""))
        check("compat: returned url serves a PNG",
              img.status_code == 200 and img.content[:8] == b"\x89PNG\r\n\x1a\n", f"{len(img.content)} bytes")

        # b64 variant + n=2
        r = client.post("/v1/images/generations", json={
            "prompt": "two of them", "size": "512x512", "num_inference_steps": 4,
            "response_format": "b64_json", "n": 2,
        })
        b64 = r.json().get("data", [])
        ok_b64 = len(b64) == 2 and all(d.get("b64_json") for d in b64) and \
            __import__("base64").b64decode(b64[0]["b64_json"])[:8] == b"\x89PNG\r\n\x1a\n"
        check("compat: n=2 b64_json returns two real PNGs", ok_b64)

        r = client.post("/v1/images/generations", json={"prompt": "x", "size": "wide"})
        check("compat: bad size -> 422", r.status_code == 422)

        # --- same-origin static serving (no CORS, no second process) ---
        root = client.get("/")
        check("serves index.html at /", root.status_code == 200 and "MOSS Image Playground" in root.text)
        check("serves whitelisted sibling file", client.get("/ddlc-beach.json").status_code == 200)
        check("refuses non-whitelisted file", client.get("/start.bat").status_code == 404)
        check("refuses dotfiles", client.get("/.gitignore").status_code == 404)

    print()
    elapsed = None
    print(f"{len(FAILED)} failure(s)")
    for f in FAILED:
        print("  -", f)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
