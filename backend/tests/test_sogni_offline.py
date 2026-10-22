"""Sogni engine tests with the SDK stubbed out: no network, no key, no GPU."""

from __future__ import annotations

import io
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image


class FakeProject:
    id = "proj-abc123"

    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.handlers = {}

    def on(self, event, fn):
        self.handlers[event] = fn

    async def wait_for_completion(self, timeout=None):
        for frac in (0.0, 0.5, 1.0):
            if "progress" in self.handlers:
                self.handlers["progress"](frac)
        return ["https://cdn.example/x.png"]

    async def cancel(self):
        self.cancelled = True


class FakeSogni:
    def __init__(self):
        self.created = []
        self.projects = types.SimpleNamespace(
            create=self._create, estimate_cost=self._estimate,
        )

    async def _create(self, **kwargs):
        p = FakeProject(kwargs)
        self.created.append(p)
        return p

    async def _estimate(self, payload):
        return {"spark": "1.2514", "usd": "0.0062572", "estimatedRenderSeconds": 3.92}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class FakeClientClass:
    instance = None

    @classmethod
    async def create(cls, **kwargs):
        cls.instance = FakeSogni()
        return cls.instance


# stub the SDK module before importing our backend
sdk = types.ModuleType("sogni_client")
sdk.SogniClient = FakeClientClass
sys.modules["sogni_client"] = sdk

# httpx fetch of the result URL -> a small PNG
import httpx


class FakeResp:
    content = b""

    def raise_for_status(self):
        return None


buf = io.BytesIO()
Image.new("RGB", (64, 48), (10, 20, 30)).save(buf, format="PNG")
FakeResp.content = buf.getvalue()


class FakeAsyncClient:
    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url):
        return FakeResp()


real_httpx = httpx
fake_httpx = types.ModuleType("httpx")
fake_httpx.AsyncClient = FakeAsyncClient
sys.modules["httpx"] = fake_httpx

import os

os.environ["SOGNI_API_KEY"] = "test-key-not-real"
os.environ["MOSS_OUT_DIR"] = str(ROOT / "out_test")

import sogni_backend as sb

FAILED: list[str] = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILED.append(name)


def main() -> int:
    check("enabled() true with a key", sb.enabled() is True)
    check("8-step preset is the default", sb.default_preset() == sb.PRESET_8STEP, sb.default_preset())

    # documented bounds from the moss-image model guide
    r = sb.resolve_params(sb.PRESET_8STEP, 1000, 1000, 25, 3.0)
    check("8-step pins steps to 8", r["steps"] == 8, str(r))
    check("canvas snaps to /16", r["width"] % 16 == 0 and r["height"] % 16 == 0, f"{r['width']}x{r['height']}")
    check("canvas clamps to 512-2048", sb.resolve_params(sb.PRESET_FULL, 4096, 256, 50, 3.0)["width"] == 2048)

    r2 = sb.resolve_params(sb.PRESET_FULL, 1024, 1024, 25, 3.0)
    check("full preset clamps 25 steps up to 50", r2["steps"] == 50, str(r2))
    check("guidance clamps into 3-6", sb.resolve_params(sb.PRESET_FULL, 1024, 1024, 50, 99)["guidance"] == 6.0)

    job = {"total_steps": 50, "step": 0, "width": 1024, "height": 1024}

    class P:
        prompt = "a cobalt mug that reads MORNING TIDE"
        negative_prompt = "worst quality"
        width = 1000
        height = 1000
        num_inference_steps = 25
        cfg_scale = 3.0
        sogni_preset = None

    os.environ.pop("MOSS_SOGNI_DRY_RUN", None)   # these cases exercise the SDK path
    img = sb.run_job(P(), job=job)
    check("run_job returns a PIL image", isinstance(img, Image.Image), f"{img.size}")
    proj = FakeClientClass.instance.created[0].kwargs
    check("project uses the 8-step model id", proj["model_id"] == sb.PRESET_8STEP, proj["model_id"])
    check("project has 8 steps", proj["steps"] == 8, str(proj["steps"]))
    check("8-step omits negative_prompt (it is ignored server-side)",
          "negative_prompt" not in proj, str(list(proj)))
    check("job reflects the resolved canvas, not the request",
          (job["width"], job["height"]) == (992, 992), f"{job['width']}x{job['height']}")
    check("progress reached 1.0", job["progress"] >= 0.999, str(job["progress"]))
    check("project id recorded", job.get("sogni_project_id") == "proj-abc123")

    # full preset keeps the negative prompt
    class P2(P):
        sogni_preset = sb.PRESET_FULL
        num_inference_steps = 50

    FakeClientClass.instance.created.clear()
    sb.run_job(P2(), job={"total_steps": 50, "step": 0})
    proj2 = FakeClientClass.instance.created[0].kwargs
    check("full preset sends negative_prompt", proj2.get("negative_prompt") == "worst quality")
    check("full preset uses the full model id", proj2["model_id"] == sb.PRESET_FULL)

    # dry-run mode: no SDK call at all, still a real image + full progress
    FakeClientClass.instance = None
    os.environ["MOSS_SOGNI_DRY_RUN"] = "1"
    job3 = {"total_steps": 0, "step": 0}
    img3 = sb.run_job(P(), job=job3)
    check("dry run renders without touching the SDK",
          FakeClientClass.instance is None and isinstance(img3, Image.Image), f"{img3.size}")
    check("dry run reports full progress", job3["progress"] == 1.0 and job3["step"] == 8, str(job3))
    os.environ.pop("MOSS_SOGNI_DRY_RUN", None)

    print(f"\n{len(FAILED)} failure(s)")
    for f in FAILED:
        print("  -", f)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
