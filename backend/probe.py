"""Smoke test: load the pipeline, generate one image, print real timings.

Run this BEFORE wiring your frontend at it -- it tells you whether the box can
actually hold the model and how slow a step really is.

    python probe.py                       # 1024x1024, 50 steps, mode from MOSS_MODE
    python probe.py --steps 12 --size 768 # cheap check

Nothing here is a benchmark harness; it is a "does it OOM and how long does it take".
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

import config
from util import clamp_geometry, snap_seed


class _Timer:
    def __init__(self) -> None:
        self.first_step: float | None = None
        self.steps: list[float] = []

    def cmd(self, iterable):
        total = len(iterable)
        for index, item in enumerate(iterable):
            t0 = time.time()
            yield item
            dt = time.time() - t0
            if index == 0:
                self.first_step = dt
            else:
                self.steps.append(dt)
            done = index + 1
            if total >= 25 and done % max(1, total // 10) == 0:
                print(f"  step {done}/{total}  {dt:.2f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a hand-painted sign that reads \"MOSS-Image\" above a wooden door, dusk light")
    ap.add_argument("--negative", default="nsfw, worst quality, bad anatomy, watermark, jpeg artifact")
    ap.add_argument("--size", type=int, default=1024, help="square side; clamped to a multiple of 16")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", default=None, help="disk | hybrid | fp8 | ram (defaults to MOSS_MODE)")
    ap.add_argument("--out", default="probe.png")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible -- this needs an NVIDIA GPU with a working driver")

    width, height = clamp_geometry(args.size, args.size)
    free, total = torch.cuda.mem_get_info("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}  VRAM free {free/1024**3:.1f}/{total/1024**3:.1f} GiB")
    print(f"geometry: {width}x{height} ({width*height/1e6:.2f} MP), {args.steps} steps, cfg {args.cfg}")

    t0 = time.time()
    pipe = config.build_pipeline(args.mode or config.MODE)
    print(f"loaded in {time.time()-t0:.1f}s (mode={args.mode or config.MODE})")

    timer = _Timer()
    t1 = time.time()
    torch.cuda.reset_peak_memory_stats()
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative,
        height=height,
        width=width,
        seed=snap_seed(args.seed),
        num_inference_steps=args.steps,
        cfg_scale=args.cfg,
        progress_bar_cmd=timer.cmd,
    )
    elapsed = time.time() - t1
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    image.save(args.out)

    steady = sum(timer.steps) / len(timer.steps) if timer.steps else float("nan")
    print("\n--- results ---")
    print(f"first step (compile/onload): {timer.first_step:.2f}s")
    print(f"steady step: {steady:.2f}s")
    print(f"total {elapsed:.1f}s  ({elapsed/args.steps:.2f}s/step avg)")
    print(f"peak VRAM allocated: {peak:.2f} GiB")
    print(f"saved: {Path(args.out).resolve()}")
    print(f"\nextrapolated 50-step cost: {steady*50/60:.1f} min at this resolution")


if __name__ == "__main__":
    main()
