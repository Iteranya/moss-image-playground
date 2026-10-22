# Local GPU backend for the playground

Runs **MOSS-Image-SFT** on your own machine via [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
and serves the same wire format the playground already speaks (`POST /v1/images/generations`).
Tuned for **8GB VRAM / 16GB RAM** — an RTX 4060 laptop.

It also serves `index.html` from the repo root, so the page and the API share one origin: no CORS,
no relay, no second process.

```
config.py         the VRAM-management strategy (this is the interesting file)
app.py            HTTP API: OpenAI-shaped endpoint, async job API, static hosting
util.py           geometry clamping (multiples of 16, pixel budget)
probe.py          one-shot CLI: load, generate, print real s/step + peak VRAM
static/dev.html   debug page for the async job API (progress bar, cancel)
tests/            offline test of the whole API surface, no GPU required
```

## Why self-host at all

The hosted `api.mosi.cn` endpoint is free, so throughput isn't the reason to run this. The reason is
that the weights here are the **raw MOSS-Image-SFT checkpoint** with no provider-side filter, and
nothing between you and the output. Note the default negative prompt is
`worst quality, bad anatomy, watermark, jpeg artifact` — the upstream guide suggests adding `nsfw`
to it, and this backend deliberately does not, because politely refusing to draw things is the exact
behaviour you are self-hosting to escape. Pass your own `negative_prompt` if you want it back.

## Why this works on 16GB of RAM

MOSS-Image-SFT is 5B DiT + **Qwen3.5 9B text encoder** + FLUX.2 VAE. In bf16 that's ~27GB of
weights, so "just load it" is off the table. DiffSynth's **disk offload** lazy-loads each layer
straight out of the `.safetensors` file when its forward runs — weights never have to sit in
system RAM. That is what makes a 16GB machine viable; the trade-off is that you want the weights
on a fast NVMe SSD, not a spinning disk or a slow external drive.

Other modes are wired up in `config.py` if you have more RAM than you think:

| `MOSS_MODE` | DiT | text encoder | RAM use | when |
|---|---|---|---|---|
| `disk` (default) | mmap from disk | mmap from disk | ~2-4GB | 16GB RAM. Official low-VRAM recipe. |
| `hybrid` | fp8 in RAM (~5GB) | mmap from disk | ~7-9GB | 24GB RAM, faster steps |
| `fp8` | fp8 in RAM | fp8 in RAM (~8.4GB) | ~14GB+ | 32GB RAM only |
| `ram` | bf16 in RAM (~10GB) | bf16 in RAM (~17GB) | 27GB+ | no |

## Install

Needs Windows 10/11, a current NVIDIA driver, and ~40GB free on the drive holding the weights.
Use Python 3.12 — torch has no 3.14 wheels.

```powershell
uv venv --python 3.12 .venv
.venv\Scripts\activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
git clone https://github.com/modelscope/DiffSynth-Studio.git ..\DiffSynth-Studio
uv pip install -e ..\DiffSynth-Studio
uv pip install -r requirements.txt
```

Optional, for the weight cache:

```powershell
$env:DIFFSYNTH_MODEL_BASE_PATH = "D:\models"
$env:DIFFSYNTH_DOWNLOAD_SOURCE  = "modelscope"   # or "huggingface"
```

`huggingface` also works, but then the ids change — set `MOSS_MODEL_ID=OpenMOSS-Team/MOSS-Image-SFT`.

## Run

```powershell
# develop the frontend with no GPU, no weights, no torch at all:
$env:MOSS_FAKE="1"; uvicorn app:app --port 8000

# the real thing (loads once at startup; expect minutes with disk offload)
uvicorn app:app --port 8000

# same, with the 8-step acceleration LoRA (~4-6x less sampling compute):
$env:MOSS_TURBO="1"; uvicorn app:app --port 8000
```

Then open <http://127.0.0.1:8000/>. The page probes `/health`, finds the backend, fills its
**Endpoint** field with `/v1/images/generations` by itself, and stops demanding an API key.

Measure the hardware before trusting any default:

```powershell
python probe.py --size 1024 --steps 12
```

First run downloads ~27GB. The script prints first-step time, steady-state s/step, total time,
peak VRAM, and an extrapolated 50-step cost. If it OOMs, drop to `--size 896` or `768`.

## Faster: the 8-step Turbo LoRA

DiffSynth ships an OPD step-distilled adapter for MOSS-Image-SFT — 621MB, applied to the same
pipeline: `DiffSynth-Studio/MOSS-Image-SFT-TurboLoRA / model-8steps.safetensors`. Official recipe
is `pipe.load_lora(pipe.dit, …)` + `num_inference_steps=8, cfg_scale=1.0`.

`MOSS_TURBO=1` makes the backend do that at startup:

- downloads the adapter on first run (into `DIFFSYNTH_MODEL_BASE_PATH`/`MOSS_MODEL_DIR`, default `./models`)
- `pipe.load_lora(pipe.dit, path)` after the pipeline is built
- **clamps steps to 8 and CFG to 1.0** server-side, because a distilled model at 25 steps / CFG 3
  does not just waste time, it produces burnt output. The playground's 25/3 gets rewritten to 8/1.
- `/health` reports `turbo: {enabled, active, steps, cfg, error}`

Why it is ~4-6x rather than 3x: 25→8 steps is 3.1x, and CFG 1.0 removes the negative branch
entirely (one forward per step instead of pos + SLG-halved neg).

| | steps | CFG | forwards/step | 1024² estimate |
|---|---|---|---|---|
| default | 25 | 3.0 | ~1.5 | 70–125 s |
| turbo | 8 | 1.0 | 1 | 18–40 s |

Caveats, honestly:

- `active: false` in `/health` means the adapter did **not** load and steps stay at full — a failed
  adapter never silently degrades you to 8-step output. The reason lands in `turbo.error`.
- `load_lora` is documented against a **non-offloaded** DiT (the official validate script builds
  the pipeline without `vram_config`). Loading it on top of `MOSS_MODE=disk` is the one thing here
  I could not test without the weights. If it errors, try `MOSS_MODE=hybrid` first.
- Distilled output is not identical to CFG-3 output: slightly flatter composition, softer
  prompt adherence. Compare with `probe.py --steps 8 --cfg 1.0` before committing to it.
- At 8 steps the **9B text encoder becomes the bottleneck**, so `max_sequence_length` matters more
  than it did.

## Prompt size: the two ways it bites

The playground builds its prompt as a layout JSON with one `identity` per box, so prompts are long
by design — the shipped `blue-archive` sample is **8,451 characters**, `ddlc-beach.json` is 3,774.

**1. It costs real seconds, once per prompt.** The 9B text encoder does one prefill pass over the
whole prompt; that pass is wall-clock, not per-step. Encoding is ~linear in tokens. At 8 steps
turbo the encoder can be *most* of the wait: 8 steps × ~1.4–2.3 s = 11–18 s of sampling versus
20–35 s of encoding for an 8.4k-char prompt. `max_sequence_length` is the knob that shortens it.

**2. Above `max_sequence_length` it truncates silently.** The pipeline calls
`tokenizer(..., truncation=True, max_length=max_sequence_length)` — no exception, no log, the tail
just stops existing. Because a layout JSON carries `gen_brief` first and the elements after it, what
gets cut is the **last boxes** — you get a picture missing the characters you spent the longest
describing, and it looks like the model ignoring you rather than a cut prompt.

So the backend counts it and tells you:

| where | fields |
|---|---|
| `GET /api/jobs/{id}` (set before sampling starts) | `prompt_chars`, `prompt_tokens`, `prompt_truncated` |
| `/v1/images/generations` → `usage` | same three |
| `metadata-<id>.json` sidecar | same three |
| server log | a `WARNING` line naming both numbers when it will be cut |

The count reproduces what the encoder actually sees, chat-template wrapper included
(`apply_chat_template(..., add_generation_prompt=True, enable_thinking=True)`), so it is not a
char/4 guess when a real pipeline is loaded.

Practical targets for this box: keep prompts **under ~2k tokens** (~7k chars), keep per-box
`identity` to a dozen words instead of a paragraph, and put shared style/lighting in `gen_brief`
rather than repeating it per box. Raising `max_sequence_length` above 4096 buys nothing — that is
the model's trained ceiling.

## API

Two surfaces, one engine.

**OpenAI-shaped** (what `index.html` calls) — `POST /v1/images/generations`, blocking:

```json
{ "model": "moss-image-sft-1.0", "prompt": "...", "n": 1, "size": "1776x1168",
  "response_format": "url", "negative_prompt": "...", "num_inference_steps": 25,
  "guidance_scale": 3, "seed": 1234, "output_watermark": {"enabled": false} }
```

→ `{created, model, seed, data:[{url}|{b64_json}], usage:{output_width, output_height, …}}`.
`output_watermark` is accepted and ignored — this pipeline has no watermark head.

**Async job API** (better for a custom frontend):

| route | what |
|---|---|
| `GET /health` | `ready`, `loading`, `error`, `mode`, `gpu.vram_free_gib`, `busy`, `queued` |
| `POST /api/generate` | body → `{job_id, status, queue_position}`. Returns immediately. |
| `GET /api/jobs/{id}` | `status` (`queued\|running\|done\|error\|cancelled`), `step`, `total_steps`, `progress` 0-1, `elapsed_s`, `width/height`, `seed`, `image_url` |
| `GET /api/jobs/{id}/image` | the PNG |
| `GET /api/jobs/{id}/metadata` | exact params + timings for that image |
| `POST /api/jobs/{id}/cancel` | stops a running job at the next sampler step, or a queued one |
| `GET /api/jobs` | recent jobs |

`POST /api/generate` body:

```json
{
  "prompt": "a hand-painted sign reading \"MOSS-Image\" above a wooden door, dusk light",
  "negative_prompt": "nsfw, worst quality, bad anatomy, watermark, jpeg artifact",
  "width": 1024, "height": 1024,
  "num_inference_steps": 50,
  "cfg_scale": 3.0,
  "sigma_shift": 5.0,
  "seed": null,
  "max_sequence_length": 4096,
  "uncond_skip_layers": [22],
  "max_pixels": 1048576
}
```

Server-side rules, because a frontend should not be able to OOM the card:

- width/height snap to multiples of 16 (the model errors otherwise)
- total pixels are clamped to `max_pixels`, capped at 1.57M, aspect ratio preserved
- one generation at a time; extra requests queue (an 8GB card cannot do two)
- `uncond_skip_layers: [22]` is the sampler's SLG default and **halves** the negative-branch
  compute. Pass `[]` only if you want the slow, maximally realistic variant.

| env | default | notes |
|---|---|---|
| `MOSS_MODE` | `disk` | see table above |
| `MOSS_FAKE` | unset | `1` = stand-in pipeline, no torch needed |
| `MOSS_TURBO` | unset | `1` = load the 8-step OPD LoRA, clamp steps≤8 / CFG≤1.0 |
| `MOSS_TURBO_STEPS` / `MOSS_TURBO_CFG` | `8` / `1.0` | turbo ceilings |
| `MOSS_TURBO_LORA` | unset | path to a local `model-8steps.safetensors`, skips the download |
| `MOSS_TURBO_LORA_ID` / `_FILE` | `DiffSynth-Studio/MOSS-Image-SFT-TurboLoRA` / `model-8steps.safetensors` | ModelScope source |
| `MOSS_MODEL_ID` | `openmoss/MOSS-Image-SFT` | ModelScope id; HF id is `OpenMOSS-Team/MOSS-Image-SFT` |
| `MOSS_VRAM_RESERVE` | `0.7` | GiB kept free; docs say set slightly under total (8GB card → ~7.3) |
| `MOSS_COMPAT_MAX_PIXELS` | `1048576` | pixel budget for the `/v1` endpoint |
| `MOSS_OUT_DIR` | `./out` | PNG + JSON sidecar per job |
| `MOSS_STATIC_DIR` | repo root | where `index.html` is served from |
| `MOSS_CORS` | `*` | comma list of allowed origins for your frontend |
| `MOSS_PRELOAD` | `1` | `0` = load lazily on first request |
| `MOSS_KEEP_JOBS` | `50` | job entries kept in memory |

## Tuning for an 8GB card

- **1MP, 1024×1024.** 1.5M is the ceiling on this card, 2K is not happening.
- **Steps:** guide says ≥50 for best results. 30 is the usual quality/time compromise; the
  playground's default of 25 is fine while iterating on a layout. Official settings are
  50 steps / CFG 3.0 / shift 5.0 / Euler — the pipeline already does Euler over the shift-5 grid.
- **Short prompts:** lower `max_sequence_length` (1024 is plenty under ~700 words). The text
  encoder is the 9B one, so this is real time saved per prompt.
- **CFG 3.0** is right for long prompts; raise toward 4-5 for short ones.
- Keep the output folder off a slow disk if the weights are mmap'd from an SSD.

## Troubleshooting

- `no CUDA device visible` → driver/torch mismatch; reinstall torch from the cu12x index.
- First step is 10-100× the others → weight onload + kernel autotune. Normal. It is why the API
  keeps the pipeline resident instead of reloading per request.
- OOM at 1024 → `MOSS_MODE=disk` (already default), lower `MOSS_VRAM_RESERVE`, close the browser,
  then drop resolution.
- `/health` says `pipeline failed to load` → that message is the real exception; usually a partial
  download. Delete `./models` and let it re-fetch.
- Disk offload requires `.safetensors`; `.bin`/`.ckpt` weights won't stream.

## What is verified

`python tests/test_backend_offline.py` stubs torch + the pipeline and exercises the real FastAPI
app: geometry clamping, queueing, progress reporting, seed handling, PNG delivery, metadata
sidecar, 422 on bad input, mid-run cancellation, the `/v1/images/generations` compat shape
(the playground's 1776×1168 default arriving as 1248×832), `b64_json`, whitelisted static
hosting, and turbo clamping in both directions (25/3 → 8/1 when active, untouched when off).
**42/42 pass.**

Also checked by hand, since they need a real server: `MOSS_FAKE=1 MOSS_TURBO=1` boots and reports
`turbo: {enabled: true, active: false}` — i.e. "asked for, but no adapter loaded, so do not clamp".
That is the failure mode that matters: a missing or broken LoRA must never silently give you
8-step output at CFG 1.

What is **not** verified: actual inference. There is no GPU and no 27GB of weights in the dev
environment, so `probe.py` and the real model path are untested until they run on the GPU box.
Run `probe.py` first — cheapest way to find out whether the disk-offload plan holds up.
