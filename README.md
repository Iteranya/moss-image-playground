# MOSS Image Playground

A single-page layout editor for [MOSS-Image](https://platform.mosi.cn/) text-to-image generation
(`POST https://api.mosi.cn/v1/images/generations`).

Draw boxes on a canvas, give each box its own prompt, add a global scene brief, hit **Generate**.
The page builds the layout JSON for you and calls the API directly from your browser — no build
step, no dependencies.

Two backends, same page:

| | needs | notes |
|---|---|---|
| **Hosted API** (`api.mosi.cn`) | an API key | what the Endpoint field is empty for |
| **Your own GPU** ([`backend/`](backend/README.md)) | NVIDIA card + DiffSynth | MOSS-Image-SFT locally, no key, no 30s CDN cap |

## Use it

Open the deployed page and paste your own MOSS API key into the key field.

- The key is stored in **your browser's localStorage only**. It is never baked into the page,
  never sent anywhere except `api.mosi.cn`, and never committed to this repo.
- Nothing on this site costs anyone anything until a key is pasted.

## Features

- **Draw** a box anywhere on the canvas, **move** it by dragging, **resize** from the corner handle,
  `Del` to delete, `Esc` to deselect, `Ctrl+Enter` to generate.
- Per box: `type` (`obj` / `panel` / `text`), `salience` (`primary` / `secondary` / `background`),
  an identity prompt, and a verbatim string for `text` boxes.
- Boxes are stored **normalized**, so changing the canvas size rescales the whole layout instead of
  breaking it. Importing a layout snaps the canvas to the size its coordinates imply.
- **JSON tab** is the source of truth — hand-edit it, then `JSON → boxes` to get draggable boxes back.
  Both prompt schemas (`scene_type`/`gen_brief`/`elements` and `summary`/`items`) are accepted.
- Live canvas validation (pixel cap, multiple of 16) and a generation-time estimate, because the CDN
  in front of the API kills any request at **~30 s** (HTTP 524). Oversized jobs are retried once at
  half the steps instead of failing silently.

## Example

`ddlc-beach.json` is a four-character beach ensemble prompt — load it with `JSON → boxes`.

## Run locally

Any static file server works:

```bash
python -m http.server 8765
```

`file://` will **not** work: the browser sends `Origin: null` and the API rejects it with 403.
Serve it over http instead.

## Run it on your own GPU

`backend/` hosts MOSS-Image-SFT locally (DiffSynth-Studio, tuned for 8GB VRAM) and serves this page
on the same origin, so there is no CORS, no relay and no API key:

```powershell
cd backend
uv venv --python 3.12 .venv ; .venv\Scripts\activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt
$env:MOSS_FAKE="1"; uvicorn app:app --port 8000    # frontend dev, no GPU/weights needed
uvicorn app:app --port 8000                        # real inference
$env:MOSS_TURBO="1"; uvicorn app:app --port 8000   # + 8-step Turbo LoRA (~4-6x faster)
```

With `MOSS_TURBO=1` the backend loads DiffSynth's OPD 8-step LoRA and clamps requests to 8 steps at
CFG 1.0 — the playground's 25/3 gets rewritten, because a step-distilled model at high CFG burns out.

Open <http://127.0.0.1:8000/>. On boot the page probes `/health`; if a local backend answers it
fills the **Endpoint** field with `/v1/images/generations` (the hint next to it reads **local GPU**),
skips the API-key requirement, and drops the ~30s CDN cap from the time estimate. Requests are
clamped to a 1MP budget server-side, so the page's default 1776×1168 comes back as 1248×832 on an
8GB card. See [`backend/README.md`](backend/README.md).

## Hosting caveat (important)

The page itself is static and hosts anywhere, **but the MOSS API refuses non-localhost origins.**
Measured on the CDN in front of `api.mosi.cn`:

| `Origin` sent | preflight (OPTIONS) |
|---|---|
| `http://localhost:3000` | `204` + `access-control-allow-origin` echoed |
| `http://127.0.0.1:8765` | `204` + `access-control-allow-origin` echoed |
| `https://example.com` | `403`, no CORS headers |
| `https://iteranya.github.io` | `403`, no CORS headers |

A `POST` from a hosted origin does return `200` server-side, but with no `access-control-allow-*`
headers — the browser blocks the response, and the request still consumes quota. So this copy is a
read-only demo of the UI; generation only works when served from `localhost`.

Bridging that gap needs a relay you control (e.g. a Cloudflare Worker that forwards to
`/v1/images/generations` and adds the CORS headers), with the page's endpoint pointed at it.

## Self-hosting (VPS)

Serve the page and the API from one origin and there is no CORS to fight:

```bash
cd relay
STATIC_DIR=.. PORT=9000 python3 relay/server.py
```

Then put TLS in front (Caddy: `play.example.com { reverse_proxy 127.0.0.1:9000 }`), open
`https://play.example.com/`, and set the **Endpoint** field to the relative path
`/v1/images/generations`. Python 3.8+, no dependencies.

Details and the standalone-relay variant: [`relay/README.md`](relay/README.md).

## Batch queue (render on a rented GPU)

The **Batch queue** card collects layouts and emits a `prompts.json` for
[`moss-batch`](https://github.com/Iteranya/moss-batch), which renders them on a
SaladCloud GPU and shuts itself down when the batch ends.

- **+ add current layout** snapshots `toLayout()` plus the canvas size it was
  drawn at. The queue persists in `localStorage` (`moss.batch`).
- Each row validates its own bboxes: coordinates are centred on (0,0), so a box
  needs `|x| ≤ width/2` and `|y| ≤ height/2`. Rendering a 1776×1168 layout at
  1024×1024 crops characters out of frame — the warning catches it before you
  pay for it.
- **Export prompts.json** downloads the doc, **Copy JSON** puts it on the
  clipboard, **Clear** empties the queue.
- Set **batch runner** to `http://127.0.0.1:8787` and run
  `python batch_server.py` from the `moss-batch` checkout to POST jobs straight
  to the GPU. Leave it empty and the button just exports the file.

`bbox` stays `[y1, x1, y2, x2]` — y first, exactly as `toLayout()` writes it. The
full schema is in
[`moss-batch/BATCH_CONTRACT.md`](https://github.com/Iteranya/moss-batch/blob/main/BATCH_CONTRACT.md).

## Files

| file | what |
|---|---|
| `index.html` | the whole app (HTML + CSS + JS, no dependencies) |
| `demo-layout.json` | example layout loaded on first visit |
| `ddlc-beach.json` | another example: four characters, beach episode |
| `relay/server.py` | stdlib-only CORS relay for hosting this off localhost |
| `relay/worker.js` | same thing as a Cloudflare Worker, if you prefer that |

Licence: MIT.
