# MOSS Image Playground

A single-page layout editor for [MOSS-Image](https://platform.mosi.cn/) text-to-image generation
(`POST https://api.mosi.cn/v1/images/generations`).

Draw boxes on a canvas, give each box its own prompt, add a global scene brief, hit **Generate**.
The page builds the layout JSON for you and calls the API directly from your browser — no backend,
no build step, no dependencies.

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

## Making the hosted copy work (relay)

Static hosting can't fix this on its own, so run the relay in `relay/` on any box you control
(a VPS, a Raspberry Pi, whatever):

```bash
PORT=9000 ALLOW_ORIGINS=https://<your-pages-host> python3 relay/server.py
```

No dependencies. Put TLS in front (Caddy or nginx + certbot) — an https page refuses to call an
http relay. Then paste the relay URL into the playground's **Endpoint** field:

```
https://relay.example.com/v1/images/generations
```

The banner disappears the moment an endpoint is set, and the key still lives only in your
browser's localStorage — it just transits your own relay on the way to the API.
Details: [`relay/README.md`](relay/README.md).

## Files

| file | what |
|---|---|
| `index.html` | the whole app (HTML + CSS + JS, no dependencies) |
| `demo-layout.json` | example layout loaded on first visit |
| `ddlc-beach.json` | another example: four characters, beach episode |
| `relay/server.py` | stdlib-only CORS relay for hosting this off localhost |
| `relay/worker.js` | same thing as a Cloudflare Worker, if you prefer that |

Licence: MIT.
