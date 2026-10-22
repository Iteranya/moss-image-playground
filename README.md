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

## Files

| file | what |
|---|---|
| `index.html` | the whole app (HTML + CSS + JS, no dependencies) |
| `demo-layout.json` | example layout loaded on first visit |
| `ddlc-beach.json` | another example: four characters, beach episode |

Licence: MIT.
