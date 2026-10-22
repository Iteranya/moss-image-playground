# Which button do I press

## TL;DR

| I want… | do this |
| --- | --- |
| **Sogni Supernet** (no GPU, no 27GB download, needs the key) | double-click `run.bat` |
| **My own GPU** (free, offline, slow) | `set MOSS_BACKEND=local` then `run.bat` |
| **Fake images, no GPU, no key** (frontend work) | `set MOSS_FAKE=1` then `run.bat` |
| **Just the API, no browser** | `cd backend` then `python run_sogni.py` |
| **Cost probe before spending Spark** | `curl http://127.0.0.1:8000/api/sogni/account` |

The browser opens itself at <http://127.0.0.1:8000/>.

`run.bat` no longer touches the local model unless you ask for it. Default is `MOSS_BACKEND=sogni`,
which means: no weights downloaded, no torch needed, no disk offload, nothing about your laptop's
GPU involved.

---

## The three engines

Pick one in the **Engine** dropdown in the left panel. It only appears when the backend is running.

| dropdown entry | who renders | cost | notes |
| --- | --- | --- | --- |
| **Sogni Supernet · MOSS 8-step** | Sogni's workers | ~1.25 Spark ≈ **$0.006** | DMD preset. Pinned to 8 steps, negative prompt ignored. Default. |
| **Sogni Supernet · MOSS full 50-step** | Sogni's workers | ~8.9 Spark ≈ **$0.044** | full-step SFT. 50-100 steps; asking for 25 is silently clamped to 50. |
| **This machine (local weights)** | your RTX 4060 | free | needs the 27GB download + torch. Disabled until the local pipeline loads. |

Options that can't render right now are greyed out, and the Endpoint badge says which is live
(`Sogni Supernet` vs `local GPU`).

---

## run.bat, verbosely

```bat
rem Sogni (default). Reads the key from .secrets\sogni.env, opens the browser, starts uvicorn.
run.bat

rem Local GPU instead. First run downloads ~27GB; MOSS_MODE defaults to disk offload.
set MOSS_BACKEND=local
run.bat

rem No GPU, no key, no network: renders a labelled placeholder so you can test the UI.
set MOSS_FAKE=1
run.bat

rem Sogni without spending Spark: exercises the whole job pipeline, mock image, zero cost.
set MOSS_SOGNI_DRY_RUN=1
run.bat
```

When you're done, close the terminal window (or Ctrl-C in it). Nothing keeps running.

---

## Clearing a `set` variable in cmd

`set MOSS_BACKEND=local` sticks for that window. To undo:

```bat
set MOSS_BACKEND=
set MOSS_FAKE=
set MOSS_SOGNI_DRY_RUN=
```

A fresh terminal starts clean, so the default (Sogni) comes back.

---

## The API key

It lives in `.secrets/sogni.env` (gitignored, not committed, not printed anywhere):

```
SOGNI_API_KEY=...
```

Override without editing the file:

```bat
set SOGNI_API_KEY=whatever
run.bat
```

Rotate it at <https://dashboard.sogni.ai/api-key>, then replace that one line.

---

## Useful URLs while it's running

| URL | what |
| --- | --- |
| `http://127.0.0.1:8000/` | the playground |
| `/health` | engines available, local pipeline state, turbo state |
| `/api/sogni/account` | live Spark balance + per-preset price per image |
| `/api/jobs` | recent jobs |
| `/docs` | FastAPI's own interactive API docs |

---

## If something's wrong

| symptom | cause |
| --- | --- |
| Engine dropdown missing | backend not answering `/health` — check the terminal |
| "This machine" greyed out | local weights not loaded. Expected unless `MOSS_BACKEND=local` |
| Sogni options greyed out | `SOGNI_API_KEY` empty or `.secrets/sogni.env` missing |
| `ModuleNotFoundError: torch` in the log | harmless when running Sogni; only local inference needs torch |
| Spark balance not showing | `/api/sogni/account` failed — open it directly to see the error |
| Render fails with a 4xx from Sogni | preset/canvas out of documented bounds; see `backend/README.md` |
