"""Start the backend with the Sogni engine, reading the key from ../.secrets/sogni.env.

Used by ../run.bat, but callable directly:
    python backend/run_sogni.py            # 127.0.0.1:8000
    python backend/run_sogni.py --port 8100
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV_FILE = HERE.parent / ".secrets" / "sogni.env"


def load_key() -> str | None:
    key = os.environ.get("SOGNI_API_KEY")
    if key:
        return key
    if not ENV_FILE.exists():
        return None
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip().upper() == "SOGNI_API_KEY":
            return value.strip().strip('"').strip("'")
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    key = load_key()
    if not key:
        print(f"[run] no SOGNI_API_KEY in env and none found in {ENV_FILE}", file=sys.stderr)
        return 1
    os.environ["SOGNI_API_KEY"] = key
    os.environ.setdefault("MOSS_BACKEND", "sogni")
    # Sogni renders remotely: there is no local pipeline to preload, so never try to build one.
    os.environ.setdefault("MOSS_PRELOAD", "0")

    print(f"[run] engine={os.environ['MOSS_BACKEND']} · key ...{key[-4:]} · http://{args.host}:{args.port}/")

    import uvicorn

    uvicorn.run("app:app", host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
