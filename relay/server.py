#!/usr/bin/env python3
"""
MOSS Image relay — a CORS shim for api.mosi.cn, for a plain VPS.

Why: api.mosi.cn only returns CORS headers for `https://platform.mosi.cn` and
`localhost`. From any other origin (GitHub Pages, your own domain) the preflight
gets a bare 403, so a static page cannot call the API directly. This relay sits in
front, forwards the request server-side, and adds the CORS headers the browser needs.

Objectives: stdlib only, no dependencies, no state. The API key travels in the
Authorization header from the browser through this process to the API — it is never
logged or written to disk.

MODE 1 — serve the site AND the API on one origin (recommended, no CORS at all):
    STATIC_DIR=. PORT=9000 python3 server.py
    -> http://<host>:9000/  , Endpoint field: /v1/images/generations

MODE 2 — relay only, page hosted elsewhere:
    PORT=9000 ALLOW_ORIGINS=https://iteranya.github.io python3 server.py
    -> Endpoint field: https://<your-domain>/v1/images/generations

Test:
    curl -i -X OPTIONS localhost:9000 -H "Origin: https://iteranya.github.io" \
         -H "Access-Control-Request-Method: POST"

Put TLS in front (required — an https page cannot call an http relay):
    Caddy:  relay.example.com { reverse_proxy 127.0.0.1:9000 }
    nginx: proxy_pass http://127.0.0.1:9000;  (+ certbot)
"""

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = "https://api.mosi.cn/v1/images/generations"
PORT = int(os.environ.get("PORT", "9000"))
BIND = os.environ.get("BIND", "0.0.0.0")
ALLOW = [o.strip() for o in os.environ.get("ALLOW_ORIGINS", "*").split(",") if o.strip()]
UPSTREAM_TIMEOUT = int(os.environ.get("UPSTREAM_TIMEOUT", "120"))

# Point this at the folder holding index.html to have one process serve both the
# playground and /v1/ — then the page calls a RELATIVE endpoint, which is same-origin
# and involves no CORS whatsoever.
STATIC_DIR = os.environ.get("STATIC_DIR", "").strip()

# Optional shared secret for the proxied path when you cannot put basic auth in
# front of it (an explicit Authorization: Bearer header from the page would
# otherwise collide with HTTP basic auth). Send it as ?token=… or X-Relay-Token.
RELAY_TOKEN = os.environ.get("RELAY_TOKEN", "").strip()

MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}


def cors(origin: str):
    wildcard = "*" in ALLOW
    permitted = wildcard or origin in ALLOW
    return {
        "Access-Control-Allow-Origin": "*" if wildcard else (origin if permitted else "null"),
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "authorization, content-type",
        "Access-Control-Max-Age": "86400",
        "Vary": "Origin",
    }, permitted


class Handler(BaseHTTPRequestHandler):
    server_version = "moss-relay/1.0"

    def _cors(self):
        origin = self.headers.get("Origin", "")
        headers, permitted = cors(origin)
        # Same-origin requests are always fine — this is what makes STATIC_DIR mode
        # work with no CORS at all: the page and /v1/ live on the same host.
        host = self.headers.get("Host", "")
        if origin and host and origin in ("http://" + host, "https://" + host):
            permitted = True
            headers["Access-Control-Allow-Origin"] = origin
        return headers, permitted, origin

    def _send(self, status: int, body: bytes, headers: dict, ctype="application/json"):
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        if body:
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _err(self, status, message, code):
        body = json.dumps({"error": {"message": message, "type": "relay_error", "code": code}}).encode()
        headers, _, _ = self._cors()
        self._send(status, body, headers)

    def do_OPTIONS(self):  # noqa: N802
        headers, permitted, _ = self._cors()
        self._send(204, b"", headers)

    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/") == "/health":
            body = b'{"ok":true,"upstream":"' + UPSTREAM.encode() + b'"}'
            self._send(200, body, {"Content-Type": "application/json"})
            return
        if STATIC_DIR:
            path = self.path.split("?", 1)[0]
            if path in ("", "/"):
                path = "/index.html"
            # never serve dotfiles: .git/ (remotes, config), .env, .gitignore, editor junk
            if any(seg.startswith(".") for seg in path.split("/") if seg):
                self._err(404, "not found: %s" % path, "static_not_found")
                return
            root = os.path.abspath(STATIC_DIR)
            full = os.path.abspath(os.path.join(root, path.lstrip("/")))
            if (full == root or full.startswith(root + os.sep)) and os.path.isfile(full):
                with open(full, "rb") as fh:
                    data = fh.read()
                ctype = MIME.get(os.path.splitext(full)[1].lower(), "application/octet-stream")
                self._send(200, data, {}, ctype)
                return
            self._err(404, "not found: %s" % path, "static_not_found")
            return
        self._err(404, "not found", "relay_not_found")

    def do_POST(self):  # noqa: N802
        headers, permitted, origin = self._cors()
        if not permitted:
            self._err(403, "origin not allowed by this relay: %s" % origin, "relay_origin_denied")
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        body = self.rfile.read(length) if length else b""
        if not body:
            self._err(400, "empty body", "relay_empty_body")
            return

        if RELAY_TOKEN:
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            token = self.headers.get("X-Relay-Token", "") or qs.get("token", [""])[0]
            if token != RELAY_TOKEN:
                self._err(403, "bad or missing relay token", "relay_token_invalid")
                return

        auth = self.headers.get("Authorization", "")
        sys.stderr.write("-> POST %s from %s (%d bytes)\n" % (self.path, origin or "-", len(body)))
        sys.stderr.flush()

        req = urllib.request.Request(UPSTREAM, data=body, method="POST")
        req.add_header("Authorization", auth)
        req.add_header("Content-Type", "application/json")
        # NB: the browser's Origin is deliberately NOT forwarded, and neither is its
        # User-Agent. The API rejects origins it doesn't know; that is why this exists.

        try:
            with urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT) as resp:
                payload = resp.read()
                ctype = resp.headers.get("Content-Type", "application/json")
                sys.stderr.write("<- %d (%d bytes)\n" % (resp.status, len(payload)))
                self._send(resp.status, payload, headers, ctype)
        except urllib.error.HTTPError as e:
            payload = e.read()
            ctype = e.headers.get("Content-Type", "application/json") if e.headers else "application/json"
            sys.stderr.write("<- %d upstream error\n" % e.code)
            self._send(e.code, payload, headers, ctype)
        except Exception as e:  # timeout, dns, reset
            sys.stderr.write("<- relay failure: %r\n" % e)
            self._err(504, "relay could not reach the API: %r" % e, "relay_upstream_failed")


if __name__ == "__main__":
    print("moss-relay on http://%s:%d -> %s" % (BIND, PORT, UPSTREAM))
    print("allowed origins: %s" % (", ".join(ALLOW) or "none"))
    ThreadingHTTPServer((BIND, PORT), Handler).serve_forever()
