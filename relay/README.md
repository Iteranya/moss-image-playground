# MOSS relay

A ~150-line, dependency-free CORS shim so the playground can call `api.mosi.cn` from a
hosted origin.

## Why it's needed

`api.mosi.cn` only sends CORS headers to `https://platform.mosi.cn` and `localhost`.
Measured:

| `Origin` | preflight |
|---|---|
| `http://127.0.0.1:8765` | `204` + `access-control-allow-origin` ✅ |
| `https://platform.mosi.cn` | `204` ✅ |
| `https://iteranya.github.io` | `403`, no CORS headers ❌ |
| `https://example.com` | `403`, no CORS headers ❌ |
| `https://docs.mosi.cn` | `403`, no CORS headers ❌ |

Nothing in their docs mentions it and there's no setting for it, so the only fix is a
server that forwards the call. That's this.

## Run it

```bash
# on the VPS
git pull
PORT=9000 ALLOW_ORIGINS=https://iteranya.github.io python3 relay/server.py
```

No dependencies — stdlib only, Python 3.8+. `ALLOW_ORIGINS` takes a comma-separated
list; omit it and the relay will serve any origin (fine for a quick test, sloppy for
something public).

Health check: `curl localhost:9000/health`

## TLS is mandatory

The page is served over `https`, so a browser will refuse to call an `http://` relay
(mixed content). Put a cert in front — Caddy is two lines:

```
relay.example.com {
    reverse_proxy 127.0.0.1:9000
}
```

nginx + certbot works the same way (`proxy_pass http://127.0.0.1:9000;`).

## systemd (optional)

```ini
[Unit]
Description=MOSS image relay
After=network.target

[Service]
WorkingDirectory=/opt/moss-image-playground
Environment=PORT=9000
Environment=ALLOW_ORIGINS=https://iteranya.github.io
ExecStart=/usr/bin/python3 relay/server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

## Wire it into the page

Open the playground, put your relay URL in the **Endpoint** field:

```
https://relay.example.com/v1/images/generations
```

The relay accepts any path, so `/v1/images/generations`, `/generate`, or just `/` all
work. The value is saved in `localStorage`, and the red "hosted copy" banner
disappears once it's set.

## What passes through

Only the `Authorization` header and the JSON body. The browser's `Origin` and
`User-Agent` are **not** forwarded — the API rejects unknown origins, which is the
whole point. Nothing is logged to disk; request lines go to stderr.

The API key therefore transits your own VPS. Since it's yours, that's fine — but it
does mean the relay must be TLS and you shouldn't put it on a host you share.
