# MOSS relay / self-host

One file that either (a) hosts the playground **and** proxies the API on the same origin —
no CORS involved at all — or (b) acts as a standalone CORS relay if you keep the site
somewhere else.

## Mode 1 — host everything on the VPS (recommended)

```bash
git pull
cd relay
STATIC_DIR=.. PORT=9000 python3 relay/server.py
```

Open `http://<your-vps>:9000/` and set the **Endpoint** field to:

```
/v1/images/generations
```

That's a *relative* path, so the browser calls your own host, the process forwards to
`api.mosi.cn` server-side, and no CORS header is ever needed. No second service, no
extra port.

Put TLS in front (Caddy auto-certifies), because the API key crosses the wire:

```
play.example.com {
    reverse_proxy 127.0.0.1:9000
}
```

`STATIC_DIR=..` points at the folder holding `index.html`. Dotfiles are refused, so
`.git/`, `.env` and friends are not served even though they live in that folder.

### systemd

```ini
[Unit]
Description=MOSS playground + relay
After=network.target

[Service]
WorkingDirectory=/opt/moss-image-playground/relay
Environment=STATIC_DIR=..
Environment=PORT=9000
ExecStart=/usr/bin/python3 /opt/moss-image-playground/relay/server.py
Restart=always
User=www-data

[Install]
WantedBy=multi-user.target
```

### If you put basic auth in front (important)

The page's request carries `Authorization: Bearer <moss key>`. That header **replaces** the
`Authorization: Basic ...` the browser would otherwise attach, so any path behind Caddy's
`basic_auth` answers 401 forever — you get an endless login prompt on every Generate.

Keep the API path out of the auth block:

```
play.example.com {
    @api path /v1/*
    handle @api {
        reverse_proxy localhost:9000
    }

    handle {
        basic_auth {
            user $2a$14$...
        }
        reverse_proxy localhost:9000
    }
}
```

The proxied path is now public, so gate it with a shared secret instead:

```bash
RELAY_TOKEN=$(openssl rand -hex 16) STATIC_DIR=.. PORT=9000 python3 server.py
```

and append it to the endpoint in the page:

```
/v1/images/generations?token=<that hex>
```

Wrong or missing token → `403 relay_token_invalid`. The page itself is still behind basic auth.

## Mode 2 — relay only, site hosted elsewhere

```bash
PORT=9000 ALLOW_ORIGINS=https://iteranya.github.io python3 relay/server.py
```

Endpoint field: `https://<your-domain>/v1/images/generations`. Here CORS *is* in play,
so `ALLOW_ORIGINS` must contain the exact origin of the page (scheme + host, no trailing
slash). Set it — the default is `*`, which turns the box into an open proxy to the API.


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
