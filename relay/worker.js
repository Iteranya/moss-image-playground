/**
 * MOSS Image relay — a CORS shim for api.mosi.cn
 *
 * Why this exists: api.mosi.cn only returns CORS headers for `https://platform.mosi.cn`
 * and localhost. From any other origin (e.g. GitHub Pages) the preflight gets a bare 403,
 * so a static page cannot call the API directly. This Worker sits in front, forwards the
 * request server-side, and adds the CORS headers the browser needs.
 *
 * It is stateless and stores nothing. The API key travels in the Authorization header
 * from the browser through this Worker to the API — it is never logged or persisted.
 *
 * Deploy:
 *   npm install -g wrangler
 *   wrangler login
 *   cd relay && wrangler deploy
 *
 * Then put the printed URL (https://moss-relay.<your-subdomain>.workers.dev) into the
 * page's Endpoint field.
 *
 * Lock it down (recommended once it works):
 *   wrangler secret put ALLOW_ORIGINS   ->  https://iteranya.github.io
 * or edit [vars] in wrangler.toml. Default is "*" (any origin may use your relay).
 */

const UPSTREAM = 'https://api.mosi.cn/v1/images/generations';

function corsHeaders(origin, allow) {
  const wildcard = allow.includes('*');
  const permitted = wildcard || allow.includes(origin);
  return {
    'Access-Control-Allow-Origin': wildcard ? '*' : (permitted ? origin : 'null'),
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'authorization, content-type',
    'Access-Control-Max-Age': '86400',
    'Vary': 'Origin',
    '_permitted': permitted,
  };
}

export default {
  async fetch(request, env) {
    const allow = (env.ALLOW_ORIGINS || '*')
      .split(',').map(s => s.trim()).filter(Boolean);
    const origin = request.headers.get('Origin') || '';
    const headers = corsHeaders(origin, allow);

    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers });
    }
    if (request.method !== 'POST') {
      return new Response(JSON.stringify({ error: { message: 'POST only', type: 'invalid_request' } }),
        { status: 405, headers: { ...headers, 'Content-Type': 'application/json' } });
    }
    if (!headers._permitted) {
      return new Response(JSON.stringify({ error: { message: 'origin not allowed by this relay: ' + origin, type: 'forbidden', code: 'relay_origin_denied' } }),
        { status: 403, headers: { ...headers, 'Content-Type': 'application/json' } });
    }

    const body = await request.text();

    // Note: the browser's Origin header is deliberately NOT forwarded — the API
    // rejects origins it doesn't know, which is the whole reason this relay exists.
    const upstream = await fetch(UPSTREAM, {
      method: 'POST',
      headers: {
        'Authorization': request.headers.get('Authorization') || '',
        'Content-Type': 'application/json',
      },
      body,
    });

    const text = await upstream.text();
    return new Response(text, {
      status: upstream.status,
      headers: {
        ...headers,
        'Content-Type': upstream.headers.get('Content-Type') || 'application/json',
        'X-Relay-Upstream-Status': String(upstream.status),
      },
    });
  },
};
