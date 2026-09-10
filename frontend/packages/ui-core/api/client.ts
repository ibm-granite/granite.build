/**
 * Returns the base URL prefix for API calls.
 *
 * Dev mode (yarn dev): always returns a relative path. next.config.ts rewrites
 * /api/* → GBSERVER_API_URL when set, forwarding server-side (no CORS). When
 * GBSERVER_API_URL is not set, no proxy is configured — API calls return 404 and
 * pages show empty states, but the UI itself loads fine.
 *
 * Standalone mode (make build-frontend): GBSERVER_API_URL is baked into the bundle
 * at build time. When set, axios calls target that URL directly. When unset (default),
 * relative paths are used — works because gbserver serves the frontend at the same
 * origin and handles all /api/* requests itself.
 */
export function apiBase(path: string): string {
  if (process.env.NODE_ENV === 'production' && process.env.GBSERVER_API_URL) {
    return `${process.env.GBSERVER_API_URL}${path}`
  }
  return path
}

/**
 * Returns the base URL prefix for AutoTuneX API calls.
 *
 * Dev mode (yarn dev): always returns a relative path. next.config.ts rewrites
 * /api/autotunex/* → AUTOTUNEX_API_URL/api/v1/* when set, forwarding
 * server-side (no CORS, cookies pass through automatically). When
 * AUTOTUNEX_API_URL is not set, no proxy is configured — API calls return 404
 * and AutoTuneX pages show empty states, but the UI itself loads fine.
 *
 * Standalone mode (make build-frontend): always relative. Unlike `apiBase`
 * above, this deliberately ignores AUTOTUNEX_API_URL: gbserver serves the
 * frontend and proxies /api/autotunex/* to the AutoTuneX API server-side
 * (api/autotunex_proxy.py), which is the whole reason that proxy exists. Sending
 * the browser straight at AUTOTUNEX_API_URL/api/v1 instead would need
 * credentialed CORS (cors_allow_origins) and session_cookie_same_site="none"
 * configured on AutoTuneX; nothing here sets either, and next.config.ts's `env:`
 * block inlines the value into the client bundle, so honouring it here meant
 * `AUTOTUNEX_API_URL=… make build-frontend` shipped a silently broken app.
 */
export function autotunexApiBase(path: string): string {
  return `/api/autotunex${path}`
}
