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
 * Standalone mode (make build-frontend): AUTOTUNEX_API_URL is baked into the
 * bundle at build time. When set, axios calls target that URL's /api/v1
 * directly. When unset, relative paths are used.
 */
export function autotunexApiBase(path: string): string {
  if (process.env.NODE_ENV === 'production' && process.env.AUTOTUNEX_API_URL) {
    return `${process.env.AUTOTUNEX_API_URL}/api/v1${path}`
  }
  return `/api/autotunex${path}`
}
