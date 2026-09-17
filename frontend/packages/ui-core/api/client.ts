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
 * Per-request overrides for the gbserver client.
 *
 * ui-core's client targets a relative `/api/v1` with no credentials, which is
 * right for a standalone deployment where gbserver serves the frontend from the
 * same origin and needs no auth. A hosted deployment differs on two axes: it
 * attaches a bearer token, and it may route through a per-environment prefix
 * chosen at runtime by an environment switcher.
 *
 * Without a seam here, moving any component that calls gbserver into ui-core
 * silently swaps its authenticated client for this unauthenticated one — the
 * requests still go out, they just come back 401 and the component renders as
 * though the data were missing. That is a genuinely hard bug to see, so the
 * seam exists to make the divergence explicit rather than accidental.
 *
 * Every hook is called **per request**, not once at configure time, because both
 * the token and the active environment can change while the app is running.
 */
export interface GbserverClientOverrides {
  /** Base URL for this request. Defaults to `apiBase('/api/v1')`. */
  resolveBaseUrl?: () => string
  /** Extra headers, typically `Authorization`. Merged over existing headers. */
  resolveHeaders?: () => Record<string, string> | undefined
  /**
   * Called when a request comes back 401, before the error propagates. For a
   * host that needs to clear a stale token and restart its login flow.
   */
  onUnauthorized?: (error: unknown) => void
}

let overrides: GbserverClientOverrides = {}

/**
 * Install host-specific behaviour on ui-core's gbserver client.
 *
 * Call once, before the first request — from module scope of something the app
 * shell imports, not from an effect, since a query can fire before effects run.
 * Calling it again replaces the previous overrides wholesale.
 */
export function configureGbserverClient(next: GbserverClientOverrides): void {
  overrides = next
}

/** Read the installed overrides. For ui-core's own client wiring. */
export function gbserverClientOverrides(): GbserverClientOverrides {
  return overrides
}
