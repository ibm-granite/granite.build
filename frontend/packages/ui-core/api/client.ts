import axios, { type AxiosInstance } from 'axios'

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
 * Per-request overrides for ui-core's API clients.
 *
 * ui-core's clients target relative paths with no credentials, which is right for
 * a standalone deployment where gbserver serves the frontend from the same origin
 * and needs no auth. A hosted deployment differs on two axes: it attaches a
 * bearer token, and it may route through a per-environment prefix chosen at
 * runtime by an environment switcher.
 *
 * Without a seam here, moving any component that calls an API into ui-core
 * silently swaps its authenticated client for an unauthenticated one — the
 * requests still go out, they just come back 401 and the component renders as
 * though the data were missing. That is a genuinely hard bug to see, so the seam
 * exists to make the divergence explicit rather than accidental.
 *
 * **This is deliberately one seam for all four clients, not one per client.** The
 * problem is not specific to any of them: `gbserver`, `analytics`, `chat` and
 * `dataProcessing` are each a module-private `axios.create()` that no host
 * interceptor can reach, and more than one scopes by identity server-side — both
 * chat's session scoping and `analytics`'s saved failure-trend ownership resolve
 * through `resolve_identity()`, so both collapse to one shared identity when
 * requests arrive unidentified. A per-client hook would have to be rediscovered
 * and rebuilt for each one.
 *
 * Every hook is called **per request**, not once at configure time, because both
 * the token and the active environment can change while the app is running.
 */
export interface ApiClientOverrides {
  /**
   * Base URL for this request, **for the gbserver client only**.
   *
   * Headers and the 401 hook are shared by every client, because identity is
   * needed everywhere. Base URL is not, and sharing it would be a bug: the four
   * clients sit on different paths, so a replacement written to point at a
   * different gbserver would rewrite `/api/analytics/…` to a gbserver path and
   * 404 every request.
   *
   * So clients opt in via `allowHostBaseUrl`, and only gbserver does. The default
   * is the safe direction: a new client added later ignores this hook unless its
   * author asks for it.
   */
  resolveBaseUrl?: () => string
  /**
   * Extra headers, typically `Authorization`. Merged over existing headers.
   *
   * May be async: a host that has to refresh an expiring token before it can
   * answer returns a promise, and every call path here awaits it. Returning
   * `undefined` contributes nothing, which is the same as having no provider.
   */
  resolveHeaders?: () =>
    | Record<string, string>
    | undefined
    | Promise<Record<string, string> | undefined>
  /**
   * Called when a request comes back 401, before the error propagates. For a
   * host that needs to clear a stale token and restart its login flow.
   */
  onUnauthorized?: (error: unknown) => void
}

/**
 * Previous name for {@link ApiClientOverrides}, from when the seam covered only
 * the gbserver client. Kept so existing hosts keep compiling.
 */
export type GbserverClientOverrides = ApiClientOverrides

let overrides: ApiClientOverrides = {}

/**
 * Install host-specific behaviour on ui-core's API clients.
 *
 * Call once, before the first request — from module scope of something the app
 * shell imports, not from an effect, since a query can fire before effects run.
 * Calling it again replaces the previous overrides wholesale.
 */
export function configureApiClient(next: ApiClientOverrides): void {
  overrides = next
}

/**
 * Previous name for {@link configureApiClient}. Identical behaviour — the hooks
 * were never gbserver-specific, only their name was.
 */
export const configureGbserverClient = configureApiClient

/** Read the installed overrides. For ui-core's own client wiring. */
export function apiClientOverrides(): ApiClientOverrides {
  return overrides
}

/** Previous name for {@link apiClientOverrides}. */
export const gbserverClientOverrides = apiClientOverrides

/**
 * Resolve the host's headers for one request, swallowing any failure.
 *
 * A provider that throws or rejects contributes no headers rather than failing
 * the call: that degrades the request to exactly the unidentified behaviour it
 * has with no provider installed, whereas propagating the error would take down
 * a widget over something like a transient token read. Exported because the
 * `/chat/stream` path uses native `fetch` — axios cannot stream cleanly
 * in-browser — so it cannot go through an interceptor.
 */
export async function resolveApiHeaders(): Promise<Record<string, string>> {
  const { resolveHeaders } = overrides
  if (!resolveHeaders) return {}
  try {
    return (await resolveHeaders()) ?? {}
  } catch {
    return {}
  }
}

/**
 * Marks a request as one this factory recognised as belonging to its own API.
 *
 * The response side cannot re-derive that from `baseURL`, because by then the
 * request interceptor has replaced it with the host's prefix — comparing against
 * the client's default there would treat every real 401 as foreign and never call
 * `onUnauthorized`. axios carries custom config fields through to `error.config`,
 * so the decision is recorded once on the way out and read back on the way in.
 */
const OWNED_REQUEST = '__uiCoreApiRequest'

type OwnedConfig = { [OWNED_REQUEST]?: boolean }

/**
 * An axios instance wired to the host overrides above.
 *
 * **Every ui-core API client must be built here, not with `axios.create()`.** An
 * instance made directly gets none of the wiring below, so a host's headers and
 * its 401 hook never reach it — and that failure is silent: the requests still go
 * out, they just arrive unidentified. This is the one rule to carry over when
 * adding a client. It is a convention rather than something the tests enforce;
 * a source scan cannot tell reliably whether a module can reach a constructor.
 *
 * Replaces four hand-rolled `axios.create()` calls that each had to remember the
 * same wiring, and only one of which ever had it. Anything a promoted component
 * reaches for has to be injectable or promoting it is a silent regression; this
 * is that rule applied to the client, the same way `config/routeShapes.ts` applies
 * it to URL shape.
 *
 * **Overrides apply only to requests still pointing at `baseURL`.** A request that
 * sets its own is not addressing this API at all — `getBuildStepLog` passes `''`
 * so gbserver's `log_path` is used exactly as given. Rewriting that would prepend
 * the host's prefix to an already-complete path, and attaching the host's bearer
 * token is worse than cosmetic: an absolute `log_path` can point at another
 * origin, so the token would be disclosed there and can make a presigned
 * object-store URL reject the request for carrying a second auth mechanism.
 *
 * `allowHostBaseUrl` opts a client into `resolveBaseUrl`. **Only gbserver passes
 * it**, and a new client should not without a reason — see that field's docs for
 * why sharing it across clients would 404 the other three.
 */
export function createApiClient(
  baseURL: string,
  options: { allowHostBaseUrl?: boolean } = {},
): AxiosInstance {
  const instance = axios.create({ baseURL })

  // Consulted per request rather than at module load, so a changing token or a
  // switched environment is picked up without a reload. Async because a provider
  // may need to refresh a token first; axios awaits request interceptors.
  instance.interceptors.request.use(async (config) => {
    if (config.baseURL !== baseURL) return config
    ;(config as typeof config & OwnedConfig)[OWNED_REQUEST] = true
    const { resolveBaseUrl } = overrides
    if (options.allowHostBaseUrl && resolveBaseUrl) config.baseURL = resolveBaseUrl()
    const extra = await resolveApiHeaders()
    for (const [name, value] of Object.entries(extra)) {
      config.headers.set(name, value)
    }
    return config
  })

  instance.interceptors.response.use(
    (response) => response,
    (error) => {
      const err = error as {
        response?: { status?: number }
        config?: OwnedConfig
      }
      if (err?.response?.status === 401 && err?.config?.[OWNED_REQUEST]) {
        overrides.onUnauthorized?.(error)
      }
      return Promise.reject(error)
    },
  )

  return instance
}
