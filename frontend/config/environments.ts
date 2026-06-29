/**
 * Static constants used by the env switcher.
 * Runtime config (auth provider, environment name, GitHub client ID) is
 * fetched from GET /api/config and provided via AuthContext — it is not
 * baked into the bundle.
 */

/** URL query parameter used to encode the active environment override. */
export const ENV_OVERRIDE_PARAM = 'env'
