import type { NextConfig } from 'next'

if (process.env.NODE_ENV !== 'production') {
  process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0'
}

const isProd = process.env.NODE_ENV === 'production'
const gbserverApiUrl = process.env.GBSERVER_API_URL

const nextConfig: NextConfig = {
  // output: 'export' is standalone-only — it conflicts with rewrites (used in dev).
  ...(isProd ? { output: 'export' } : {}),
  trailingSlash: true,
  skipTrailingSlashRedirect: true,
  // @granite-build/ui-core ships raw TS/TSX source (no build step) from the workspace —
  // Next only compiles first-party code by default, so opt this workspace package in too.
  transpilePackages: ['@granite-build/ui-core'],
  // Expose GBSERVER_API_URL to the client bundle without a NEXT_PUBLIC_ prefix.
  env: { GBSERVER_API_URL: gbserverApiUrl ?? '' },
  // Dev mode: proxy /api/* to gbserver server-side (no CORS). Optional — omit
  // GBSERVER_API_URL to run the UI with no backend (pages load, data shows empty).
  ...(!isProd && gbserverApiUrl
    ? {
        async rewrites() {
          // gbserver is strict about trailing slashes (some routes require a
          // trailing slash, e.g. /api/v1/builds/, others reject it). Preserve
          // the client's slash verbatim: match slash-terminated paths first so
          // :path* doesn't drop the final "/" before the query string.
          return [
            { source: '/api/:path*/', destination: `${gbserverApiUrl}/api/:path*/` },
            { source: '/api/:path*', destination: `${gbserverApiUrl}/api/:path*` },
          ]
        },
      }
    : {}),
}

export default nextConfig
