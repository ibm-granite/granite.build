import type { NextConfig } from 'next'

if (process.env.NODE_ENV !== 'production') {
  process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0'
}

const isProd = process.env.NODE_ENV === 'production'
const gbserverApiUrl = process.env.GBSERVER_API_URL
const autotunexApiUrl = process.env.AUTOTUNEX_API_URL

const nextConfig: NextConfig = {
  // output: 'export' is standalone-only — it conflicts with rewrites (used in dev).
  ...(isProd ? { output: 'export' } : {}),
  trailingSlash: true,
  skipTrailingSlashRedirect: true,
  // @granite-build/ui-core ships raw TS/TSX source (no build step) from the workspace —
  // Next only compiles first-party code by default, so opt this workspace package in too.
  transpilePackages: ['@granite-build/ui-core'],
  // Expose the API URLs to the client bundle without a NEXT_PUBLIC_ prefix.
  env: { GBSERVER_API_URL: gbserverApiUrl ?? '', AUTOTUNEX_API_URL: autotunexApiUrl ?? '' },
  // Dev mode: proxy /api/* to gbserver and /api/autotunex/* to the AutoTuneX
  // server, both server-side (no CORS). Both are optional — omit either URL to
  // run the UI without that backend (pages load, data shows empty). In standalone
  // builds gbserver does the same forwarding itself (api/autotunex_proxy.py).
  ...(!isProd && (gbserverApiUrl || autotunexApiUrl)
    ? {
        async rewrites() {
          // gbserver is strict about trailing slashes (some routes require a
          // trailing slash, e.g. /api/v1/builds/, others reject it). Preserve
          // the client's slash verbatim: match slash-terminated paths first so
          // :path* doesn't drop the final "/" before the query string.
          const rules: { source: string; destination: string }[] = []
          // AutoTuneX first: /api/autotunex/* is more specific than the /api/:path*
          // catch-all below, which would otherwise swallow it and send AutoTuneX
          // calls to gbserver.
          if (autotunexApiUrl) {
            rules.push(
              {
                source: '/api/autotunex/:path*/',
                destination: `${autotunexApiUrl}/api/v1/:path*/`,
              },
              {
                source: '/api/autotunex/:path*',
                destination: `${autotunexApiUrl}/api/v1/:path*`,
              }
            )
          }
          if (gbserverApiUrl) {
            rules.push(
              { source: '/api/:path*/', destination: `${gbserverApiUrl}/api/:path*/` },
              { source: '/api/:path*', destination: `${gbserverApiUrl}/api/:path*` }
            )
          }
          return rules
        },
      }
    : {}),
}

export default nextConfig
