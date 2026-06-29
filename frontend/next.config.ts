import type { NextConfig } from 'next'

// IBM internal endpoints use a private CA that Node.js doesn't trust by default.
// This matches Vite's http-proxy behavior which was permissive about TLS in dev.
if (process.env.NODE_ENV !== 'production') {
  process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0'
}

const isStandaloneExport = Boolean(process.env.STANDALONE_EXPORT)

const nextConfig: NextConfig = {
  // Static export for embedding in gbserver (standalone mode).
  // Server mode (no output setting) is used for IBM deployments where Next.js
  // runs as its own server alongside the Python sidecar.
  ...(isStandaloneExport && {
    output: 'export',
    trailingSlash: true,
  }),
  skipTrailingSlashRedirect: true,
  // Analytics sidecar proxy — server mode only. In static export mode gbserver
  // proxies /api/analytics/* to the sidecar directly.
  ...(!isStandaloneExport && {
    async rewrites() {
      return [
        { source: '/api/analytics/:path*', destination: 'http://localhost:8090/api/analytics/:path*' },
      ]
    },
  }),
}

export default nextConfig
