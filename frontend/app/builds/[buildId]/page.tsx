import BuildDetailPageClient from './BuildDetailPageClient'

// force-static: generate a static shell; all data fetching happens client-side.
// output: 'export' requires at least one entry — the placeholder is never navigated
// to in practice; direct navigation is handled by the SPA fallback on gbserver.
export const dynamic = 'force-static'

export function generateStaticParams() {
  return [{ buildId: '_' }]
}

export default function Page(props: { params: Promise<{ buildId: string }> }) {
  return <BuildDetailPageClient {...props} />
}
