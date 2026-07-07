import BuildDetailPageClient from './BuildDetailPageClient'

export const dynamic = 'force-static'

export function generateStaticParams() {
  return [{ buildId: '_' }]
}

export default function Page() {
  return <BuildDetailPageClient />
}
