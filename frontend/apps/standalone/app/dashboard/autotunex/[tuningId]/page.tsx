import TuningDetailPageClient from './TuningDetailPageClient'

export const dynamic = 'force-static'

export function generateStaticParams() {
  return [{ tuningId: '_' }]
}

export default function Page() {
  return <TuningDetailPageClient />
}
