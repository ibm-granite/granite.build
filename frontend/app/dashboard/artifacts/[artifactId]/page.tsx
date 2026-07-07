import ArtifactDetailPageClient from './ArtifactDetailPageClient'

export const dynamic = 'force-static'

export function generateStaticParams() {
  return [{ artifactId: '_' }]
}

export default function Page() {
  return <ArtifactDetailPageClient />
}
