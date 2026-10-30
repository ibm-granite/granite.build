'use client'

import { Suspense, useEffect, useState } from 'react'
import { InlineNotification, SkeletonText } from '@carbon/react'
import { useQuery } from '@tanstack/react-query'
import { getArtifact, listArtifacts } from '@granite-build/ui-core/api/gbserver'
import { PageHeader } from '@granite-build/ui-core/components/PageHeader'
import { ARTIFACT_TYPE_CONFIG, artifactTypeKey } from '@granite-build/ui-core/config/artifactTypes'
import { ArtifactDetails } from './ArtifactDetails'
import { useRouteId } from '@granite-build/ui-core/hooks/useRouteId'
import { isUUID } from '@granite-build/ui-core/lib/uuid'

// useSearchParams() bails the page out to client-side rendering up to the
// nearest Suspense boundary during static export — without one here, the
// statically-exported HTML (built with no query param) and the client's
// first render (which already sees the real ?id=) disagree, tripping a
// hydration mismatch (React error #418). The fallback below matches what
// the static export produces so hydration has nothing to reconcile against.
export default function ArtifactDetailPage() {
  return (
    <Suspense fallback={<ArtifactDetailFallback />}>
      <ArtifactDetailContent />
    </Suspense>
  )
}

function ArtifactDetailFallback() {
  return (
    <div>
      <div style={{ padding: '2rem 1.5rem 1.5rem' }}>
        <PageHeader
          crumbs={[
            { label: 'Granite.build', to: '/dashboard' },
            { label: 'Artifacts', to: '/dashboard/artifacts' },
            { label: '…' },
          ]}
        />
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginTop: '0.5rem' }}>
          <SkeletonText width="300px" />
        </div>
      </div>
    </div>
  )
}

// A lineage click may open the page by URI, in any spelling; null when nothing
// is registered under it.
async function artifactByUri(uri: string) {
  return (await listArtifacts({ uri, match_lineage_uri: true })).items[0] ?? null
}

function ArtifactDetailContent() {
  const artifactId = useRouteId('/dashboard/artifacts')

  const { data: artifact, isLoading, error } = useQuery({
    queryKey: ['artifact', artifactId],
    queryFn: () => (isUUID(artifactId) ? getArtifact(artifactId) : artifactByUri(artifactId)),
    enabled: Boolean(artifactId),
    retry: false,
  })

  // The static export renders with no ?id (so not loading), while the client's
  // first render already has one (so loading): the two trees differ above the
  // tabs, which shifts every useId under them (Carbon's tab/panel ids) and trips
  // a hydration mismatch. Render the export's fallback until mounted.
  const [mounted, setMounted] = useState(false)
  useEffect(() => setMounted(true), [])
  if (!mounted) return <ArtifactDetailFallback />

  if (!isLoading && !error && artifact === null) {
    return (
      <div style={{ padding: '1rem 1.5rem' }}>
        <InlineNotification kind="warning" title="Artifact not found" subtitle={`Nothing in the artifact registry is registered under ${artifactId}.`} hideCloseButton />
      </div>
    )
  }

  if (error) {
    return (
      <div style={{ padding: '1rem 1.5rem' }}>
        <InlineNotification kind="error" title="Failed to load artifact" subtitle={String(error)} />
      </div>
    )
  }

  const typeIcon = artifact
    ? ARTIFACT_TYPE_CONFIG[artifactTypeKey(artifact.artifact_type)]?.icon
    : null

  return (
    <div>
      <div style={{ padding: '2rem 1.5rem 1.5rem' }}>
        <PageHeader
          crumbs={[
            { label: 'Granite.build', to: '/dashboard' },
            { label: 'Artifacts', to: '/dashboard/artifacts' },
            { label: artifact?.name ?? '…' },
          ]}
        />
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginTop: '0.5rem' }}>
          {isLoading ? (
            <SkeletonText width="300px" />
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              {typeIcon}
              <h4 style={{ margin: 0 }}>{artifact?.name}</h4>
            </div>
          )}
        </div>
      </div>

      <ArtifactDetails artifact={artifact ?? undefined} loading={isLoading} artifactId={artifact?.uuid ?? artifactId} />
    </div>
  )
}
