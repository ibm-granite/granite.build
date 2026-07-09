'use client'

import { Suspense, useEffect, useState } from 'react'
import { useSearchParams } from 'next/navigation'
import { InlineNotification, SkeletonText } from '@carbon/react'
import { useQuery } from '@tanstack/react-query'
import { getArtifact } from '@/api/gbserver'
import { PageHeader } from '@/components/PageHeader'
import { ARTIFACT_TYPE_CONFIG, artifactTypeKey } from '@/config/artifactTypes'
import { ArtifactDetails } from './ArtifactDetails'

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

function ArtifactDetailContent() {
  // The real id lives in the ?id= query param, not location.hash — a hash read
  // in a mount-only effect breaks when navigating from one artifact's page to
  // another (e.g. via the lineage graph), since both are the same "_" route to
  // Next's router and a one-time hash read never sees the new id.
  // useSearchParams() is reactive, but it isn't enough by itself: Next's router
  // patches window.history, so our own cosmetic replaceState below (stripping
  // the query param once we've adopted the id) makes useSearchParams() briefly
  // report no id again. Latching the id into state — only ever overwritten by
  // a new *non-empty* param value — survives that revert.
  const searchParams = useSearchParams()
  const paramId = searchParams.get('id')
  const [artifactId, setArtifactId] = useState(paramId ?? '')

  useEffect(() => {
    if (paramId && paramId !== artifactId) {
      setArtifactId(paramId)
    }
  }, [paramId, artifactId])

  useEffect(() => {
    if (artifactId) {
      window.history.replaceState(null, '', `/dashboard/artifacts/${artifactId}/`)
    }
  }, [artifactId])

  const { data: artifact, isLoading, error } = useQuery({
    queryKey: ['artifact', artifactId],
    queryFn: () => getArtifact(artifactId!),
    enabled: Boolean(artifactId),
  })

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

      <ArtifactDetails artifact={artifact} loading={isLoading} artifactId={artifactId!} />
    </div>
  )
}
