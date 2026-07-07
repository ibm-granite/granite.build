'use client'

import { useState, useEffect } from 'react'
import { InlineNotification, SkeletonText } from '@carbon/react'
import { useQuery } from '@tanstack/react-query'
import { getArtifact } from '@/api/gbserver'
import { PageHeader } from '@/components/PageHeader'
import { ARTIFACT_TYPE_CONFIG, artifactTypeKey } from '@/config/artifactTypes'
import { ArtifactDetails } from './ArtifactDetails'

export default function ArtifactDetailPage() {
  const [artifactId, setArtifactId] = useState('')
  useEffect(() => {
    const id = window.location.hash.slice(1)
    if (id) {
      setArtifactId(id)
      window.history.replaceState(null, '', `/artifacts/${id}/`)
    }
  }, [])

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
            { label: 'Granite.build', to: '/' },
            { label: 'Artifacts', to: '/artifacts' },
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
