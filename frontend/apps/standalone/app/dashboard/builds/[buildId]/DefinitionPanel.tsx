'use client'

import { CopyButton, SkeletonText } from '@carbon/react'
import { useQuery } from '@tanstack/react-query'
import { getBuildArchiveFiles } from '@granite-build/ui-core/api/gbserver'

interface Props {
  buildId: string
}

export function DefinitionPanel({ buildId }: Props) {
  const { data, isLoading, error } = useQuery({
    queryKey: ['build-archive', buildId],
    queryFn: () => getBuildArchiveFiles(buildId),
    staleTime: 60_000,
  })

  if (isLoading) {
    return <div style={{ padding: '1rem' }}><SkeletonText paragraph lineCount={8} /></div>
  }

  if (error || !data) {
    return <p style={{ padding: '1rem', color: 'var(--cds-text-secondary)' }}>No build definition available.</p>
  }

  const yaml =
    data['build.yaml'] ??
    data[Object.keys(data).find((k) => k.endsWith('.yaml') || k.endsWith('.yml')) ?? ''] ??
    (Object.keys(data).length ? JSON.stringify(data, null, 2) : null)

  if (!yaml) {
    return <p style={{ padding: '1rem', color: 'var(--cds-text-secondary)' }}>No build definition available.</p>
  }

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', justifyContent: 'flex-end', padding: '0.25rem 0.5rem', flexShrink: 0 }}>
        <CopyButton
          autoAlign
          feedback="Copied!"
          iconDescription="Copy definition"
          onClick={() => navigator.clipboard.writeText(yaml)}
          size="sm"
        />
      </div>
      <pre style={{
        margin: 0,
        padding: '0.5rem 2rem 1rem',
        flex: 1,
        overflow: 'auto',
        whiteSpace: 'pre-wrap',
        wordBreak: 'break-word',
        fontSize: '0.8125rem',
        lineHeight: 1.6,
        fontFamily: 'IBM Plex Mono, monospace',
        background: 'var(--cds-layer)',
      }}>
        {yaml}
      </pre>
    </div>
  )
}
