'use client'

import { CodeSnippet, Layer, SkeletonText } from '@carbon/react'
import styles from './DefinitionPanel.module.scss'
import { useQuery } from '@tanstack/react-query'
import { getBuildArchiveFiles } from '../api/gbserver'

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
    /* The review's headline finding was that this view "looks too plain,
       probably because of the all-white background", against an older view that
       rendered the definition on gray.

       The cause is a token that does not mean what the old code assumed:
       `--cds-layer` is layer-01, and in the g10 theme both apps use, layer-01 is
       #ffffff while `background` is #f4f4f4 — the inverse of the `white` theme.
       So `background: var(--cds-layer)` painted white on white.

       <Layer> moves this subtree one level deeper, where `--cds-layer` resolves
       to layer-02: #f4f4f4 in light, and #393939 against layer-01's #262626 in
       g100. Correct in both themes, which a hardcoded gray would not be.

       CodeSnippet alone would not have fixed it — .cds--snippet sets
       `background-color: $layer`, the same contextual token — so the <Layer>
       wrapper is the load-bearing part, not the component swap. */
    <Layer className={styles.definitionLayer}>
      <CodeSnippet
        type="multi"
        feedback="Copied!"
        copyButtonDescription="Copy build definition"
        aria-label="Build definition"
        /* The default cap is 15 rows behind a "Show more" toggle, which would
           hide most of a build.yaml in what is already a full-height tab. 0
           disables the cap: the button is gated on this being > 0. */
        maxCollapsedNumberOfRows={0}
        maxExpandedNumberOfRows={0}
        /* Preserves the previous <pre>'s pre-wrap / break-word behaviour;
           CodeSnippet defaults to no wrapping. */
        wrapText
        className={styles.definitionSnippet}
      >
        {yaml}
      </CodeSnippet>
    </Layer>
  )
}
