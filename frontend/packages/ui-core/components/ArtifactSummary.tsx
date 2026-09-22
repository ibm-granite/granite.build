'use client'

import * as React from 'react'
import { CopyButton } from '@carbon/react'
import styles from './ArtifactSummary.module.scss'
import { getHuggingFaceUrl } from './LineageGraph/diagramUtilities'

/**
 * The minimum needed to identify an artifact. Deliberately not `Artifact`: the
 * lineage graph knows a node's id, title and type but has not necessarily
 * fetched the full record, and requiring one would force a request just to
 * label a dialog.
 */
export interface ArtifactSummaryFields {
  /**
   * Artifact UUID, when known. Optional because lineage refs carry only a name
   * and URI — the graph can describe such a node but cannot cite its id, and
   * showing an empty "Artifact ID" would look like data loss.
   */
  id?: string
  /** Human-readable name, when known. */
  name?: string
  /** Artifact type as the API spells it (`MODEL`, `model`, `Fileset`, …). */
  artifactType?: string
  /** Artifact URI, rendered as a HuggingFace link when it resolves to one. */
  uri?: string
}

function DetailField({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className={styles.fieldLabel}>{label}</div>
      <div className={styles.fieldValue}>{children}</div>
    </div>
  )
}

/**
 * Compact identity block for an artifact — id, name, type and URI.
 *
 * Exists because the lineage graph's artifact dialog said nothing useful: it
 * asked whether to navigate and showed no identifying detail at all, so a user
 * who clicked the wrong node had no way to tell. The fields here are the ones
 * the artifact detail page already shows, minus everything that needs the full
 * record.
 *
 * The URI becomes a link only when {@link getHuggingFaceUrl} resolves it, which
 * already understands all four `hf://` spellings; anything else renders as
 * plain monospace text rather than a dead link.
 */
export function ArtifactSummary({ artifact }: { artifact: ArtifactSummaryFields }) {
  const hfUrl = artifact.uri ? getHuggingFaceUrl(artifact.uri) : null

  return (
    <dl className={styles.list}>
      {artifact.name && <DetailField label="Name">{artifact.name}</DetailField>}

      {artifact.id && (
        <DetailField label="Artifact ID">
          <span className={styles.copyRow}>
            <code className={styles.mono}>{artifact.id}</code>
            <CopyButton
              feedback="Copied!"
              iconDescription="Copy artifact ID"
              onClick={() => navigator.clipboard.writeText(artifact.id!)}
              size="sm"
            />
          </span>
        </DetailField>
      )}

      {artifact.artifactType && (
        <DetailField label="Artifact Type">{artifact.artifactType.toLowerCase()}</DetailField>
      )}

      {artifact.uri && (
        <DetailField label="URI">
          {hfUrl ? (
            <a
              href={hfUrl}
              target="_blank"
              rel="noopener noreferrer"
              className={`${styles.mono} ${styles.link}`}
            >
              {artifact.uri}
            </a>
          ) : (
            <span className={styles.mono}>{artifact.uri}</span>
          )}
        </DetailField>
      )}
    </dl>
  )
}
