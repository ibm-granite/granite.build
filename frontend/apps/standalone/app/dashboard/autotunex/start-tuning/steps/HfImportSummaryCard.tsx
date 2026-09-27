'use client'

import { Button, InlineNotification, Tag } from '@carbon/react'
import { Edit } from '@carbon/icons-react'
import { hfMissingRequiredColumns, hfSnapshotSummary, type HfImportSnapshot } from './hfImport'
import styles from './HfImportSummaryCard.module.scss'

interface HfImportSummaryCardProps {
  snapshot: HfImportSnapshot
  /** The currently selected algorithm's required columns, to check the frozen mapping against. */
  requiredColumns: string[]
  /** Drops the frozen request; the HuggingFace form comes back in its place. */
  onChange: () => void
}

/**
 * The frozen HuggingFace request, shown read-only when Step 1 is revisited after
 * Next. The form is not rebuilt instead because its probe clears the column mapping
 * on mount and the AI re-suggests it, which could silently replace a hand-corrected
 * mapping before Launch.
 */
export function HfImportSummaryCard({ snapshot, requiredColumns, onChange }: HfImportSummaryCardProps) {
  const summary = hfSnapshotSummary(snapshot)
  const missingColumns = hfMissingRequiredColumns(snapshot.payload.column_mapping, requiredColumns)
  return (
    <div className={styles.card}>
      <div className={styles.header}>
        <span className={styles.title}>HuggingFace dataset</span>
        <Tag type="cyan" size="sm">Imports at launch</Tag>
      </div>
      {missingColumns.length > 0 && (
        <InlineNotification
          kind="warning"
          lowContrast
          hideCloseButton
          title={`This mapping doesn't cover ${missingColumns.join(', ')} required by the selected algorithm.`}
          subtitle="Click Change to remap."
        />
      )}
      <dl className={styles.rows}>
        <dt>Name</dt>
        <dd>{summary.name}</dd>
        <dt>Source</dt>
        <dd>{summary.source}</dd>
        <dt>Config</dt>
        <dd>{summary.config}</dd>
        <dt>Train</dt>
        <dd>{summary.trainSplit}</dd>
        <dt>Validation</dt>
        <dd>{summary.validation}</dd>
        <dt>Mapping</dt>
        <dd>
          <ul className={styles.mapping}>
            {summary.mapping.map(({ target, source }) => (
              <li key={target}>
                <code>{target}</code> ← <code>{source}</code>
              </li>
            ))}
          </ul>
        </dd>
      </dl>
      <p className={styles.usable}>{summary.usable}</p>
      <Button kind="tertiary" size="sm" renderIcon={Edit} onClick={onChange}>
        Change
      </Button>
    </div>
  )
}
