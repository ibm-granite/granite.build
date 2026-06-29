'use client'

import type { BuildStatus } from '@/types'

const STATUS_CONFIG: Record<BuildStatus, { label: string; color: string }> = {
  success:   { label: 'Success',   color: 'var(--cds-support-success)' },
  failed:    { label: 'Failed',    color: 'var(--cds-support-error)' },
  running:   { label: 'Running',   color: 'var(--cds-support-info)' },
  pending:   { label: 'Pending',   color: 'var(--cds-icon-disabled)' },
  submitted: { label: 'Submitted', color: 'var(--cds-support-info)' },
  suspended: { label: 'Suspended', color: 'var(--cds-support-warning)' },
  cancelled: { label: 'Cancelled', color: 'var(--cds-icon-disabled)' },
  deleted:   { label: 'Deleted',   color: 'var(--cds-icon-disabled)' },
  planned:   { label: 'Planned',   color: 'var(--cds-border-subtle-01)' },
}

interface Props {
  status: BuildStatus
  showLabel?: boolean
}

export function BuildStatusBadge({ status, showLabel = true }: Props) {
  const cfg = STATUS_CONFIG[status] ?? STATUS_CONFIG.cancelled
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.375rem' }}>
      <span
        aria-hidden="true"
        style={{
          display: 'inline-block',
          width: 10,
          height: 10,
          borderRadius: '50%',
          backgroundColor: cfg.color,
          flexShrink: 0,
        }}
      />
      {showLabel && (
        <span style={{ fontSize: '0.875rem', lineHeight: 1 }}>{cfg.label}</span>
      )}
    </span>
  )
}
