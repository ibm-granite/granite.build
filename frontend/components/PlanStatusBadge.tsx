'use client'

const STATUS_CONFIG: Record<string, { label: string; color: string }> = {
  draft:     { label: 'Draft',     color: 'var(--cds-icon-disabled)' },
  executing: { label: 'Executing', color: 'var(--cds-support-info)' },
  completed: { label: 'Completed', color: 'var(--cds-support-success)' },
  failed:    { label: 'Failed',    color: 'var(--cds-support-error)' },
}

interface Props {
  status: string
}

export function PlanStatusBadge({ status }: Props) {
  const key = status.toLowerCase()
  const cfg = STATUS_CONFIG[key] ?? { label: status, color: 'var(--cds-icon-disabled)' }
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
      <span style={{ fontSize: '0.875rem', lineHeight: 1 }}>{cfg.label}</span>
    </span>
  )
}
