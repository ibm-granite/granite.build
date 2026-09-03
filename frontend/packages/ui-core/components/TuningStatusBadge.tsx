'use client'

import type { TuningStatus } from '../types'

// Mirrors BuildStatusBadge's shape+color convention (frontend/components/BuildStatusBadge.tsx)
// applied to the 7 tuning-job statuses instead of Build's 10.
type ShapeKind = 'circle' | 'circle-outline' | 'triangle-outline' | 'prohibit' | 'square'

function Shape({ kind, color, size }: { kind: ShapeKind; color: string; size: number }) {
  const common = { width: size, height: size, viewBox: '0 0 16 16', 'aria-hidden': true, style: { flexShrink: 0 } }
  switch (kind) {
    case 'circle':
      return <svg {...common}><circle cx="8" cy="8" r="6.5" fill={color} /></svg>
    case 'circle-outline':
      return <svg {...common}><circle cx="8" cy="8" r="5.75" fill="none" stroke={color} strokeWidth="1.5" /></svg>
    case 'triangle-outline':
      return <svg {...common}><polygon points="8,1.5 14.5,14 1.5,14" fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" /></svg>
    case 'square':
      return <svg {...common}><rect x="2.5" y="2.5" width="11" height="11" fill={color} /></svg>
    case 'prohibit':
      return (
        <svg {...common}>
          <circle cx="8" cy="8" r="6.5" fill={color} />
          <line x1="4" y1="12" x2="12" y2="4" stroke="white" strokeWidth="1.5" />
        </svg>
      )
  }
}

const RED = 'var(--cds-support-error)'
const GREEN = 'var(--cds-support-success)'
const BLUE = 'var(--cds-support-info)'
const GRAY = '#6f6f6f'

const STATUS_CONFIG: Record<TuningStatus, { label: string; color: string; shape: ShapeKind }> = {
  running:    { label: 'Running',    color: BLUE,  shape: 'circle-outline' },
  completed:  { label: 'Completed',  color: GREEN, shape: 'circle' },
  error:      { label: 'Error',      color: RED,   shape: 'prohibit' },
  terminated: { label: 'Terminated', color: GRAY,  shape: 'prohibit' },
  pending:    { label: 'Pending',    color: GRAY,  shape: 'circle-outline' },
  paused:     { label: 'Paused',     color: GRAY,  shape: 'triangle-outline' },
}

interface Props {
  status: TuningStatus
  showLabel?: boolean
}

export function TuningStatusBadge({ status, showLabel = true }: Props) {
  // `SUBMITTED` no longer exists as a status — any unrecognized/legacy value
  // (or a bad server payload) falls back to `pending` per the migration plan.
  const cfg = STATUS_CONFIG[status] ?? STATUS_CONFIG.pending
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.375rem' }}>
      <Shape kind={cfg.shape} color={cfg.color} size={16} />
      {showLabel && <span style={{ fontSize: '0.875rem', lineHeight: 1 }}>{cfg.label}</span>}
    </span>
  )
}
