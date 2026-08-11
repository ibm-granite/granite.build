'use client'

import type { BuildStatus } from '../types'

// Carbon's shape-indicator pattern (carbondesignsystem.com/patterns/status-indicator-pattern)
// has no ready-made icon set — implementers draw the shapes themselves. Drawn
// here as plain SVG primitives (16x16 viewBox, matching Carbon's icon grid)
// rather than picked from @carbon/icons-react, since Carbon's icon library is
// semantic icons, not raw geometric shapes, and using a semantic icon that
// merely "looked close" didn't match the guideline's actual shapes.
type ShapeKind = 'circle' | 'circle-outline' | 'triangle' | 'triangle-outline' | 'diamond' | 'square' | 'prohibit'

function Shape({ kind, color, size }: { kind: ShapeKind; color: string; size: number }) {
  const common = { width: size, height: size, viewBox: '0 0 16 16', 'aria-hidden': true, style: { flexShrink: 0 } }
  switch (kind) {
    case 'circle':
      return <svg {...common}><circle cx="8" cy="8" r="6.5" fill={color} /></svg>
    case 'circle-outline':
      return <svg {...common}><circle cx="8" cy="8" r="5.75" fill="none" stroke={color} strokeWidth="1.5" /></svg>
    case 'triangle':
      return <svg {...common}><polygon points="8,1.5 14.5,14 1.5,14" fill={color} /></svg>
    case 'triangle-outline':
      return <svg {...common}><polygon points="8,1.5 14.5,14 1.5,14" fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" /></svg>
    case 'diamond':
      return <svg {...common}><polygon points="8,1.5 14.5,8 8,14.5 1.5,8" fill={color} /></svg>
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

// Carbon's status color palette (carbondesignsystem.com/patterns/status-indicator-pattern
// #status-color-palette). Red/green/purple/blue are real theme-adaptive Carbon
// CSS custom properties (verified against the compiled theme CSS); orange,
// yellow, and gray are Carbon's raw color-ramp values (Orange 40, Yellow 30,
// Gray 60), which Carbon itself does not re-theme between light and dark.
const RED = 'var(--cds-support-error)'
const GREEN = 'var(--cds-support-success)'
const ORANGE = 'var(--cds-support-caution-major)'
const YELLOW = 'var(--cds-support-caution-minor)'
const BLUE = 'var(--cds-support-info)'
const PURPLE = 'var(--cds-support-caution-undefined)'
const GRAY = '#6f6f6f'

// Shapes are reused across semantically related statuses (e.g. running/pending/
// planned are all "circle-outline", failed/cancelled are both "prohibit"),
// matching how Carbon's own reference table reuses the same triangle for
// Critical/High and the same gray across Not-started/Pending/Unknown —
// distinguished by color within the pair.
const STATUS_CONFIG: Record<BuildStatus, { label: string; color: string; shape: ShapeKind }> = {
  running:          { label: 'Running',    color: BLUE,   shape: 'circle-outline' },   // "Incomplete"/"In progress"
  success:          { label: 'Success',    color: GREEN,  shape: 'circle' },           // "Stable"
  failed:           { label: 'Failed',     color: RED,    shape: 'prohibit' },         // "Failed"
  invalid:          { label: 'Invalid',    color: PURPLE, shape: 'diamond' },          // "Undefined"
  pending:          { label: 'Pending',    color: GRAY,   shape: 'circle-outline' },   // "Draft"/"Not started"
  submitted:        { label: 'Submitted',  color: BLUE,   shape: 'square' },           // "Informative"
  retry_pending:    { label: 'Retrying',   color: YELLOW, shape: 'triangle-outline' }, // "Cautious"
  cancel_requested: { label: 'Cancelling', color: ORANGE, shape: 'triangle' },         // "Critical"/"High"
  cancelled:        { label: 'Cancelled',  color: GRAY,   shape: 'prohibit' },         // no direct match
  planned:          { label: 'Planned',    color: GRAY,   shape: 'circle-outline' },  // "Draft"/"Not started"
}

interface Props {
  status: BuildStatus
  showLabel?: boolean
}

export function BuildStatusBadge({ status, showLabel = true }: Props) {
  const cfg = STATUS_CONFIG[status] ?? STATUS_CONFIG.cancelled
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.375rem' }}>
      <Shape kind={cfg.shape} color={cfg.color} size={16} />
      {showLabel && (
        <span style={{ fontSize: '0.875rem', lineHeight: 1 }}>{cfg.label}</span>
      )}
    </span>
  )
}
