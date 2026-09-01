'use client'

import { Tag, Tooltip } from '@carbon/react'

const TAG_COLORS = ['blue', 'purple', 'teal', 'magenta'] as const

const TAG_PAD_REM = 1.5
const CHAR_REM = 0.6
const GAP_REM = 0.25
const MORE_REM = 4
const CELL_REM = 20
const MAX_TAGS = 4

function splitTags(tags: string[]): { visible: string[]; hidden: string[] } {
  for (let n = Math.min(tags.length, MAX_TAGS); n >= 0; n--) {
    const visible = tags.slice(0, n)
    const hidden = tags.slice(n)
    const tagsWidth = visible.reduce((w, t, i) => w + (i > 0 ? GAP_REM : 0) + TAG_PAD_REM + t.length * CHAR_REM, 0)
    const moreWidth = hidden.length > 0 ? (n > 0 ? GAP_REM : 0) + MORE_REM : 0
    if (tagsWidth + moreWidth <= CELL_REM) return { visible, hidden }
  }
  return { visible: [], hidden: tags }
}

interface Props {
  tags: string[]
}

export function TagsCell({ tags }: Props) {
  if (!tags || tags.length === 0) return null

  const { visible, hidden } = splitTags(tags)

  return (
    <span style={{ display: 'flex', flexWrap: 'nowrap', overflow: 'hidden', gap: '0.25rem', alignItems: 'center' }}>
      {visible.map((t, i) => (
        <Tag key={t} type={TAG_COLORS[i % TAG_COLORS.length]} size="sm">{t}</Tag>
      ))}
      {hidden.length > 0 && (
        <Tooltip label={hidden.join(', ')} align="bottom">
          <span
            className="cds--helper-text-01"
            style={{ color: 'var(--cds-text-secondary)', whiteSpace: 'nowrap', cursor: 'default' }}
          >
            +{hidden.length} more
          </span>
        </Tooltip>
      )}
    </span>
  )
}
