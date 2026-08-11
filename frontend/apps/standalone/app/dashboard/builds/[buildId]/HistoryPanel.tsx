'use client'

import {
  StructuredListWrapper,
  StructuredListBody,
  StructuredListRow,
  StructuredListCell,
} from '@carbon/react'
import { ChevronDown } from '@carbon/icons-react'
import { Children, isValidElement, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkBreaks from 'remark-breaks'
import type { BuildEvent } from '@granite-build/ui-core/types'

interface Props {
  events: BuildEvent[]
}

const COLLAPSED_LINE_LIMIT = 10
// 10 lines × (0.75rem font × 1.5 line-height) + 2rem vertical padding
const COLLAPSED_MAX_HEIGHT = 'calc(10 * 1.125rem + 2rem)'

function CollapsiblePre({ content }: { content: string }) {
  const isLong = content.split('\n').length > COLLAPSED_LINE_LIMIT
  const [expanded, setExpanded] = useState(false)
  const [hovered, setHovered] = useState(false)

  return (
    <div style={{ position: 'relative', margin: '0.5rem 0' }}>
      <pre style={{
        background: 'var(--cds-layer, #f4f4f4)',
        padding: '1rem',
        overflowX: 'auto',
        overflowY: 'hidden',
        margin: 0,
        fontFamily: 'IBM Plex Mono, monospace',
        fontSize: '0.75rem',
        lineHeight: '1.5',
        whiteSpace: 'pre-wrap',
        wordBreak: 'break-all',
        ...(isLong && !expanded ? { maxHeight: COLLAPSED_MAX_HEIGHT } : {}),
      }}>
        {content}
      </pre>

      {/* Gradient overlay — sits above content but below the border, so the border stays sharp */}
      {isLong && !expanded && (
        <div
          aria-hidden
          style={{
            position: 'absolute',
            bottom: 0,
            left: 0,
            right: 0,
            height: '3rem',
            background: 'linear-gradient(to bottom, transparent, var(--cds-layer, #f4f4f4) 55%)',
            pointerEvents: 'none',
          }}
        />
      )}

      {isLong && (
        <button
          onClick={() => setExpanded((e) => !e)}
          onMouseEnter={() => setHovered(true)}
          onMouseLeave={() => setHovered(false)}
          style={{
            position: 'absolute',
            insetBlockEnd: 0,
            insetInlineEnd: 0,
            zIndex: 10,
            display: 'inline-flex',
            alignItems: 'center',
            height: '2rem',
            padding: '0 1rem',
            border: 0,
            background: hovered
              ? 'var(--cds-layer-hover, #e8e8e8)'
              : 'var(--cds-layer, #f4f4f4)',
            color: 'var(--cds-text-primary)',
            fontSize: '0.875rem',
            lineHeight: '1.25rem',
            fontFamily: 'var(--cds-body-compact-01-font-family, sans-serif)',
            cursor: 'pointer',
          }}
        >
          <span style={{ position: 'relative', insetBlockStart: '-1px' }}>
            {expanded ? 'Show less' : 'Show more'}
          </span>
          <ChevronDown
            style={{
              fill: 'var(--cds-icon-primary)',
              marginInlineStart: '0.5rem',
              transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
              transition: 'transform 110ms cubic-bezier(0.2, 0, 0.38, 0.9)',
            }}
          />
        </button>
      )}
    </div>
  )
}

export function HistoryPanel({ events }: Props) {
  if (!events || events.length === 0) {
    return <p style={{ padding: '1rem' }}>No history events.</p>
  }

  return (
    <StructuredListWrapper>
      <StructuredListBody>
        {events.filter((ev) => ev.description?.trim()).map((ev, i) => (
          <StructuredListRow key={i} style={i === 0 ? { borderBlockStart: 'none' } : undefined}>
            <StructuredListCell noWrap>
              {new Date(ev.time).toLocaleString()}
            </StructuredListCell>
            <StructuredListCell>
              <div style={{
                fontFamily: 'var(--cds-body-01-font-family, "IBM Plex Sans", sans-serif)',
                fontSize: 'var(--cds-body-01-font-size, 0.875rem)',
                fontWeight: 'var(--cds-body-01-font-weight, 400)',
                lineHeight: 'var(--cds-body-01-line-height, 1.42857)',
                letterSpacing: 'var(--cds-body-01-letter-spacing, 0.16px)',
              }}>
              <ReactMarkdown
                remarkPlugins={[remarkBreaks]}
                components={{
                  pre({ children }) {
                    let content = ''
                    Children.forEach(children, (child) => {
                      if (isValidElement(child)) {
                        content = String((child.props as { children?: unknown }).children ?? '').replace(/\n$/, '')
                      }
                    })
                    return <CollapsiblePre content={content} />
                  },
                  code({ children }) {
                    return (
                      <code style={{
                        fontFamily: '"IBM Plex Mono", "Menlo", monospace',
                        fontSize: '0.875em',
                        background: 'var(--cds-layer-accent, #e0e0e0)',
                        padding: '0 0.25rem',
                        borderRadius: '2px',
                      }}>
                        {String(children)}
                      </code>
                    )
                  },
                  h1: ({ children }) => <h4 style={{ marginTop: '0.75rem', marginBottom: '0.25rem', fontWeight: 600 }}>{children}</h4>,
                  h2: ({ children }) => <h5 style={{ marginTop: '0.75rem', marginBottom: '0.25rem', fontWeight: 600 }}>{children}</h5>,
                  h3: ({ children }) => <h6 style={{ marginTop: '0.5rem', marginBottom: '0.25rem', fontWeight: 600 }}>{children}</h6>,
                  p: ({ children }) => <p style={{ marginTop: 0, marginBottom: '0.5rem', fontSize: '0.875rem', lineHeight: 1.42857, letterSpacing: '0.16px' }}>{children}</p>,
                }}
              >
                {ev.description}
              </ReactMarkdown>
              </div>
            </StructuredListCell>
          </StructuredListRow>
        ))}
      </StructuredListBody>
    </StructuredListWrapper>
  )
}
