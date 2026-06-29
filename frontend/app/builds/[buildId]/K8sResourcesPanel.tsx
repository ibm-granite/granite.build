'use client'

import type React from 'react'
import { Accordion, AccordionItem, Tag } from '@carbon/react'
import type { K8sResource, BuildStatus } from '@/types'
import { BuildStatusBadge } from '@/components/BuildStatusBadge'

interface Props {
  resources: K8sResource[]
}

function normalizeK8sStatus(s: string): BuildStatus {
  if (s === 'succeeded') return 'success'
  if (s === 'error') return 'failed'
  return s as BuildStatus
}

export function K8sResourcesPanel({ resources }: Props) {
  if (!resources || resources.length === 0) {
    return <p style={{ padding: '1rem', color: '#525252' }}>No K8s resources found for this build.</p>
  }

  const byKind = resources.reduce<Record<string, K8sResource[]>>((acc, r) => {
    ;(acc[r.kind] ??= []).push(r)
    return acc
  }, {})

  const thStyle: React.CSSProperties = {
    padding: '0.75rem 1rem',
    fontSize: '0.875rem',
    fontWeight: 600,
    color: 'var(--cds-text-secondary)',
    textAlign: 'left',
    whiteSpace: 'nowrap',
  }
  const tdStyle: React.CSSProperties = {
    padding: '0.75rem 1rem',
    fontSize: '0.875rem',
    verticalAlign: 'top',
  }

  return (
    <Accordion>
      {Object.entries(byKind).map(([kind, items]) => (
        <AccordionItem key={kind} title={`${kind} (${items.length})`} open>
          <table style={{ width: '100%', borderCollapse: 'collapse', tableLayout: 'fixed' }}>
            <colgroup>
              <col />
              <col style={{ width: '160px' }} />
              <col style={{ width: '120px' }} />
              <col style={{ width: '72px' }} />
              <col style={{ width: '88px' }} />
              <col style={{ width: '72px' }} />
            </colgroup>
            <thead>
              <tr>
                <th style={thStyle}>Name</th>
                <th style={thStyle}>Namespace</th>
                <th style={thStyle}>Status</th>
                <th style={thStyle}>CPU</th>
                <th style={thStyle}>Memory</th>
                <th style={thStyle}>GPU</th>
              </tr>
            </thead>
            <tbody>
              {items.map((r, i) => (
                <tr key={i}>
                  <td style={tdStyle}>
                    <code style={{ fontSize: '0.875rem' }}>{r.name}</code>
                    {r.failure_message && (
                      <p style={{ margin: '0.25rem 0 0', fontSize: '0.875rem', color: '#da1e28' }}>
                        {r.failure_message}
                      </p>
                    )}
                  </td>
                  <td style={{ ...tdStyle, whiteSpace: 'nowrap', color: 'var(--cds-text-secondary)' }}>
                    {r.namespace ?? '—'}
                  </td>
                  <td style={{ ...tdStyle, whiteSpace: 'nowrap' }}>
                    {r.build_status
                      ? <span title={r.failure_reason ?? undefined}><BuildStatusBadge status={normalizeK8sStatus(r.build_status)} /></span>
                      : (r.status ?? '—')}
                  </td>
                  <td style={{ ...tdStyle, whiteSpace: 'nowrap' }}>{r.cpu ?? '—'}</td>
                  <td style={{ ...tdStyle, whiteSpace: 'nowrap' }}>{r.memory ?? '—'}</td>
                  <td style={{ ...tdStyle, whiteSpace: 'nowrap' }}>
                    {r.gpu != null && r.gpu > 0 ? <Tag type="purple" size="sm">×{r.gpu}</Tag> : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </AccordionItem>
      ))}
    </Accordion>
  )
}
