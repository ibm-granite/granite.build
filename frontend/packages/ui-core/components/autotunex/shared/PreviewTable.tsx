'use client'

import { useMemo } from 'react'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@carbon/react'
import { derivePreviewHeaders, previewCellText, type PreviewTableHeader } from '../../../lib/autotunex/previewCell'

export type { PreviewTableHeader }

interface PreviewTableProps {
  rows: Record<string, any>[]
  /**
   * Explicit column order and labels. Omit to derive them from the union of keys
   * across every row -- not row 0's keys, which drops a column absent there on
   * ragged data.
   */
  headers?: PreviewTableHeader[]
  /** Rows rendered at most. Omit for no cap. */
  maxRows?: number
  /** Truncate rendered cell text to this many characters. Omit for no truncation. */
  maxCellChars?: number
  /** Which end of the cell text maxCellChars keeps. Default 'start'. */
  cropFrom?: 'start' | 'end'
  /**
   * Target -> source column. A source column's header gets a second line naming
   * the target(s) it maps to. Blank sources are ignored.
   */
  columnMapping?: Record<string, string>
  /** Rendered in place of the table when there is nothing to show. */
  emptyMessage?: string
}

export function PreviewTable({
  rows,
  headers,
  maxRows,
  maxCellChars,
  cropFrom,
  columnMapping,
  emptyMessage,
}: PreviewTableProps) {
  const resolvedHeaders = useMemo<PreviewTableHeader[]>(() => {
    if (headers) return headers
    return derivePreviewHeaders(rows)
  }, [headers, rows])

  const targetsBySource = useMemo(() => {
    const out: Record<string, string[]> = {}
    for (const [target, source] of Object.entries(columnMapping ?? {})) {
      if (source) (out[source] ??= []).push(target)
    }
    return out
  }, [columnMapping])

  if (rows.length === 0 || resolvedHeaders.length === 0) {
    return emptyMessage ? (
      <p style={{ padding: '1rem 0', color: 'var(--cds-text-secondary, #525252)' }}>{emptyMessage}</p>
    ) : null
  }

  const visibleRows = maxRows == null ? rows : rows.slice(0, maxRows)

  return (
    <div style={{ overflowX: 'auto' }}>
      <Table size="sm">
        <TableHead>
          <TableRow>
            {resolvedHeaders.map((header) => (
              <TableHeader key={header.key}>
                {header.header}
                {targetsBySource[header.key] && (
                  <div style={{ fontWeight: 400, color: 'var(--cds-text-secondary, #525252)' }}>
                    → {targetsBySource[header.key].join(', ')}
                  </div>
                )}
              </TableHeader>
            ))}
          </TableRow>
        </TableHead>
        <TableBody>
          {visibleRows.map((row, index) => (
            <TableRow key={index}>
              {resolvedHeaders.map((header) => {
                const value = row[header.key]
                return <TableCell key={header.key}>{previewCellText(value, maxCellChars, cropFrom)}</TableCell>
              })}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}
