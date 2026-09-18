'use client'

import { useMemo } from 'react'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@carbon/react'

export interface PreviewTableHeader {
  key: string
  header: string
}

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
  /** Rendered in place of the table when there is nothing to show. */
  emptyMessage?: string
}

export function PreviewTable({
  rows,
  headers,
  maxRows,
  maxCellChars,
  emptyMessage,
}: PreviewTableProps) {
  const resolvedHeaders = useMemo<PreviewTableHeader[]>(() => {
    if (headers) return headers
    const keys = Array.from(new Set(rows.flatMap((row) => Object.keys(row))))
    return keys.map((key) => ({ key, header: key }))
  }, [headers, rows])

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
              <TableHeader key={header.key}>{header.header}</TableHeader>
            ))}
          </TableRow>
        </TableHead>
        <TableBody>
          {visibleRows.map((row, index) => (
            <TableRow key={index}>
              {resolvedHeaders.map((header) => {
                const value = row[header.key]
                // A null cell renders empty. One of the two original copies rendered
                // the literal text "null" here, because typeof null === 'object'
                // routed it into JSON.stringify.
                const text =
                  value == null ? '' : typeof value === 'string' ? value : JSON.stringify(value)
                return (
                  <TableCell key={header.key}>
                    {maxCellChars != null && text.length > maxCellChars
                      ? `${text.slice(0, maxCellChars)}...`
                      : text}
                  </TableCell>
                )
              })}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}
