'use client'

import {
  DataTable,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableHeader,
  TableRow,
  InlineNotification,
  SkeletonText,
} from '@carbon/react'
import { useQuery } from '@tanstack/react-query'
import { getArtifactFiles } from '@/api/gbserver'

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

const HEADERS = [
  { key: 'path',     header: 'Path' },
  { key: 'size',     header: 'Size' },
  { key: 'checksum', header: 'Checksum' },
]

export function FilesPanel({ artifactId }: { artifactId: string }) {
  const { data: files, isLoading, error } = useQuery({
    queryKey: ['artifact-files', artifactId],
    queryFn: () => getArtifactFiles(artifactId),
  })

  if (error) return <InlineNotification kind="error" title="Failed to load files" subtitle={String(error)} style={{ margin: '1rem' }} />
  if (isLoading) return <div style={{ padding: '1.5rem' }}><SkeletonText paragraph lineCount={6} /></div>

  const rows = (files ?? []).map((f, i) => ({ id: String(i), ...f }))

  return (
    <div style={{ padding: '1.5rem' }}>
      <p style={{ fontSize: '0.875rem', color: 'var(--cds-text-secondary, #525252)', marginBottom: '0.75rem' }}>
        {rows.length} file{rows.length !== 1 ? 's' : ''}
      </p>
      <DataTable rows={rows} headers={HEADERS} isSortable>
        {({ rows: tableRows, headers, getTableProps, getHeaderProps, getRowProps }) => (
          <TableContainer>
            <Table {...getTableProps()} size="sm">
              <TableHead>
                <TableRow>
                  {headers.map((h) => {
                    const { key: _k, ...hProps } = getHeaderProps({ header: h })
                    return <TableHeader key={h.key} {...hProps}>{h.header}</TableHeader>
                  })}
                </TableRow>
              </TableHead>
              <TableBody>
                {tableRows.map((row) => (
                  <TableRow {...getRowProps({ row })} key={row.id}>
                    {row.cells.map((cell) => (
                      <TableCell
                        key={cell.id}
                        style={{
                          fontFamily: cell.info.header !== 'size' ? 'monospace' : undefined,
                          fontSize: '0.875rem',
                        }}
                      >
                        {cell.info.header === 'size' && typeof cell.value === 'number'
                          ? formatBytes(cell.value)
                          : (cell.value ?? '—')}
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </DataTable>
    </div>
  )
}
