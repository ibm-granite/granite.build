// Copyright Granite.Build Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
import { getArtifactContents } from '@granite-build/ui-core/api/gbserver'

export function ContentsPanel({ artifactId }: { artifactId: string }) {
  const { data, isLoading, error } = useQuery({
    queryKey: ['artifact-contents', artifactId],
    queryFn: () => getArtifactContents(artifactId),
  })

  if (error) return <InlineNotification kind="error" title="Failed to load contents" subtitle={String(error)} style={{ margin: '1rem' }} />
  if (isLoading) return <div style={{ padding: '1.5rem' }}><SkeletonText paragraph lineCount={8} /></div>

  const columns = data?.columns ?? []
  const rows = (data?.rows ?? []).map((row, i) => ({
    id: String(i),
    ...Object.fromEntries(columns.map((col, j) => [col, row[j] ?? null])),
  }))
  const headers = columns.map((c) => ({ key: c, header: c }))

  return (
    <div style={{ padding: '1.5rem' }}>
      <p style={{ fontSize: '0.875rem', color: 'var(--cds-text-secondary, #525252)', marginBottom: '0.75rem' }}>
        {data?.total ?? rows.length} row{(data?.total ?? rows.length) !== 1 ? 's' : ''}
      </p>
      <DataTable rows={rows} headers={headers}>
        {({ rows: tableRows, headers: hs, getTableProps, getHeaderProps, getRowProps }) => (
          <TableContainer>
            <Table {...getTableProps()} size="sm">
              <TableHead>
                <TableRow>
                  {hs.map((h) => {
                    const { key: _k, ...hProps } = getHeaderProps({ header: h })
                    return <TableHeader key={h.key} {...hProps}>{h.header}</TableHeader>
                  })}
                </TableRow>
              </TableHead>
              <TableBody>
                {tableRows.map((row) => (
                  <TableRow {...getRowProps({ row })} key={row.id}>
                    {row.cells.map((cell) => (
                      <TableCell key={cell.id} style={{ fontSize: '0.875rem' }}>
                        {cell.value === null || cell.value === undefined ? '—' : String(cell.value)}
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
