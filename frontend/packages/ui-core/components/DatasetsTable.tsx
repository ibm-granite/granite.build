'use client'

import { useEffect, useMemo, useState } from 'react'
import {
  DataTable,
  DataTableSkeleton,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableHeader,
  TableRow,
  TableToolbar,
  TableToolbarContent,
  TableToolbarSearch,
  TableSelectAll,
  TableSelectRow,
  TableBatchActions,
  TableBatchAction,
  Pagination,
  Button,
  Toggle,
  Link as CarbonLink,
  InlineNotification,
} from '@carbon/react'
import { Add, TrashCan } from '@carbon/icons-react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import axios from 'axios'
import type { Dataset } from '../types'
import { getDatasets, deleteDataset } from '../api/autotunex'
import { listSpaces } from '../api/gbserver'
import { SettingsDeleteModal } from './SettingsDeleteModal'
import { SettingsDatasetView } from './SettingsDatasetView'
import { SettingsDatasetCreate } from './SettingsDatasetCreate'

const HEADERS = [
  { key: 'name', header: 'Name' },
  { key: 'train_records', header: 'Training samples' },
  { key: 'validation_records', header: 'Validation samples' },
  { key: 'created_at', header: 'Created on' },
]

function formatCompact(n: number): string {
  if (n == null || Number.isNaN(n)) return '0'
  return new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(n)
}

function isUndeletable(d: Dataset): boolean {
  return (d.associated_jobs?.length ?? 0) > 0
}

export function DatasetsTable() {
  const queryClient = useQueryClient()
  const [selectedIds, setSelectedIds] = useState<string[]>([])
  const [deleteOpen, setDeleteOpen] = useState(false)
  const [deleteError, setDeleteError] = useState<string | undefined>(undefined)
  const [createOpen, setCreateOpen] = useState(false)
  const [viewId, setViewId] = useState<string | null>(null)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(10)
  const [searchInput, setSearchInput] = useState('')
  const [q, setQ] = useState('')
  const [scope, setScope] = useState<'own' | 'all'>('own')

  // Debounce free-text search into `q` and reset to page 1 on change.
  useEffect(() => {
    const timer = setTimeout(() => {
      setQ(searchInput)
      setPage(1)
    }, 300)
    return () => clearTimeout(timer)
  }, [searchInput])

  // `selectedIds` shadows Carbon's own selection, and Carbon rebuilds its
  // checkboxes from whatever rows it is handed. A selection left over from a
  // previous page/size/search/scope would therefore stay in `selectedIds`
  // while disappearing from the UI — the batch bar and the confirmation count
  // would disagree, and the delete would remove datasets the user can no longer
  // see. It also keeps `anyUndeletable` honest, since `byId` only holds the
  // current page. Clear it whenever the visible set changes.
  useEffect(() => {
    setSelectedIds((prev) => (prev.length === 0 ? prev : []))
  }, [page, pageSize, q, scope])

  // No "current active space" concept exists in this dashboard (no space
  // context/provider), so the own/all scope toggle is gated on the viewer
  // being an admin of at least one space — same convention used by the
  // tunings list. Shared `["spaces"]` queryKey avoids a duplicate fetch.
  const { data: spaces = [] } = useQuery({
    queryKey: ['spaces'],
    queryFn: listSpaces,
  })
  const isSpaceAdmin = spaces.some((s) => s.is_admin)

  const { data, isLoading } = useQuery({
    queryKey: ['autotunex', 'datasets', page, pageSize, q, scope],
    queryFn: () => getDatasets({ page, pageSize, q: q || undefined, scope }),
    placeholderData: (prev) => prev,
  })
  const items = data?.items ?? []
  const total = data?.total ?? 0

  const byId = useMemo(() => new Map(items.map((d) => [d.id, d])), [items])
  const selectedDatasets = selectedIds.map((id) => byId.get(id)).filter(Boolean) as Dataset[]
  const anyUndeletable = selectedDatasets.some(isUndeletable)

  const deleteMutation = useMutation({
    mutationFn: async (ids: string[]) => {
      for (const id of ids) await deleteDataset(id, scope)
    },
    onSuccess: (_data, ids) => {
      queryClient.invalidateQueries({ queryKey: ['autotunex', 'datasets'] })
      setSelectedIds([])
      setDeleteOpen(false)
      setDeleteError(undefined)
      // If the delete emptied the last page, clamp back onto the new last
      // page and let the invalidated query above refetch it — no in-memory
      // re-slicing of a locally-shrunk array.
      const newTotal = Math.max(0, total - ids.length)
      const lastPage = Math.max(1, Math.ceil(newTotal / pageSize))
      if (page > lastPage) setPage(lastPage)
    },
    onError: (err) => {
      if (axios.isAxiosError(err) && err.response?.status === 409) {
        setDeleteError('This dataset is in use by a running job and cannot be deleted.')
      } else {
        setDeleteError('Something went wrong while deleting. Please try again.')
      }
    },
  })

  const rows = items.map((d) => ({
    id: d.id,
    name: d.name,
    train_records: formatCompact(d.train_records),
    validation_records: formatCompact(d.validation_records),
    created_at: d.created_at ?? '',
  }))

  if (isLoading) {
    return <DataTableSkeleton headers={HEADERS} rowCount={5} showHeader={false} showToolbar={false} />
  }

  return (
    <>
      <DataTable rows={rows} headers={HEADERS} isSortable>
        {({ rows: tableRows, headers, getTableProps, getHeaderProps, getRowProps, getSelectionProps, getBatchActionProps }) => {
          const batchActionProps = getBatchActionProps()
          return (
            <TableContainer
              title="Data sets"
              description="Uploaded data sets available for tuning, with their training and validation sample counts."
            >
              <TableToolbar>
                <TableBatchActions {...batchActionProps} onCancel={() => setSelectedIds([])}>
                  <TableBatchAction
                    renderIcon={TrashCan}
                    disabled={anyUndeletable}
                    onClick={() => { setDeleteError(undefined); setDeleteOpen(true) }}
                  >
                    Delete
                  </TableBatchAction>
                </TableBatchActions>
                <TableToolbarContent>
                  <TableToolbarSearch
                    persistent
                    placeholder="Search datasets…"
                    onChange={(_e, value) => setSearchInput(value ?? '')}
                  />
                  {isSpaceAdmin && (
                    <Toggle
                      id="datasets-scope-toggle"
                      labelText=""
                      labelA="Mine"
                      labelB="All"
                      toggled={scope === 'all'}
                      onToggle={(checked) => { setScope(checked ? 'all' : 'own'); setPage(1) }}
                      size="sm"
                    />
                  )}
                  <Button renderIcon={Add} onClick={() => setCreateOpen(true)}>
                    Create New Dataset
                  </Button>
                </TableToolbarContent>
              </TableToolbar>
              <Table {...getTableProps()} size="md">
                <TableHead>
                  <TableRow>
                    <TableSelectAll
                      {...getSelectionProps()}
                      onSelect={(e) => {
                        getSelectionProps().onSelect(e)
                        setSelectedIds((e.target as HTMLInputElement).checked ? tableRows.map((r) => r.id) : [])
                      }}
                    />
                    {headers.map((h) => {
                      const { key: _k, ...hProps } = getHeaderProps({ header: h })
                      return <TableHeader key={h.key} {...hProps}>{h.header}</TableHeader>
                    })}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {tableRows.map((row) => {
                    const { key: _k, ...rowProps } = getRowProps({ row })
                    const selectionProps = getSelectionProps({ row })
                    return (
                      <TableRow key={row.id} {...rowProps}>
                        <TableSelectRow
                          {...selectionProps}
                          onSelect={(e) => {
                            selectionProps.onSelect(e)
                            const checked = (e.target as HTMLInputElement).checked
                            setSelectedIds((prev) => (checked ? [...prev, row.id] : prev.filter((id) => id !== row.id)))
                          }}
                        />
                        {row.cells.map((cell) => (
                          <TableCell key={cell.id}>
                            {cell.info.header === 'name' ? (
                              <CarbonLink href="#" onClick={(e) => { e.preventDefault(); setViewId(row.id) }}>
                                {cell.value}
                              </CarbonLink>
                            ) : cell.info.header === 'created_at' ? (
                              cell.value ? new Date(cell.value as string).toLocaleString() : '—'
                            ) : (
                              (cell.value as React.ReactNode)
                            )}
                          </TableCell>
                        ))}
                      </TableRow>
                    )
                  })}
                </TableBody>
              </Table>
              <Pagination
                totalItems={total}
                pageSize={pageSize}
                page={page}
                pageSizes={[10, 20, 50]}
                onChange={({ page: p, pageSize: ps }) => { setPage(p); setPageSize(ps) }}
              />
            </TableContainer>
          )
        }}
      </DataTable>

      {anyUndeletable && selectedIds.length > 0 && (
        <InlineNotification
          kind="info"
          title="Some selected datasets can't be deleted"
          subtitle="Datasets in use by a tuning cannot be deleted."
          lowContrast
          hideCloseButton
          style={{ marginTop: '0.5rem' }}
        />
      )}

      <SettingsDeleteModal
        open={deleteOpen}
        count={selectedIds.length}
        itemLabel="dataset"
        isDeleting={deleteMutation.isPending}
        errorMessage={deleteError}
        onClose={() => { setDeleteOpen(false); setDeleteError(undefined) }}
        onConfirm={() => deleteMutation.mutate(selectedIds)}
      />

      <SettingsDatasetCreate
        open={createOpen}
        onClose={() => setCreateOpen(false)}
        onCreated={() => setCreateOpen(false)}
      />

      <SettingsDatasetView open={viewId != null} datasetId={viewId} onClose={() => setViewId(null)} scope={scope} />
    </>
  )
}
