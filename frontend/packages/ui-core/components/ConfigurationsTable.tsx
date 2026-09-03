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
  Modal,
  InlineNotification,
  InlineLoading,
} from '@carbon/react'
import { Add, TrashCan } from '@carbon/icons-react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import axios from 'axios'
import type { Configuration } from '../types'
import { getConfigurations, deleteConfiguration, getConfiguration } from '../api/autotunex'
import { listSpaces } from '../api/gbserver'
import { SettingsDeleteModal } from './SettingsDeleteModal'
import { SettingsConfigCreate } from './SettingsConfigCreate'
import { ConfigDisplay } from './ConfigDisplay'

const SYSTEM_CONFIG_USER_ID = '00000000-0000-0000-0000-000000000001'

const HEADERS = [
  { key: 'name', header: 'Name' },
  { key: 'tunings', header: 'Tunings' },
  { key: 'created_at', header: 'Created on' },
]

function isUndeletable(c: Configuration): boolean {
  return (c.associated_jobs?.length ?? 0) > 0 || c.user_id === SYSTEM_CONFIG_USER_ID
}

export function ConfigurationsTable() {
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
    queryKey: ['autotunex', 'configurations', page, pageSize, q, scope],
    queryFn: () => getConfigurations({ page, pageSize, q: q || undefined, scope }),
    placeholderData: (prev) => prev,
  })
  const items = data?.items ?? []
  const total = data?.total ?? 0

  const { data: viewedConfig, isLoading: isViewLoading } = useQuery({
    queryKey: ['autotunex-config', viewId],
    queryFn: () => getConfiguration(viewId as string, scope),
    enabled: viewId != null,
  })

  const byId = useMemo(() => new Map(items.map((c) => [c.id, c])), [items])
  const selectedConfigs = selectedIds.map((id) => byId.get(id)).filter(Boolean) as Configuration[]
  const anyUndeletable = selectedConfigs.some(isUndeletable)

  const deleteMutation = useMutation({
    mutationFn: async (ids: string[]) => {
      for (const id of ids) await deleteConfiguration(id, scope)
    },
    onSuccess: (_data, ids) => {
      queryClient.invalidateQueries({ queryKey: ['autotunex', 'configurations'] })
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
        setDeleteError('This configuration is in use by a running job and cannot be deleted.')
      } else {
        setDeleteError('Something went wrong while deleting. Please try again.')
      }
    },
  })

  const rows = items.map((c) => ({
    id: c.id,
    name: c.name,
    tunings: c.associated_jobs?.length ?? 0,
    created_at: c.created_at ?? '',
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
              title="Configurations"
              description="Reusable tuning configurations you can select when starting a new tuning."
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
                    placeholder="Search configurations…"
                    onChange={(_e, value) => setSearchInput(value ?? '')}
                  />
                  {isSpaceAdmin && (
                    <Toggle
                      id="configurations-scope-toggle"
                      labelText=""
                      labelA="Mine"
                      labelB="All"
                      toggled={scope === 'all'}
                      onToggle={(checked) => { setScope(checked ? 'all' : 'own'); setPage(1) }}
                      size="sm"
                    />
                  )}
                  <Button renderIcon={Add} onClick={() => setCreateOpen(true)}>
                    Create New Configuration
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
          title="Some selected configurations can't be deleted"
          subtitle="Configurations in use by a tuning, or the built-in system configuration, cannot be deleted."
          lowContrast
          hideCloseButton
          style={{ marginTop: '0.5rem' }}
        />
      )}

      <SettingsDeleteModal
        open={deleteOpen}
        count={selectedIds.length}
        itemLabel="configuration"
        isDeleting={deleteMutation.isPending}
        errorMessage={deleteError}
        onClose={() => { setDeleteOpen(false); setDeleteError(undefined) }}
        onConfirm={() => deleteMutation.mutate(selectedIds)}
      />

      <SettingsConfigCreate
        open={createOpen}
        onClose={() => setCreateOpen(false)}
        onCreated={() => setCreateOpen(false)}
      />

      <Modal
        open={viewId != null}
        passiveModal
        modalHeading={viewedConfig ? `Configuration: ${viewedConfig.name}` : 'Configuration'}
        size="lg"
        onRequestClose={() => setViewId(null)}
      >
        {isViewLoading || !viewedConfig ? (
          <InlineLoading description="Loading configuration…" />
        ) : (
          <ConfigDisplay configuration={viewedConfig} />
        )}
      </Modal>
    </>
  )
}
