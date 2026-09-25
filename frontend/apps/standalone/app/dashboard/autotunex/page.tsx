'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'
import { InlineNotification } from '@carbon/react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import axios from 'axios'
import { getJobs, deleteJob } from '@granite-build/ui-core/api/autotunex'
import { deleteEach, isBulkDeleteError } from '@granite-build/ui-core/lib/autotunex/bulkDelete'
import { pruneSelection } from '@granite-build/ui-core/lib/autotunex/tableSelection'
import { adminDefaultScope } from '@granite-build/ui-core/api/client'
import { useAutotunexIsAdmin } from '@granite-build/ui-core/hooks/useAutotunexIsAdmin'
import { AutotunexTabs } from '@granite-build/ui-core/components/autotunex/shared/AutotunexTabs'
import { TuningsTable } from '@granite-build/ui-core/components/autotunex/tunings/TuningsTable'
import { TuningDeleteModal } from '@granite-build/ui-core/components/autotunex/tunings/TuningDeleteModal'
import { TuningCompareModal } from '@granite-build/ui-core/components/autotunex/tunings/TuningCompareModal'

export default function AutoTuneXPage() {
  const router = useRouter()
  const queryClient = useQueryClient()

  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(10)
  const [q, setQ] = useState('')
  // null until the admin flips the toggle; `scope` below resolves the default.
  const [scopeChoice, setScope] = useState<'own' | 'all' | null>(null)
  const [selectedIds, setSelectedIds] = useState<string[]>([])
  const [deleteOpen, setDeleteOpen] = useState(false)
  const [deleteError, setDeleteError] = useState<string | undefined>(undefined)
  const [compareOpen, setCompareOpen] = useState(false)

  const searchDebounceRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  useEffect(() => () => clearTimeout(searchDebounceRef.current), [])

  // Only an AutoTuneX admin may request scope=all, so only they get the toggle.
  const { isAdmin } = useAutotunexIsAdmin()
  const scope = scopeChoice ?? (isAdmin ? adminDefaultScope() : 'own')

  const { data, isLoading, error } = useQuery({
    queryKey: ['autotunex-jobs', page, pageSize, q, scope],
    queryFn: () => getJobs({ page, pageSize, q, scope }),
    placeholderData: (prev) => prev,
  })

  // Memoised so it is a stable dependency for the selection-pruning effect below.
  const items = useMemo(() => data?.items ?? [], [data])
  const total = data?.total ?? 0

  const deleteMutation = useMutation({
    mutationFn: (ids: string[]) => deleteEach(ids, (id) => deleteJob(id, scope)),
    onSuccess: (_data, ids) => {
      setSelectedIds([])
      setDeleteOpen(false)
      setDeleteError(undefined)
      // If the delete emptied the last page, clamp back onto the new last
      // page and let the onSettled invalidation refetch it — no in-memory
      // re-slicing of a locally-shrunk array.
      const newTotal = Math.max(0, total - ids.length)
      const lastPage = Math.max(1, Math.ceil(newTotal / pageSize))
      if (page > lastPage) setPage(lastPage)
    },
    onError: (err) => {
      // A best-effort delete really did remove the ids that succeeded, so narrow
      // the selection to the survivors — otherwise confirming again re-issues a
      // DELETE for rows that are already gone.
      if (isBulkDeleteError(err)) setSelectedIds(err.failedIds)
      const cause = isBulkDeleteError(err) ? err.firstError : err
      if (axios.isAxiosError(cause) && cause.response?.status === 409) {
        setDeleteError('This tuning is still running and cannot be deleted.')
      } else {
        setDeleteError('Something went wrong while deleting. Please try again.')
      }
    },
    // onSettled, not onSuccess: a partially-failed delete still removed rows, and
    // without a refetch the table keeps rendering them.
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['autotunex-jobs'] })
    },
  })

  const handlePageChange = useCallback((p: number, ps: number) => {
    setPage(p)
    setPageSize(ps)
  }, [])

  const handleSearch = useCallback((term: string) => {
    clearTimeout(searchDebounceRef.current)
    searchDebounceRef.current = setTimeout(() => {
      setQ(term)
      setPage(1)
    }, 300)
  }, [])

  const handleScopeChange = useCallback((newScope: 'own' | 'all') => {
    setScope(newScope)
    setPage(1)
  }, [])

  // `selectedIds` shadows the table's own selection. Carbon does not rebuild its
  // checkboxes from the rows it is handed -- it carries `isSelected` forward for
  // every id it still knows -- so this prunes to the visible rows rather than
  // emptying, which is exactly what Carbon's own selection does. Emptying left
  // rows ticked with an empty shadow, and Delete then confirmed a count of 0,
  // removed nothing and closed as a success. Pruning keeps the two in step: a job
  // that leaves the page/search/scope drops out of both, so the delete can still
  // never reach a job the user cannot see, and the compare modal can never receive
  // fewer jobs than the count shown. See pruneSelection.
  useEffect(() => {
    setSelectedIds((prev) => pruneSelection(prev, items.map((j) => j.id)))
  }, [items])

  const selectedJobs = items.filter((j) => selectedIds.includes(j.id))

  return (
    <div style={{ padding: '1.5rem' }}>
      <AutotunexTabs active="tunings" />

      {error && (
        <InlineNotification
          kind="error"
          title="Failed to load tunings"
          subtitle={String(error)}
          style={{ marginBottom: '1rem' }}
        />
      )}

      <TuningsTable
        jobs={items}
        total={total}
        page={page}
        pageSize={pageSize}
        isLoading={isLoading}
        selectedIds={selectedIds}
        onSelectedIdsChange={setSelectedIds}
        onPageChange={handlePageChange}
        onSearch={handleSearch}
        scope={scope}
        onScopeChange={handleScopeChange}
        showScopeToggle={isAdmin}
        onRowClick={(id) => router.push(`/dashboard/autotunex/_/?id=${id}`)}
        onDeleteSelected={() => setDeleteOpen(true)}
        onCompareSelected={() => setCompareOpen(true)}
      />

      <TuningDeleteModal
        open={deleteOpen}
        count={selectedIds.length}
        isDeleting={deleteMutation.isPending}
        errorMessage={deleteError}
        onClose={() => { setDeleteOpen(false); setDeleteError(undefined) }}
        onConfirm={() => deleteMutation.mutate(selectedIds)}
      />

      <TuningCompareModal
        open={compareOpen}
        jobs={selectedJobs}
        onClose={() => setCompareOpen(false)}
      />
    </div>
  )
}
