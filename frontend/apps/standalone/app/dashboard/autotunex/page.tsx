'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'
import { InlineNotification } from '@carbon/react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { getJobs, deleteJob } from '@granite-build/ui-core/api/autotunex'
import { listSpaces } from '@granite-build/ui-core/api/gbserver'
import { AutotunexTabs } from '@granite-build/ui-core/components/AutotunexTabs'
import { TuningsTable } from '@granite-build/ui-core/components/TuningsTable'
import { TuningDeleteModal } from '@granite-build/ui-core/components/TuningDeleteModal'
import { TuningCompareModal } from '@granite-build/ui-core/components/TuningCompareModal'

export default function AutoTuneXPage() {
  const router = useRouter()
  const queryClient = useQueryClient()

  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(10)
  const [q, setQ] = useState('')
  const [scope, setScope] = useState<'own' | 'all'>('own')
  const [selectedIds, setSelectedIds] = useState<string[]>([])
  const [deleteOpen, setDeleteOpen] = useState(false)
  const [compareOpen, setCompareOpen] = useState(false)

  const searchDebounceRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  useEffect(() => () => clearTimeout(searchDebounceRef.current), [])

  // Reused verbatim from the builds/artifacts pages (`["spaces"]` queryKey) so
  // this shares the same React Query cache entry rather than issuing a
  // duplicate `listSpaces()` fetch. There's no "current active space" concept
  // in this dashboard (no space context/provider — grepped for one), so the
  // scope toggle is gated on "is admin of at least one space" rather than a
  // single active space's `is_admin`.
  const { data: spaces = [] } = useQuery({
    queryKey: ['spaces'],
    queryFn: listSpaces,
  })
  const isSpaceAdmin = spaces.some((s) => s.is_admin)

  const { data, isLoading, error } = useQuery({
    queryKey: ['autotunex-jobs', page, pageSize, q, scope],
    queryFn: () => getJobs({ page, pageSize, q, scope }),
    placeholderData: (prev) => prev,
  })

  const items = data?.items ?? []
  const total = data?.total ?? 0

  const deleteMutation = useMutation({
    mutationFn: async (ids: string[]) => {
      for (const id of ids) await deleteJob(id, scope)
    },
    onSuccess: (_data, ids) => {
      queryClient.invalidateQueries({ queryKey: ['autotunex-jobs'] })
      setSelectedIds([])
      setDeleteOpen(false)
      // If the delete emptied the last page, clamp back onto the new last
      // page and let the invalidated query above refetch it — no in-memory
      // re-slicing of a locally-shrunk array.
      const newTotal = Math.max(0, total - ids.length)
      const lastPage = Math.max(1, Math.ceil(newTotal / pageSize))
      if (page > lastPage) setPage(lastPage)
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
        showScopeToggle={isSpaceAdmin}
        onRowClick={(id) => router.push(`/dashboard/autotunex/_/?id=${id}`)}
        onDeleteSelected={() => setDeleteOpen(true)}
        onCompareSelected={() => setCompareOpen(true)}
      />

      <TuningDeleteModal
        open={deleteOpen}
        count={selectedIds.length}
        isDeleting={deleteMutation.isPending}
        onClose={() => setDeleteOpen(false)}
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
