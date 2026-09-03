'use client'

import {
  StructuredListWrapper,
  StructuredListHead,
  StructuredListRow,
  StructuredListCell,
  StructuredListBody,
  InlineNotification,
  InlineLoading,
  Button,
  Link as CarbonLink,
} from '@carbon/react'
import { Download } from '@carbon/icons-react'
import { useQuery } from '@tanstack/react-query'
import axios from 'axios'
import { getJobAssets, resultArchiveUrl, resultFileUrl } from '../api/autotunex'
import { listSpaces } from '../api/gbserver'
import type { TuningStatus } from '../types'

function formatBytes(bytes: number): string {
  if (!bytes) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let i = 0
  let v = bytes
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v.toFixed(1)} ${units[i]}`
}

function formatModified(modified: string | null): string {
  if (!modified) return '—'
  const d = new Date(modified)
  return Number.isNaN(d.getTime()) ? '—' : d.toLocaleString()
}

interface Props {
  jobId: string
  jobStatus: TuningStatus
}

export function TuningResultsPanel({ jobId, jobStatus }: Props) {
  // Same cached `['spaces']` query the rest of the detail view uses to pick a
  // scope — admins read `scope=all` so they can see assets for jobs they don't
  // own. No extra fetch: React Query dedupes on the shared key.
  const { data: spaces = [] } = useQuery({ queryKey: ['spaces'], queryFn: listSpaces })
  const scope = spaces.some((s) => s.is_admin) ? 'all' : 'own'

  // Output assets only exist once the job has completed. Gating the fetch here
  // also avoids hammering the endpoint with guaranteed 409s while a job runs —
  // Carbon mounts every tab panel, so this renders even when Results isn't the
  // active tab.
  const enabled = jobStatus === 'completed'

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['autotunex-job-assets', jobId, scope],
    queryFn: () => getJobAssets(jobId, scope),
    enabled,
  })

  if (!enabled) {
    return (
      <InlineNotification
        kind="info"
        title="No results yet"
        subtitle="Output assets become available once the tuning completes."
        hideCloseButton
      />
    )
  }

  if (isLoading) {
    return <InlineLoading description="Loading results…" />
  }

  if (isError) {
    const status = axios.isAxiosError(error) ? error.response?.status : undefined
    if (status === 409) {
      return (
        <InlineNotification
          kind="info"
          title="Results aren't ready yet"
          subtitle="The job's output artifacts are still being finalized. Try again shortly."
          hideCloseButton
        />
      )
    }
    return (
      <InlineNotification
        kind="error"
        title="Failed to load results"
        subtitle={String(error)}
        hideCloseButton
      />
    )
  }

  const assets = data ?? []

  if (assets.length === 0) {
    return <InlineNotification kind="info" title="No results data available" hideCloseButton />
  }

  return (
    <div>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: '0.5rem',
        }}
      >
        <h5 style={{ margin: 0 }}>Output assets</h5>
        <Button kind="tertiary" size="sm" renderIcon={Download} href={resultArchiveUrl(jobId, scope)}>
          Download all
        </Button>
      </div>
      <StructuredListWrapper>
        <StructuredListHead>
          <StructuredListRow head>
            <StructuredListCell head>File name</StructuredListCell>
            <StructuredListCell head>File size</StructuredListCell>
            <StructuredListCell head>Modified</StructuredListCell>
          </StructuredListRow>
        </StructuredListHead>
        <StructuredListBody>
          {assets.map((asset) => {
            // `path` keys the download; filenames repeat across trial dirs.
            const key = asset.path ?? asset.filename
            return (
              <StructuredListRow key={key}>
                <StructuredListCell>
                  <CarbonLink href={resultFileUrl(jobId, key, scope)}>{asset.filename}</CarbonLink>
                </StructuredListCell>
                <StructuredListCell>{formatBytes(asset.size)}</StructuredListCell>
                <StructuredListCell>{formatModified(asset.modified)}</StructuredListCell>
              </StructuredListRow>
            )
          })}
        </StructuredListBody>
      </StructuredListWrapper>
    </div>
  )
}
