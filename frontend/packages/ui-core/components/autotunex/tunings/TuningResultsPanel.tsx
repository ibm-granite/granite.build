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
import { Download, Launch } from '@carbon/icons-react'
import { useQuery } from '@tanstack/react-query'
import axios from 'axios'
import { getJob, getJobAssets, resultArchiveUrl, resultFileUrl } from '../../../api/autotunex'
import { useAutotunexIsAdmin } from '../../../hooks/useAutotunexIsAdmin'
import type { TuningStatus } from '../../../types'
import { formatBytes } from '../../../lib/autotunex/formatBytes'
import { tuningHfArtifactUri } from '../../../lib/autotunex/hfFilesUrl'
import { getHuggingFaceUrl } from '../../LineageGraph/diagramUtilities'

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
  // Same cached admin check the rest of the detail view uses to pick a scope —
  // AutoTuneX admins read `scope=all` so they can see assets for jobs they don't
  // own. No extra fetch: React Query dedupes on the shared key.
  const { isAdmin } = useAutotunexIsAdmin()
  const scope = isAdmin ? 'all' : 'own'

  // Output assets only exist once the job has completed. Both callers mount this
  // panel only for a completed job, so this is the one gate that remains: it keeps
  // a stray mount from hitting the endpoint for a guaranteed 409.
  const enabled = jobStatus === 'completed'

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['autotunex-job-assets', jobId, scope],
    queryFn: () => getJobAssets(jobId, scope),
    enabled,
  })

  // The full job, for its tuning task's artifact_uri -- the repo the assets above
  // are listed from. The build page only holds the build-id lookup's JobDetail,
  // which carries no `tasks`. Same key and scope as the tuning detail page's own
  // job query, so there React Query serves it from cache.
  const { data: fullJob } = useQuery({
    queryKey: ['autotunex-job', jobId, isAdmin],
    queryFn: () => getJob(jobId, scope),
    enabled,
  })
  const hfUri = tuningHfArtifactUri(fullJob?.tasks)
  const hfRepoUrl = hfUri ? getHuggingFaceUrl(hfUri) : null
  // "Files and versions". The URI names no revision; `main` is where gbserver pushes.
  const hfFilesUrl = hfRepoUrl ? `${hfRepoUrl}/tree/main` : null

  if (isLoading) {
    return <InlineLoading description="Loading results…" />
  }

  if (isError) {
    const status = axios.isAxiosError(error) ? error.response?.status : undefined
    // Only a completed job is fetched (`enabled`), and AutoTuneX records the
    // tuning task's output before it marks the job completed, so a 409 here will
    // not clear by waiting -- the job finished without recorded output artifacts.
    if (status === 409) {
      return (
        <InlineNotification
          kind="info"
          title="No results recorded"
          subtitle="This job completed, but no output artifacts were recorded for it."
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
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          {hfFilesUrl && (
            <Button kind="tertiary" size="sm" renderIcon={Launch} href={hfFilesUrl} target="_blank" rel="noopener noreferrer">
              View in HuggingFace
            </Button>
          )}
          <Button kind="tertiary" size="sm" renderIcon={Download} href={resultArchiveUrl(jobId, scope)}>
            Download all
          </Button>
        </div>
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
          {assets.map((asset, index) => {
            // `path` keys the download and is the only thing that makes a row
            // unique -- filenames repeat across trial dirs. It is nullable (see
            // TuningAsset), and two path-less assets sharing a filename used to
            // collide on both the React key and the download URL, so each rendered
            // a link to the same wrong target. Fall back to the index for the key,
            // and drop the link entirely when there is no path to build it from.
            const downloadPath = asset.path
            return (
              <StructuredListRow key={downloadPath ?? `${index}:${asset.filename}`}>
                <StructuredListCell>
                  {downloadPath ? (
                    <CarbonLink href={resultFileUrl(jobId, downloadPath, scope)}>{asset.filename}</CarbonLink>
                  ) : (
                    asset.filename
                  )}
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
