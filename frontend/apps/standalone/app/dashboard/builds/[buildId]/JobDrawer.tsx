'use client'

import * as React from 'react'
import { Button, IconButton, InlineLoading } from '@carbon/react'
import { Close } from '@carbon/icons-react'
import { useInfiniteQuery, useQuery } from '@tanstack/react-query'
import { getBuild, getBuildStatus, getLineageJobs, type LineageJobEntry } from '@granite-build/ui-core/api/gbserver'
import type { IndexElkNode } from '@granite-build/ui-core/components/LineageGraph/indexGraph'
import StepDrawer from './StepDrawer'
import styles from './LineagePanel.module.scss'

interface Props {
  node: IndexElkNode
  onClose: () => void
  drawerRef?: React.Ref<HTMLDivElement>
  closeButtonRef?: React.Ref<HTMLButtonElement>
}

const str = (v: unknown) => (typeof v === 'string' && v ? v : undefined)

// The details of a job (run) node that came from the lineage index, shared by the
// build and artifact lineage panels. The job is read from GET /lineage/jobs; one
// granite.build ran is shown with the build's own step drawer, anything else with
// what the index recorded about it.
export default function JobDrawer(props: Props) {
  // A grouped node stands for every job with the same source and target (an
  // in-place rewrite is the case where both are one artifact), not for its
  // representative: list them all rather than open one.
  if (typeof props.node.indexNode?.metadata?.run_count === 'number') return <GroupedJobsDrawer {...props} />
  return <SingleJobDrawer {...props} />
}

function SingleJobDrawer({ node, onClose, drawerRef, closeButtonRef }: Props) {
  const meta = node.indexNode?.metadata ?? {}
  const jobId = str(meta.job_id) ?? (node.indexNode?.id.startsWith('run:') ? node.indexNode.id.slice('run:'.length) : undefined)

  const { data: jobs, isLoading: jobLoading, error: jobError } = useQuery({
    queryKey: ['lineage-job', jobId],
    queryFn: () => getLineageJobs({ job_id: jobId!, limit: 1 }),
    enabled: Boolean(jobId),
    staleTime: 5 * 60 * 1000,
    retry: false,
  })
  const job = jobs?.jobs[0]

  const buildId = str(meta.gb_build_id)
    ?? job?.tags.find((t) => t.startsWith('build_id='))?.slice('build_id='.length)
  const runId = str(meta.gb_target_run_uuid) ?? jobId
  const { data: build } = useQuery({
    queryKey: ['build', buildId],
    queryFn: () => getBuild(buildId!),
    enabled: Boolean(buildId),
    staleTime: 5 * 60 * 1000,
  })
  const { data: buildStatus } = useQuery({
    queryKey: ['build-status', buildId],
    queryFn: () => getBuildStatus(buildId!),
    enabled: Boolean(buildId),
    staleTime: 5 * 60 * 1000,
    retry: false,
  })
  const target = React.useMemo(
    () => Object.values(buildStatus?.targets ?? {}).find((t) => t.uuid === runId),
    [buildStatus, runId]
  )

  if (buildId) {
    return (
      <StepDrawer
        targetName={target?.target_name ?? node.title ?? node.id}
        target={target}
        build={build}
        onClose={onClose}
        drawerRef={drawerRef}
        closeButtonRef={closeButtonRef}
      />
    )
  }

  const title = node.title || node.id
  const status = job?.status || str(meta.job_status)
  return (
    <div ref={drawerRef} className={styles.stepSidePanel} role="dialog" aria-label={`Job details — ${title}`}>
      <div className={styles.stepSidePanelHeader}>
        <div className={styles.stepSidePanelIdentity}>
          <h4 className={styles.stepSidePanelHeading}>{title}</h4>
          <div className={styles.stepSidePanelSubtitle}>Job{status ? ` · ${status}` : ''}</div>
        </div>
        <IconButton ref={closeButtonRef} kind="ghost" label="Close" align="bottom" onClick={onClose}>
          <Close />
        </IconButton>
      </div>
      <div className={styles.stepSidePanelBody}>
        {jobLoading && <InlineLoading description="Loading job…" />}
        {Boolean(jobError) && <p>Failed to load job: {String(jobError)}</p>}
        {/* The index has no record of it: fall back to what the graph node carried. */}
        <DetailList rows={job ? jobRows(job) : Object.entries(meta)} />
      </div>
    </div>
  )
}

const GROUPED_PAGE = 50

// Every job with the node's source and target, paged from
// GET /lineage/jobs with the node's jobs_query. One row per job; a row opens its details.
function GroupedJobsDrawer({ node, onClose, drawerRef, closeButtonRef }: Props) {
  const meta = node.indexNode?.metadata ?? {}
  // jobs_query is the exact filter the server says lists this node's jobs; a group
  // with an empty side has no source (or target) to anchor on, so it uses `terminal`.
  const query = (meta.jobs_query ?? {}) as { uri?: string; output?: string; terminal?: 'input' | 'output' }
  const uri = query.uri ?? str(meta.source_uri)
  const terminal = query.terminal
  const output = terminal ? undefined : (query.output ?? str(meta.target_uri) ?? uri)
  const runCount = typeof meta.run_count === 'number' ? meta.run_count : undefined
  const [openJobId, setOpenJobId] = React.useState<string | null>(null)

  const { data, isLoading, isFetchingNextPage, hasNextPage, fetchNextPage, error } = useInfiniteQuery({
    queryKey: ['lineage-grouped-jobs', uri, output, terminal],
    queryFn: ({ pageParam }) => getLineageJobs({ uri: uri!, output, terminal, limit: GROUPED_PAGE, offset: pageParam }),
    initialPageParam: 0,
    getNextPageParam: (last) => (last.offset + last.jobs.length < last.total && last.jobs.length > 0 ? last.offset + last.jobs.length : undefined),
    enabled: Boolean(uri),
    staleTime: 5 * 60 * 1000,
    retry: false,
  })
  const jobs = React.useMemo(() => data?.pages.flatMap((p) => p.jobs) ?? [], [data])
  const total = data?.pages[0]?.total ?? runCount

  const title = node.title || node.id
  return (
    <div ref={drawerRef} className={styles.stepSidePanel} role="dialog" aria-label={`Jobs — ${title}`}>
      <div className={styles.stepSidePanelHeader}>
        <div className={styles.stepSidePanelIdentity}>
          <h4 className={styles.stepSidePanelHeading}>{title}</h4>
          <div className={styles.stepSidePanelSubtitle}>
            {total !== undefined ? `${total} ${total === 1 ? 'run' : 'runs'}` : 'Runs'} · {terminal === 'input' ? 'no recorded input' : terminal === 'output' ? 'no recorded output' : uri === output ? 'in-place rewrites' : 'same source and target'}
          </div>
        </div>
        <IconButton ref={closeButtonRef} kind="ghost" label="Close" align="bottom" onClick={onClose}>
          <Close />
        </IconButton>
      </div>
      <div className={styles.stepSidePanelBody}>
        {isLoading && <InlineLoading description="Loading jobs…" />}
        {Boolean(error) && <p>Failed to load jobs: {String(error)}</p>}
        <ul style={{ listStyle: 'none', margin: 0, padding: 0 }}>
          {jobs.map((job) => {
            const open = openJobId === job.job_id
            const name = str(job.job.name) ?? job.job_id
            return (
              <li key={job.job_id} style={{ borderBottom: '1px solid var(--cds-border-subtle)' }}>
                <button
                  type="button"
                  aria-expanded={open}
                  onClick={() => setOpenJobId(open ? null : job.job_id)}
                  style={{
                    all: 'unset', cursor: 'pointer', display: 'grid', width: '100%',
                    gridTemplateColumns: '1fr auto', gap: '0.25rem 1rem', padding: '0.5rem 0',
                    fontSize: '0.875rem', boxSizing: 'border-box',
                  }}
                >
                  <span style={{ wordBreak: 'break-all' }}>{name}</span>
                  <span style={{ color: 'var(--cds-text-secondary)' }}>{job.status}</span>
                  <span style={{ color: 'var(--cds-text-secondary)', gridColumn: '1 / -1' }}>
                    {[job.owner, str(job.job.started_at) ?? job.started_at].filter(Boolean).join(' · ')}
                  </span>
                </button>
                {open && <div style={{ paddingBottom: '0.75rem' }}><DetailList rows={jobRows(job)} /></div>}
              </li>
            )
          })}
        </ul>
        {hasNextPage && (
          <Button kind="ghost" size="sm" disabled={isFetchingNextPage} onClick={() => fetchNextPage()}>
            {isFetchingNextPage ? 'Loading…' : `Show more (${jobs.length} of ${total})`}
          </Button>
        )}
      </div>
    </div>
  )
}

function jobRows(job: LineageJobEntry): [string, unknown][] {
  return [
    ['job_id', job.job_id],
    ['namespace', job.job_namespace],
    ['space', job.space_name],
    ['owner', job.owner],
    ['source', job.source_system],
    ['status', job.status],
    ['started_at', job.started_at],
    ['tags', job.tags.join(', ')],
    ['inputs', job.inputs.join('\n')],
    ['outputs', job.outputs.join('\n')],
    ...Object.entries(job.job),
  ]
}

function DetailList({ rows }: { rows: [string, unknown][] }) {
  const shown = rows.filter(([, v]) => v !== null && v !== undefined && v !== '')
  return (
    <dl style={{ display: 'grid', gridTemplateColumns: 'max-content 1fr', gap: '0.5rem 1rem', fontSize: '0.875rem' }}>
      {shown.map(([k, v], i) => (
        <React.Fragment key={`${k}-${i}`}>
          <dt style={{ color: 'var(--cds-text-secondary)' }}>{k}</dt>
          <dd style={{ wordBreak: 'break-all', whiteSpace: 'pre-line' }}>{typeof v === 'string' ? v : JSON.stringify(v)}</dd>
        </React.Fragment>
      ))}
    </dl>
  )
}
