'use client'

import * as React from 'react'
import { Button, IconButton, InlineLoading, InlineNotification } from '@carbon/react'
import { Close } from '@carbon/icons-react'
import { useInfiniteQuery, useQuery } from '@tanstack/react-query'
import { adaptStatus, getBuild, getBuildStatus, getLineageJobs, type LineageJobEntry } from '@granite-build/ui-core/api/gbserver'
import type { BuildTargetRun } from '@granite-build/ui-core/types'
import type { IndexElkNode } from '@granite-build/ui-core/components/LineageGraph/indexGraph'
import StepDrawer from './StepDrawer'
import StepDetailsPanel, { ExecutionSummary, Field, Section } from './StepDetailsPanel'
import { stepDrawerSummary } from './stepDrawerSummary'
import { BuildStatusBadge } from '@granite-build/ui-core/components/BuildStatusBadge'
import styles from './LineagePanel.module.scss'

interface Props {
  node: IndexElkNode
  onClose: () => void
  drawerRef?: React.Ref<HTMLDivElement>
  closeButtonRef?: React.Ref<HTMLButtonElement>
}

const str = (v: unknown) => (typeof v === 'string' && v ? v : undefined)
// Lineage producers write times either as ISO strings or as epoch milliseconds
// (the lakehouse does the latter); both come out as ISO.
const time = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) ? new Date(v).toISOString() : str(v))
// Other producers' spellings of the build statuses.
const STATUS_ALIASES: Record<string, string> = { successful: 'success', succeeded: 'success', completed: 'success', failure: 'failed', error: 'failed', canceled: 'cancelled' }
const jobStatus = (s: string) => adaptStatus(STATUS_ALIASES[s.toLowerCase()] ?? s)

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
    ?? (job ? buildIdOf(job) : undefined)
  const runId = str(meta.gb_target_run_uuid) ?? jobId
  const { data: build } = useQuery({
    queryKey: ['build', buildId],
    queryFn: () => getBuild(buildId!),
    enabled: Boolean(buildId),
    staleTime: 5 * 60 * 1000,
  })
  const { data: buildStatus, isLoading: statusLoading } = useQuery({
    queryKey: ['build-status', buildId],
    queryFn: () => getBuildStatus(buildId!),
    enabled: Boolean(buildId),
    staleTime: 5 * 60 * 1000,
    retry: false,
  })
  const statusTarget = React.useMemo(
    () => Object.values(buildStatus?.targets ?? {}).find((t) => t.uuid === runId),
    [buildStatus, runId]
  )

  const title = node.title || node.id
  // The build's own target run when this server has it; otherwise one rebuilt from
  // what the lineage index captured, so both render through the same drawer.
  const target = statusTarget ?? (job ? targetFromJob(job, title) : undefined)
  const fromIndex = !statusTarget && !(buildId && statusLoading)

  return (
    <StepDrawer
      targetName={statusTarget?.target_name ?? title}
      target={target}
      build={statusTarget ? build : undefined}
      buildId={buildId}
      onClose={onClose}
      drawerRef={drawerRef}
      closeButtonRef={closeButtonRef}
      // Carbon's inline notification: persistent, in context, and not something
      // to act on, so info, low contrast and no close button.
      notice={fromIndex && (
        <InlineNotification
          kind="info"
          lowContrast
          hideCloseButton
          title="From the lineage index"
          // A plain-string subtitle, so it takes the notification's own compact
          // type; the full build id is in the Build ID field below.
          subtitle={[
            buildId
              ? `Build ${buildId.slice(0, 8)} is not on this server, so its target and step runs are unknown.`
              : 'Not a granite.build run.',
            'Showing what the lineage index recorded',
          ].join(' ') + (job && !job.job_recorded ? '; it has no job record, only lineage rows.' : '.')}
        />
      )}
    >
      {fromIndex && (
        <>
          {jobLoading && <InlineLoading description="Loading job…" />}
          {Boolean(jobError) && <p className={styles.stepMessage}>Failed to load job: {String(jobError)}</p>}
          <JobIndexSections job={job} target={target} meta={meta} jobId={jobId} buildId={buildId} />
        </>
      )}
    </StepDrawer>
  )
}

const NA = 'N/A'

/**
 * What the lineage index knows about one job beyond its steps: its span, through
 * the step cards' Execution block (the index records one start and finish per
 * execution, so it shows even with no steps), and its identity and endpoints.
 * Shared by the single-job fallback and each row of a grouped node.
 */
function JobIndexSections({ job, target, meta = {}, jobId, buildId }: {
  job: LineageJobEntry | undefined
  target: BuildTargetRun | undefined
  meta?: Record<string, unknown>
  jobId?: string
  buildId?: string
}) {
  return (
    <>
      {target && (
        <Section title="Execution">
          <ExecutionSummary step={{ step_name: target.target_name, status: target.status, started_at: target.started_at, finished_at: target.finished_at }} />
        </Section>
      )}
      <Section title="Lineage">
        <CodeField label="Build ID" value={buildId} />
        <CodeField label="Target run" value={str(meta.gb_target_run_uuid)} />
        <CodeField label="Job ID" value={job?.job_id ?? jobId} />
        <Field label="Namespace">{orNA(job?.job_namespace || str(job?.job.namespace) || str(meta.job_namespace))}</Field>
        <Field label="Owner">{orNA(job?.owner || str(job?.job.owner) || str(meta.owner))}</Field>
        <Field label="Source system">{orNA(job?.source_system || str(meta.source_system))}</Field>
        {job && <UriField label="Inputs" uris={job.inputs} />}
        {job && <UriField label="Outputs" uris={job.outputs} />}
        {job && <UriField label="Tags" uris={job.tags} />}
      </Section>
    </>
  )
}

/**
 * A target run rebuilt from a lineage job, for a build this server does not have.
 * The index records one status and span per execution and, for granite.build,
 * each step's definition URI and redacted config -- not a step's own status or
 * timing, which are left unknown rather than borrowed from the job.
 */
function targetFromJob(job: LineageJobEntry, title: string): BuildTargetRun {
  return {
    uuid: job.job_id,
    target_name: title,
    status: jobStatus(job.status || str(job.job.status) || ''),
    started_at: job.started_at || time(job.job.started_at),
    finished_at: time(job.job.completed_at),
    steps: stepsOf(job.job_input_params).map((step) => ({
      step_name: str(step.uri)?.split('/').pop() || NA,
      status: adaptStatus('unknown'),
      uri: str(step.uri),
      config: isNonEmptyObject(step.config) ? step.config : undefined,
    })),
  }
}

function orNA(value: string | undefined): React.ReactNode {
  return value ? value : <span className={styles.stepMuted}>{NA}</span>
}

function CodeField({ label, value }: { label: string; value: string | undefined }) {
  return (
    <Field label={label}>
      {value ? <code className={styles.stepCode}>{value}</code> : <span className={styles.stepMuted}>{NA}</span>}
    </Field>
  )
}

function UriField({ label, uris }: { label: string; uris: string[] }) {
  if (uris.length === 0) return null
  return (
    <Field label={label}>
      <ul className={styles.stepUriList}>
        {uris.map((u) => <li key={u}><code className={styles.stepCode}>{u}</code></li>)}
      </ul>
    </Field>
  )
}

// granite.build records job_input_params as {steps: [{uri, config, config_dir}]};
// another producer's shape yields no steps.
function stepsOf(params: Record<string, unknown> | undefined): Record<string, unknown>[] {
  const steps = params?.steps
  return Array.isArray(steps) ? steps.filter((s): s is Record<string, unknown> => Boolean(s) && typeof s === 'object') : []
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
            const target = targetFromJob(job, str(job.job.name) ?? job.job_id)
            const { subtitle, summary } = stepDrawerSummary(target)
            return (
              <li key={job.job_id} style={{ borderBottom: '1px solid var(--cds-border-subtle)' }}>
                {/* The row reads like the step drawer header: name, steps, status
                    badge and span, from the same stepDrawerSummary. */}
                <button
                  type="button"
                  aria-expanded={open}
                  onClick={() => setOpenJobId(open ? null : job.job_id)}
                  style={{
                    all: 'unset', cursor: 'pointer', display: 'grid', width: '100%',
                    gridTemplateColumns: '1fr auto', gap: '0.25rem 1rem', padding: '0.75rem 0',
                    boxSizing: 'border-box',
                  }}
                >
                  <span className={styles.stepCardName} style={{ wordBreak: 'break-all' }}>{target.target_name}</span>
                  {target.status ? <BuildStatusBadge status={target.status} /> : <span className={styles.stepMuted}>{NA}</span>}
                  <span className={styles.stepMuted} style={{ gridColumn: '1 / -1', fontSize: '0.75rem' }}>
                    {[target.steps.length > 0 ? subtitle : undefined, summary].filter(Boolean).join(' · ') || NA}
                  </span>
                </button>
                {open && (
                  <div style={{ paddingBottom: '1.5rem' }}>
                    <StepDetailsPanel targetName={target.target_name} target={target} buildId={buildIdOf(job)} />
                    <div className={styles.stepExtraSections}>
                      <JobIndexSections job={job} target={target} buildId={buildIdOf(job)} />
                    </div>
                  </div>
                )}
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

function buildIdOf(job: LineageJobEntry): string | undefined {
  return job.tags.find((t) => t.startsWith('build_id='))?.slice('build_id='.length)
}

function isNonEmptyObject(v: unknown): v is Record<string, unknown> {
  return Boolean(v) && typeof v === 'object' && Object.keys(v as object).length > 0
}
