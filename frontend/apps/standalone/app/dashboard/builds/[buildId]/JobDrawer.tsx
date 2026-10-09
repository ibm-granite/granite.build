'use client'

import * as React from 'react'
import { Button, IconButton, InlineLoading, InlineNotification } from '@carbon/react'
import { Close } from '@carbon/icons-react'
import { useInfiniteQuery, useQuery } from '@tanstack/react-query'
import { adaptStatus, getLineageJobDetail, getLineageJobs, type LineageJobDetail, type LineageJobEntry } from '@granite-build/ui-core/api/gbserver'
import type { BuildTargetRun } from '@granite-build/ui-core/types'
import type { IndexElkNode } from '@granite-build/ui-core/components/LineageGraph/indexGraph'
import StepDrawer from './StepDrawer'
import StepDetailsPanel, { ExecutionSummary, Field, Section, hasValue, humanizeKey } from './StepDetailsPanel'
import { stepDrawerSummary, toIsoTimestamp } from './stepDrawerSummary'
import { BuildStatusBadge } from '@granite-build/ui-core/components/BuildStatusBadge'
import styles from './LineagePanel.module.scss'

interface Props {
  node: IndexElkNode
  onClose: () => void
  drawerRef?: React.Ref<HTMLDivElement>
  closeButtonRef?: React.Ref<HTMLButtonElement>
}

const str = (v: unknown) => (typeof v === 'string' && v ? v : undefined)
// Lineage producers write times as ISO strings or as epoch numbers / numeric
// strings (the lakehouse sends epoch ms); toIsoTimestamp unifies them.
const time = toIsoTimestamp
// Other producers' spellings of the build statuses.
const STATUS_ALIASES: Record<string, string> = { successful: 'success', succeeded: 'success', completed: 'success', failure: 'failed', error: 'failed', canceled: 'cancelled' }
const jobStatus = (s: string) => adaptStatus(STATUS_ALIASES[s.toLowerCase()] ?? s)

// The details of a job (run) node that came from the lineage index, shared by the
// build and artifact lineage panels. The job is read from GET /lineage/jobs/{job_id},
// which fetches it from the store that holds it; one granite.build ran is shown with
// the build's own step drawer, anything else with what the index recorded about it.
export default function JobDrawer(props: Props) {
  // A grouped node stands for every job with the same source and target (an
  // in-place rewrite is the case where both are one artifact), not for its
  // representative: list them all rather than open one.
  //
  // But a group of one is not a group: every self-loop is collapsed regardless of how
  // many jobs it holds (graph_builder marks it unconditionally, unlike the repeated
  // A->B grouping which needs two), so a lone in-place rewrite would otherwise render
  // as a one-row list the user has to expand to reach the job. Show the job itself.
  const runCount = props.node.indexNode?.metadata?.run_count
  if (typeof runCount === 'number' && runCount > 1) return <GroupedJobsDrawer {...props} />
  if (typeof runCount === 'number') return <SingleJobDrawer {...props} soleOfGroup />
  return <SingleJobDrawer {...props} />
}

function SingleJobDrawer({ node, onClose, drawerRef, closeButtonRef, soleOfGroup }: Props & { soleOfGroup?: boolean }) {
  const meta = node.indexNode?.metadata ?? {}
  // A collapsed group names its job through representative_job_id; for a group of
  // one that representative IS the job, so it resolves the same single job.
  const jobId = str(meta.job_id) ?? str(meta.representative_job_id) ?? (node.indexNode?.id.startsWith('run:') ? node.indexNode.id.slice('run:'.length) : undefined)
  const { data: job, isLoading, error } = useJobDetail(jobId)

  const title = node.title || node.id
  const buildId = job?.build_id ?? str(meta.gb_build_id) ?? (job ? buildIdOf(job) : undefined)
  // The build's own target run when this server has it; otherwise one rebuilt from
  // what the lineage index captured, so both render through the same drawer.
  const target = job?.target ?? (job ? targetFromJob(job, title) : undefined)
  const fromIndex = !job?.target && !isLoading

  return (
    <StepDrawer
      targetName={job?.target?.target_name ?? title}
      target={target}
      build={job?.target ? job.build ?? undefined : undefined}
      buildId={buildId}
      onClose={onClose}
      drawerRef={drawerRef}
      closeButtonRef={closeButtonRef}
      // Carbon's inline notification: persistent, in context, and not something
      // to act on, so info, low contrast and no close button.
      notice={fromIndex && job && (
        <InlineNotification
          kind="info"
          lowContrast
          hideCloseButton
          title="From the lineage index"
          // A plain-string subtitle, so it takes the notification's own compact type.
          subtitle={[job.detail_error ?? 'Not a granite.build run.', 'Showing what the lineage index recorded'].join('. ') + '.'}
        />
      )}
    >
      {isLoading && <InlineLoading description="Loading job…" />}
      {Boolean(error) && <p className={styles.stepMessage}>Failed to load job: {String(error)}</p>}
      {soleOfGroup && meta.self_loop === true && (
        <Section title="Lineage shape">
          <Field label="Shape">In-place rewrite (reads and writes the same artifact)</Field>
        </Section>
      )}
      {fromIndex && <JobIndexSections job={job} target={target} meta={meta} jobId={jobId} buildId={buildId} />}
      {job && <JobPayloadSections job={job} />}
    </StepDrawer>
  )
}

// GET /lineage/jobs/{job_id}: the job with its content from whichever store holds it.
// Fetched per job and only when it is shown -- a grouped node asks for each row as
// it is opened, never for the whole page.
function useJobDetail(jobId: string | undefined) {
  return useQuery({
    queryKey: ['lineage-job-detail', jobId],
    queryFn: () => getLineageJobDetail(jobId!),
    enabled: Boolean(jobId),
    staleTime: 5 * 60 * 1000,
    retry: false,
  })
}

const PAYLOAD_TITLES: Record<string, string> = {
  job_input_params: 'Input parameters',
  execution_stats: 'Execution stats',
  job_output_stats: 'Output stats',
  source_code_details: 'Source code',
}

// Keys JobIndexSections already shows as their own rows, so the generic "everything
// else" renderers below don't repeat them.
const JOB_KEYS_SHOWN = new Set(['name', 'namespace', 'owner', 'status', 'started_at', 'completed_at'])

/**
 * Every remaining key of a free-form group, as scalar rows plus a nested block.
 *
 * The lineage index is deliberately open: each source writes whatever it recorded
 * (`category` and `type` from the lakehouse, `release_id` under origin.ids, …), so
 * nothing is allowlisted -- anything with a value is shown, scalars as rows and
 * objects/arrays as JSON, the same split StepDetailsPanel's Configuration makes.
 */
function DetailRows({ value, skip }: { value: unknown; skip?: Set<string> }) {
  if (!isPlainObject(value)) return null
  const entries = Object.entries(value).filter(([k]) => !skip?.has(k))
  const scalars = entries.filter(([, v]) => isScalarValue(v) && hasValue(v))
  const nested = entries.filter(([, v]) => !isScalarValue(v) && v != null)
  return (
    <>
      {scalars.map(([k, v]) => <Field key={k} label={humanizeKey(k)}>{String(v)}</Field>)}
      {nested.map(([k, v]) => <JsonField key={k} label={PAYLOAD_TITLES[k] ?? humanizeKey(k)} value={v} />)}
    </>
  )
}

function isScalarValue(v: unknown): boolean {
  return typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean'
}

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
}

function JsonField({ label, value }: { label: string; value: unknown }) {
  return (
    <Field label={label}>
      <pre className={styles.stepCode} style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all', margin: 0 }}>{JSON.stringify(value, null, 2)}</pre>
    </Field>
  )
}

// The large payloads the job's store holds and the index leaves out, plus the
// origin that recorded it -- which system, and its own identifiers for the job.
function JobPayloadSections({ job }: { job: LineageJobDetail }) {
  const entries = Object.entries(job.detail ?? {}).filter(([, v]) => v != null)
  const origin = isPlainObject(job.origin) ? job.origin : {}
  const hasOrigin = Object.keys(origin).length > 0 || Boolean(job.origin_url)
  if (entries.length === 0 && !hasOrigin) return null
  return (
    <>
      {entries.length > 0 && (
        <Section title="Job content">
          {entries.map(([key, value]) => (
            isScalarValue(value)
              ? <Field key={key} label={PAYLOAD_TITLES[key] ?? humanizeKey(key)}>{String(value)}</Field>
              : <JsonField key={key} label={PAYLOAD_TITLES[key] ?? humanizeKey(key)} value={value} />
          ))}
        </Section>
      )}
      {hasOrigin && (
        <Section title="Origin">
          {job.origin_url && <Field label="Source"><a href={job.origin_url} target="_blank" rel="noreferrer">{job.origin_url}</a></Field>}
          <DetailRows value={origin} />
        </Section>
      )}
    </>
  )
}

// The body of an opened row of a grouped node; mounted only when the row is open.
function GroupedJobDetail({ job: entry, target: indexTarget }: { job: LineageJobEntry; target: BuildTargetRun }) {
  const { data: job, isLoading, error } = useJobDetail(entry.job_id)
  const target = job?.target ?? indexTarget
  const buildId = job?.build_id ?? buildIdOf(entry)
  return (
    <div style={{ paddingBottom: '1.5rem' }}>
      {isLoading && <InlineLoading description="Loading job…" />}
      {Boolean(error) && <p className={styles.stepMessage}>Failed to load job: {String(error)}</p>}
      <StepDetailsPanel targetName={target.target_name} target={target} buildId={buildId} />
      <div className={styles.stepExtraSections}>
        {!job?.target && <JobIndexSections job={job ?? entry} target={target} buildId={buildId} />}
        {job && <JobPayloadSections job={job} />}
      </div>
    </div>
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
        {job && <UriField label="Inputs" uris={job.inputs} />}
        {job && <UriField label="Outputs" uris={job.outputs} />}
        {job && <UriField label="Tags" uris={job.tags} />}
      </Section>
      {/* Whatever else the producer recorded about the job -- its type, category and
          any key this UI does not know by name. Open by design: see DetailRows. */}
      {job && (
        <Section title="Job">
          <Field label="Source system">{orNA(job.source_system || str(job.origin?.system))}</Field>
          <Field label="Space">{orNA(job.space_name)}</Field>
          <DetailRows value={job.job} skip={JOB_KEYS_SHOWN} />
        </Section>
      )}
    </>
  )
}

/**
 * A target run rebuilt from a lineage job, for a build this server does not have.
 * The index records one status and span per execution -- no steps, which are
 * left empty rather than invented.
 */
function targetFromJob(job: LineageJobEntry, title: string): BuildTargetRun {
  return {
    uuid: job.job_id,
    target_name: title,
    status: jobStatus(job.status || str(job.job.status) || ''),
    started_at: time(job.started_at) || time(job.job.started_at),
    finished_at: time(job.job.completed_at),
    // The index records no step definitions; those live in the build's own status.
    steps: [],
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
                {open && <GroupedJobDetail job={job} target={target} />}
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
