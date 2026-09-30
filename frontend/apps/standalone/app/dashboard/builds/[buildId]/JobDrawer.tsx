'use client'

import * as React from 'react'
import { IconButton, InlineLoading } from '@carbon/react'
import { Close } from '@carbon/icons-react'
import { useQuery } from '@tanstack/react-query'
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
export default function JobDrawer({ node, onClose, drawerRef, closeButtonRef }: Props) {
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
