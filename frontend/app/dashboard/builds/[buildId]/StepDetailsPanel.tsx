'use client'

import * as React from 'react'
import { Link } from '@carbon/react'
import styles from './LineagePanel.module.scss'
import type { BuildStepRun, BuildTargetRun } from '@/types'
import { BuildStatusBadge } from '@/components/BuildStatusBadge'

const NOT_RECORDED = 'Not recorded'

function formatTime(value: string | undefined): string {
  if (!value) return '—'
  const parsed = new Date(value)
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString()
}

/**
 * gbserver writes step status messages as markdown (fenced blocks), which
 * renders as literal backticks in plain text. Strip the fences and keep the
 * lines so the message reads as a message.
 */
function cleanStatusMessage(msg: string): string {
  return msg
    .split('\n')
    .filter((line) => !/^\s*```/.test(line))
    .join('\n')
    .replace(/```/g, '')
    .trim()
}

/** Row of a definition list; renders `—` for absent values rather than collapsing. */
function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className={styles.stepField}>
      <span className={styles.stepFieldLabel}>{label}</span>
      <span className={styles.stepFieldValue}>{children}</span>
    </div>
  )
}

function StepCard({ step, index }: { step: BuildStepRun; index: number }) {
  const params = step.config ?? {}
  const paramKeys = Object.keys(params)

  return (
    <li className={styles.stepCard}>
      <div className={styles.stepCardHeader}>
        <span className={styles.stepCardName}>
          {index + 1}. {step.step_name}
        </span>
        <BuildStatusBadge status={step.status} />
      </div>

      <Field label="Definition URI">
        {step.uri ? <code className={styles.stepCode}>{step.uri}</code> : '—'}
      </Field>
      <Field label="Container image">
        {step.image ? (
          <code className={styles.stepCode}>{step.image}</code>
        ) : (
          <span className={styles.stepMuted}>
            {NOT_RECORDED} — resolved by the compute environment at launch
          </span>
        )}
      </Field>
      <Field label="Launcher">{step.launcher ?? '—'}</Field>
      <Field label="Started">{formatTime(step.started_at)}</Field>
      <Field label="Finished">{formatTime(step.finished_at ?? step.updated_at)}</Field>
      {step.status_msg && (
        <Field label="Message">
          <span className={styles.stepMessage}>{cleanStatusMessage(step.status_msg)}</span>
        </Field>
      )}

      <Field label="Runtime parameters">
        {paramKeys.length === 0 ? (
          <span className={styles.stepMuted}>None</span>
        ) : null}
      </Field>
      {paramKeys.length > 0 && (
        <pre className={styles.stepParams}>{JSON.stringify(params, null, 2)}</pre>
      )}
    </li>
  )
}

interface Props {
  targetName: string
  target: BuildTargetRun | undefined
  /** The build's source URI — stands in for a commit SHA, which is not recorded. */
  sourceUri?: string
}

export default function StepDetailsPanel({ targetName, target, sourceUri }: Props) {
  const steps = target?.steps ?? []

  return (
    <div className={styles.stepPanel}>
      <div className={styles.stepPanelSource}>
        <Field label="Source">
          {sourceUri ? (
            <Link href={sourceUri} target="_blank" rel="noopener noreferrer">
              {sourceUri}
            </Link>
          ) : (
            <span className={styles.stepMuted}>{NOT_RECORDED}</span>
          )}
        </Field>
        <Field label="Code commit">
          <span className={styles.stepMuted}>
            {NOT_RECORDED} — gbserver does not persist a commit SHA for build sources
          </span>
        </Field>
      </div>

      {steps.length === 0 ? (
        <p className={styles.stepEmpty}>
          No step runs recorded for target <strong>{targetName}</strong>. Steps appear
          here once the target starts running.
        </p>
      ) : (
        <ol className={styles.stepList}>
          {steps.map((step, i) => (
            <StepCard key={step.uuid ?? `${step.step_name}-${i}`} step={step} index={i} />
          ))}
        </ol>
      )}
    </div>
  )
}
