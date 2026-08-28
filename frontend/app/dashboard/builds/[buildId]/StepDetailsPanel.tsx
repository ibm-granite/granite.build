'use client'

import * as React from 'react'
import { Link } from '@carbon/react'
import styles from './LineagePanel.module.scss'
import type { BuildStepRun, BuildTargetRun } from '@/types'
import { BuildStatusBadge } from '@/components/BuildStatusBadge'

const NOT_RECORDED = 'Not recorded'

/** `10:51:22` — the clock time alone, for the compact Execution row. */
function formatClock(value: string | undefined): string {
  if (!value) return '—'
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  return parsed.toLocaleTimeString([], { hour12: false })
}

/** `Aug 22, 2026 at 10:51:28` — the one place a full timestamp is spelled out. */
function formatDateTime(value: string | undefined): string | undefined {
  if (!value) return undefined
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  const date = parsed.toLocaleDateString([], {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })
  return `${date} at ${parsed.toLocaleTimeString([], { hour12: false })}`
}

/**
 * `6s`, `2m 14s`, `1h 03m`. Derived from the start/finish pair rather than
 * read off the step — gbserver records no duration field.
 */
function formatDuration(from: string | undefined, to: string | undefined): string | undefined {
  if (!from || !to) return undefined
  const start = new Date(from).getTime()
  const end = new Date(to).getTime()
  if (Number.isNaN(start) || Number.isNaN(end)) return undefined
  const totalSeconds = Math.round((end - start) / 1000)
  // Clock skew between the launcher and gbserver can put finish before start;
  // showing a negative duration would be worse than showing none.
  if (totalSeconds < 0) return undefined
  if (totalSeconds < 60) return `${totalSeconds}s`
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  if (minutes < 60) return `${minutes}m ${String(seconds).padStart(2, '0')}s`
  const hours = Math.floor(minutes / 60)
  return `${hours}h ${String(minutes % 60).padStart(2, '0')}m`
}

/** The step's own finish time, falling back to the mirrored `updated_at`. */
function finishedAt(step: BuildStepRun): string | undefined {
  return step.finished_at ?? step.updated_at
}

/**
 * Keys in the status message that the drawer already shows elsewhere — in the
 * header (Status) or the Metadata block (the ids and the step URI). gbserver
 * emits them as a padded `Key      : Value` table (see the templates in
 * gbserver/build/run.py), so the message is almost entirely a restatement.
 */
const REDUNDANT_MESSAGE_KEYS = new Set([
  'status',
  'target name',
  'type',
  'step uri',
  'step id',
  'target id',
  'build id',
])

/**
 * gbserver writes step status messages as markdown (fenced blocks), which
 * renders as literal backticks in plain text. Strip the fences, then drop the
 * `Key : Value` lines already displayed in Metadata/header so only the trailing
 * free-text remainder (`extra_msg`) survives — that is the one part of the
 * message not shown anywhere else.
 *
 * Returns '' when nothing new is left, so the caller can omit the section.
 */
function cleanStatusMessage(msg: string): string {
  return msg
    .split('\n')
    .filter((line) => !/^\s*```/.test(line))
    .filter((line) => {
      // Keep prose and blank lines; only drop recognised key/value restatements.
      const match = /^\s*([A-Za-z][A-Za-z ]*?)\s*:\s*(.*)$/.exec(line)
      if (!match) return true
      return !REDUNDANT_MESSAGE_KEYS.has(match[1].trim().toLowerCase())
    })
    .join('\n')
    .replace(/```/g, '')
    // Collapse the blank lines the removed rows leave behind.
    .replace(/\n{3,}/g, '\n\n')
    .replace(/^#+\s.*$/gm, '')
    .trim()
}

/**
 * The shell script a step runs is the single most useful thing in its config,
 * but it sits several levels down inside an arbitrary JSON blob. The builtin
 * `command` step nests it under `command_config.command`; the docker/runpod/
 * skypilot launchers read it from their own `<launcher>.command` (see
 * gbserver/environment/{docker,runpod}.py). Probe those in order and fall back
 * to a bare top-level `command`.
 *
 * Returns the command *and* the config key it came from, so the JSON section
 * below can omit the key it already rendered as a code block.
 */
const COMMAND_CONTAINER_KEYS = [
  'command_config',
  'docker',
  'runpod',
  'skypilot',
  'bash',
  'launcher_config',
]

function extractCommand(
  config: Record<string, unknown> | undefined
): { command: string; sourceKey: string } | undefined {
  if (!config) return undefined

  for (const key of COMMAND_CONTAINER_KEYS) {
    const container = config[key] as Record<string, unknown> | undefined
    const candidate = container?.command
    if (typeof candidate === 'string' && candidate.trim()) {
      return { command: candidate.trim(), sourceKey: key }
    }
    // Some launchers pass an argv array (`["sh", "-c", "…"]`) rather than a
    // string; join it so it still reads as a command.
    if (Array.isArray(candidate) && candidate.length > 0) {
      return { command: candidate.map(String).join(' '), sourceKey: key }
    }
  }

  const top = config.command
  if (typeof top === 'string' && top.trim()) {
    return { command: top.trim(), sourceKey: 'command' }
  }
  return undefined
}

/**
 * The remaining config, minus the container the command was pulled out of when
 * that container held nothing else. Keeping a `{"command_config": {}}` husk in
 * the JSON block would just be noise.
 */
function remainingConfig(
  config: Record<string, unknown> | undefined,
  sourceKey: string | undefined
): Record<string, unknown> {
  if (!config) return {}
  if (!sourceKey) return config

  if (sourceKey === 'command') {
    const { command: _omitted, ...rest } = config
    return rest
  }

  const container = config[sourceKey] as Record<string, unknown> | undefined
  const { command: _omitted, ...containerRest } = container ?? {}
  if (Object.keys(containerRest).length === 0) {
    const { [sourceKey]: _dropped, ...rest } = config
    return rest
  }
  return { ...config, [sourceKey]: containerRest }
}

/**
 * Human labels for the config blocks worth surfacing as their own subgroup.
 * Anything not listed here stays in the Advanced/raw disclosure rather than
 * inventing a heading for a key we don't recognise.
 */
const CONFIG_GROUP_LABELS: Record<string, string> = {
  env: 'Environment',
  compute_config: 'Compute',
  download_config: 'Download',
  monitor_config: 'Monitor',
}

/** `num_gpus_per_node` → `Num gpus per node`. */
function humanizeKey(key: string): string {
  const spaced = key.replace(/[_-]+/g, ' ').trim()
  return spaced.charAt(0).toUpperCase() + spaced.slice(1)
}

/** Scalars render as rows; objects/arrays are left to the raw disclosure. */
function isScalar(value: unknown): boolean {
  return (
    value === null ||
    typeof value === 'string' ||
    typeof value === 'number' ||
    typeof value === 'boolean'
  )
}

interface ConfigGroup {
  label: string
  rows: { key: string; value: string }[]
}

/**
 * Reshape the leftover config into the drawer's Configuration section: a few
 * named subgroups of scalar rows. `env` is nested inside a launcher block
 * (config.bash.env, config.docker.env — see gbserver/environment/bash.py), so
 * both the top level and one level down are scanned.
 *
 * Only scalars are promoted; nested objects stay in the raw disclosure, which
 * keeps this from trying to render arbitrary depth as a flat table.
 */
function buildConfigGroups(config: Record<string, unknown>): ConfigGroup[] {
  const groups: ConfigGroup[] = []

  const addGroup = (key: string, raw: unknown) => {
    const label = CONFIG_GROUP_LABELS[key]
    if (!label || !raw || typeof raw !== 'object' || Array.isArray(raw)) return
    const rows = Object.entries(raw as Record<string, unknown>)
      .filter(([, v]) => isScalar(v))
      .map(([k, v]) => ({ key: humanizeKey(k), value: String(v) }))
    if (rows.length > 0) groups.push({ label, rows })
  }

  for (const [key, value] of Object.entries(config)) {
    addGroup(key, value)
    // Launcher blocks (bash/docker/skypilot/…) carry their own `env`.
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      const nested = value as Record<string, unknown>
      if (nested.env) addGroup('env', nested.env)
    }
  }

  return groups
}

/**
 * The step's runtime metadata as scalar rows. This is StoredStepRun.metadata —
 * key/values the step pushed at execution time via the LLMB_STEP_METADATA hook
 * (a resolved git `commit_hash` is the documented example), distinct from the
 * declared `config`. `commit_hash` is pulled out separately for the Source
 * block, so it is excluded here to avoid showing it twice.
 */
function metadataRows(
  metadata: Record<string, unknown> | undefined
): { key: string; value: string }[] {
  if (!metadata) return []
  return Object.entries(metadata)
    .filter(([key]) => key !== 'commit_hash')
    .filter(([, value]) => isScalar(value))
    .map(([key, value]) => ({ key: humanizeKey(key), value: String(value) }))
}

/** The runtime-resolved commit SHA, if the step recorded one. */
function commitHash(step: BuildStepRun): string | undefined {
  const value = step.metadata?.commit_hash
  return typeof value === 'string' && value.trim() ? value.trim() : undefined
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

/** A titled block within the drawer — the drawer's only structural divider. */
function Section({
  title,
  children,
  action,
}: {
  title: string
  children: React.ReactNode
  action?: React.ReactNode
}) {
  return (
    <section className={styles.stepSection}>
      <div className={styles.stepSectionHead}>
        <h5 className={styles.stepSectionTitle}>{title}</h5>
        {action}
      </div>
      {children}
    </section>
  )
}

/**
 * Started / Finished as two bordered cells. Duration is deliberately absent —
 * the drawer header already reads "Completed in 6s", so repeating it here would
 * be the third copy of the same number.
 */
function ExecutionSummary({ step }: { step: BuildStepRun }) {
  return (
    <dl className={styles.stepExecution}>
      <div className={styles.stepExecutionCell}>
        <dt className={styles.stepExecutionLabel}>Started</dt>
        <dd className={styles.stepExecutionValue}>{formatClock(step.started_at)}</dd>
      </div>
      <div className={styles.stepExecutionCell}>
        <dt className={styles.stepExecutionLabel}>Finished</dt>
        <dd className={styles.stepExecutionValue}>{formatClock(finishedAt(step))}</dd>
      </div>
    </dl>
  )
}

function CopyButton({ value }: { value: string }) {
  const [copied, setCopied] = React.useState(false)

  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(value)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1500)
    } catch {
      // Clipboard is unavailable outside a secure context; the command is
      // selectable in the block either way, so this needs no error surface.
    }
  }

  return (
    <button type="button" className={styles.stepCopyButton} onClick={onCopy}>
      {copied ? 'Copied' : 'Copy'}
    </button>
  )
}

function StepCard({
  step,
  index,
  total,
  buildId,
}: {
  step: BuildStepRun
  index: number
  total: number
  buildId?: string
}) {
  const extracted = extractCommand(step.config)
  const rest = remainingConfig(step.config, extracted?.sourceKey)
  const configGroups = React.useMemo(() => buildConfigGroups(rest), [rest])
  const restKeys = Object.keys(rest)
  const metaRows = metadataRows(step.metadata)
  const commit = commitHash(step)
  const message = step.status_msg ? cleanStatusMessage(step.status_msg) : ''
  const [showRaw, setShowRaw] = React.useState(false)
  const [showCommand, setShowCommand] = React.useState(false)

  return (
    <li className={styles.stepCard}>
      {/* With a single step the drawer header already names the status and
          timing, so the per-step heading would just repeat it. */}
      {total > 1 && (
        <div className={styles.stepCardHeader}>
          <span className={styles.stepCardName}>
            {index + 1}. {step.step_name}
          </span>
          <BuildStatusBadge status={step.status} />
        </div>
      )}

      {/* One section: the timing cells, then the command that produced them as
          a nested, expandable subsection — the command *is* what was executed,
          so it reads as part of Execution rather than a peer of it. Collapsed by
          default; commands run to many lines and would otherwise push the rest
          of the card below the fold. Only shown when a command was actually
          persisted in config; builtin steps carry no inline command, so the
          subsection is omitted rather than showing an empty placeholder. */}
      <Section title="Execution">
        <ExecutionSummary step={step} />

        {extracted && (
          <details
            className={styles.stepSubsection}
            open={showCommand}
            onToggle={(e) => setShowCommand((e.currentTarget as HTMLDetailsElement).open)}
          >
            <summary className={styles.stepSubsectionSummary}>Command</summary>
            <pre className={styles.stepCommand}>
              <code>{extracted.command}</code>
            </pre>
            {/* Inside the disclosure, so it is only offered when the command it
                copies is actually on screen. */}
            <div className={styles.stepCommandActions}>
              <CopyButton value={extracted.command} />
            </div>
          </details>
        )}
      </Section>

      {/* Recognised config blocks become labelled subgroups of scalar rows;
          everything else stays in the raw disclosure at the foot of the card. */}
      {configGroups.length > 0 && (
        <Section title="Configuration">
          {configGroups.map((group) => (
            <div key={group.label} className={styles.stepConfigGroup}>
              <div className={styles.stepConfigGroupLabel}>{group.label}</div>
              {group.rows.map((row) => (
                <Field key={row.key} label={row.key}>
                  <code className={styles.stepCode}>{row.value}</code>
                </Field>
              ))}
            </div>
          ))}
        </Section>
      )}

      <Section title="Metadata">
        <Field label="Definition URI">
          {step.uri ? <code className={styles.stepCode}>{step.uri}</code> : '—'}
        </Field>
        <Field label="Step ID">
          {step.uuid ? <code className={styles.stepCode}>{step.uuid}</code> : '—'}
        </Field>
        <Field label="Build ID">
          {buildId ? <code className={styles.stepCode}>{buildId}</code> : '—'}
        </Field>
        <Field label="Code commit">
          {commit ? (
            <code className={styles.stepCode}>{commit}</code>
          ) : (
            <span className={styles.stepMuted}>{NOT_RECORDED}</span>
          )}
        </Field>
      </Section>

      {/* Runtime key/values the step reported at execution time (commit_hash is
          pulled up into Metadata above). Persisted in StoredStepRun.metadata and
          shown nowhere else, so it gets its own block when non-empty. */}
      {metaRows.length > 0 && (
        <Section title="Runtime metadata">
          {metaRows.map((row) => (
            <Field key={row.key} label={row.key}>
              <code className={styles.stepCode}>{row.value}</code>
            </Field>
          ))}
        </Section>
      )}

      {/* Only rendered when the message carries something beyond the ids and
          status already shown above — otherwise it was pure duplication. */}
      {message && (
        <Section title="Message">
          <p className={styles.stepMessage}>{message}</p>
        </Section>
      )}

      {/* The full config, verbatim — the escape hatch for anything the grouped
          rows above didn't promote (nested objects, unrecognised blocks). */}
      {restKeys.length > 0 && (
        <details
          className={styles.stepRaw}
          open={showRaw}
          onToggle={(e) => setShowRaw((e.currentTarget as HTMLDetailsElement).open)}
        >
          <summary className={styles.stepRawSummary}>Advanced / Raw parameters</summary>
          <pre className={styles.stepParams}>{JSON.stringify(rest, null, 2)}</pre>
        </details>
      )}
    </li>
  )
}

interface Props {
  targetName: string
  target: BuildTargetRun | undefined
  /** The build's source URI — stands in for a commit SHA, which is not recorded. */
  sourceUri?: string
  /** Shown in each step's Metadata block alongside the step's own id. */
  buildId?: string
}

export default function StepDetailsPanel({
  targetName,
  target,
  sourceUri,
  buildId,
}: Props) {
  const steps = target?.steps ?? []

  return (
    <div className={styles.stepPanel}>
      {steps.length === 0 ? (
        <p className={styles.stepEmpty}>
          No step runs recorded for target <strong>{targetName}</strong>. Steps appear
          here once the target starts running.
        </p>
      ) : (
        <ol className={styles.stepList}>
          {steps.map((step, i) => (
            <StepCard
              key={step.uuid ?? `${step.step_name}-${i}`}
              step={step}
              index={i}
              total={steps.length}
              buildId={buildId}
            />
          ))}
        </ol>
      )}

      {/* Provenance is the least-consulted block in the drawer, so it sits last
          rather than above the steps as it did in the modal. */}
      <Section title="Source">
        <Field label="Source">
          {sourceUri ? (
            <Link href={sourceUri} target="_blank" rel="noopener noreferrer">
              {sourceUri}
            </Link>
          ) : (
            <span className={styles.stepMuted}>{NOT_RECORDED}</span>
          )}
        </Field>
      </Section>
    </div>
  )
}

/** Header metadata for the drawer — derived here so the header and body agree. */
export function stepDrawerSummary(target: BuildTargetRun | undefined): {
  status: BuildStepRun['status'] | undefined
  subtitle: string
  summary: string | undefined
} {
  const steps = target?.steps ?? []
  if (steps.length === 0) {
    return { status: undefined, subtitle: 'Target', summary: undefined }
  }

  // A multi-step target is only as healthy as its worst step; a single-step
  // target just reports that step.
  const status =
    steps.find((s) => s.status === 'failed')?.status ??
    steps.find((s) => s.status === 'running')?.status ??
    steps[steps.length - 1].status

  const subtitle =
    steps.length === 1
      ? `Step · ${steps[0].step_name}`
      : `${steps.length} steps · ${steps.map((s) => s.step_name).join(' → ')}`

  // Span the whole target: first start to last finish.
  const started = steps[0]?.started_at
  const finished = finishedAt(steps[steps.length - 1])
  const duration = formatDuration(started, finished)
  const stamp = formatDateTime(finished ?? started)
  const summary = [duration ? `Completed in ${duration}` : undefined, stamp]
    .filter(Boolean)
    .join(' · ')

  return { status, subtitle, summary: summary || undefined }
}
