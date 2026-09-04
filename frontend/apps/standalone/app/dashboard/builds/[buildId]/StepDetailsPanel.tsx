'use client'

import * as React from 'react'
import { CopyButton, Link } from '@carbon/react'
import styles from './LineagePanel.module.scss'
import type { BuildStepRun, BuildTargetRun } from '@granite-build/ui-core/types'
import { BuildStatusBadge } from '@granite-build/ui-core/components/BuildStatusBadge'
import { formatDurationBetween } from '@granite-build/ui-core/lib/duration'

const NOT_RECORDED = 'Not recorded'

/** `10:51:22` — the clock time alone, for the compact Execution row. */
function formatClock(value: string | undefined): string {
  if (!value) return '—'
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  return parsed.toLocaleTimeString([], { hour12: false })
}

/**
 * `Aug 22, 2026 at 10:51:28 PDT` — the one place a full timestamp is spelled
 * out. The timezone abbreviation is included because the viewer and the build
 * launcher are often in different zones, so a bare wall-clock time is ambiguous.
 */
function formatDateTime(value: string | undefined): string | undefined {
  if (!value) return undefined
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  const date = parsed.toLocaleDateString([], {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })
  const time = parsed.toLocaleTimeString([], { hour12: false, timeZoneName: 'short' })
  return `${date} at ${time}`
}

/** The step's own finish time, falling back to the mirrored `updated_at`. */
function finishedAt(step: BuildStepRun): string | undefined {
  return step.finished_at ?? step.updated_at
}

/**
 * Keys in the status message that the drawer already shows elsewhere — in the
 * header (Status) or the Metadata block (the ids and the step URI). gbserver
 * emits them as a padded `Key      : Value` table, so the message is almost
 * entirely a restatement of fields shown above.
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
 * This is a heuristic over free text, so it is deliberately conservative: a line
 * is dropped only when it *looks like a padded table row* for a known key, not
 * merely because it starts with `<known word>:`. Otherwise prose such as
 * "Status: everything nominal" or "Type: the model was rebuilt" — a sentence
 * that happens to begin with a redundant key — would be silently deleted.
 *
 * The table lives *inside* the fence and `extra_msg` after it, so the heuristic
 * applies to fenced lines; unfenced free text is only ever stripped of headings.
 *
 * The two signals that a line is a table row rather than a sentence:
 *   - a multi-word key ("Step URI", "Build ID") — prose almost never opens with
 *     one of these followed by a colon; or
 *   - a single-word key aligned with the padded-table gap gbserver emits (two or
 *     more spaces before the colon: `Status   : ...`), which prose does not have.
 *
 * Returns '' when nothing new is left, so the caller can omit the section.
 */
function cleanStatusMessage(msg: string): string {
  // Track fence open/close. gbserver emits the padded metadata table *inside* a
  // fenced block (see Run.create_message in gbserver/build/run.py), and
  // `extra_msg` is appended after the closing fence — so the redundant-key
  // heuristic must run on fenced lines, and free text outside survives verbatim.
  let inFence = false
  return msg
    .split('\n')
    .filter((line) => {
      if (/^\s*```/.test(line)) {
        inFence = !inFence
        return false // drop the fence marker itself either way
      }

      // Markdown headings are gbserver's section titles for the fields shown
      // above; drop them. Only outside a fence, where a `#` line is a heading
      // rather than a shell comment in a captured command.
      if (!inFence && /^#+\s/.test(line)) return false

      // Capture the key, the run of spaces before the colon, and the value.
      const match = /^\s*([A-Za-z][A-Za-z ]*?)( *):\s*(.*)$/.exec(line)
      if (!match) return true // prose / blank line — keep

      const key = match[1].trim().toLowerCase()
      if (!REDUNDANT_MESSAGE_KEYS.has(key)) return true // unknown key — keep

      const paddedBeforeColon = match[2].length >= 2
      const multiWordKey = key.includes(' ')
      // Drop only when it reads as a table row, not a sentence.
      return !(multiWordKey || paddedBeforeColon)
    })
    .join('\n')
    // Collapse the blank lines the removed rows leave behind.
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

/**
 * Quote one argv element for a POSIX shell.
 *
 * An argv array is a list of literal arguments, so a plain `.join(' ')` turns
 * `["sh", "-c", "echo hi && rm -rf x"]` into a command whose `&&` the shell now
 * interprets — the Copy button would hand the user something that does not match
 * what ran. Single-quote anything outside the safe set, escaping embedded single
 * quotes the usual `'\''` way.
 */
function shellQuote(arg: unknown): string {
  const s = String(arg)
  if (s.length > 0 && /^[A-Za-z0-9_@%+=:,./-]+$/.test(s)) return s
  return "'" + s.replace(/'/g, "'\\''") + "'"
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
    const container = config[key]
    if (!isPlainObject(container)) continue
    const candidate = container.command
    if (typeof candidate === 'string' && candidate.trim()) {
      return { command: candidate.trim(), sourceKey: key }
    }
    // Some launchers pass an argv array (`["sh", "-c", "…"]`) rather than a
    // string; join it into a command that is safe to paste, since the Copy
    // button hands this straight to a shell.
    if (Array.isArray(candidate) && candidate.length > 0) {
      return { command: candidate.map(shellQuote).join(' '), sourceKey: key }
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

  const container = config[sourceKey]
  // extractCommand only reports a container sourceKey after confirming it was a
  // plain object, so this normally holds — but guard anyway rather than spread a
  // scalar into index-keyed junk if the two ever fall out of step.
  if (!isPlainObject(container)) return config
  const { command: _omitted, ...containerRest } = container
  if (Object.keys(containerRest).length === 0) {
    const { [sourceKey]: _dropped, ...rest } = config
    return rest
  }
  return { ...config, [sourceKey]: containerRest }
}

/**
 * Nicer labels for the config blocks we recognise. Only an override table now:
 * any block not listed still gets a heading (its key, humanized) rather than
 * being dropped, so the section shows every block the step actually carried.
 */
const CONFIG_GROUP_LABELS: Record<string, string> = {
  // Launcher / environment blocks (see gbserver/environment/*.py).
  env: 'Environment',
  k8s: 'Kubernetes',
  kube_config: 'Kubernetes',
  lsf: 'LSF',
  docker: 'Docker',
  skypilot: 'SkyPilot',
  bash: 'Bash',
  cloud_config: 'Cloud',
  cos_config: 'COS',
  environment_config: 'Environment',
  launcher_config: 'Launcher',
  // Lifecycle / IO blocks written into step config.
  compute_config: 'Compute',
  download_config: 'Download',
  monitor_config: 'Monitor',
  setup_config: 'Setup',
  teardown_config: 'Teardown',
  input_values_config: 'Inputs',
  hfpull_config: 'HF pull',
  hfpush_config: 'HF push',
  lhpull_config: 'LH pull',
  lhpush_config: 'LH push',
  // Common free-form model/training params.
  tuning_config: 'Tuning',
  fsdp_config: 'FSDP',
}

/** Heading for a config block: a curated label if we have one, else the humanized key. */
function groupLabel(key: string): string {
  return CONFIG_GROUP_LABELS[key] ?? humanizeKey(key)
}

/** `num_gpus_per_node` → `Num gpus per node`. */
function humanizeKey(key: string): string {
  const spaced = key.replace(/[_-]+/g, ' ').trim()
  return spaced.charAt(0).toUpperCase() + spaced.slice(1)
}

/**
 * A scalar carrying an actual value — the only thing rendered as a row. Empty
 * strings and null are treated as "no value" and dropped so the UI shows only
 * the keys the step actually set, not blank placeholder rows.
 */
function hasValue(value: unknown): boolean {
  if (value === null || value === undefined) return false
  if (typeof value === 'string') return value.trim() !== ''
  return typeof value === 'number' || typeof value === 'boolean'
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

/**
 * A plain object, not an array or scalar. config is free-form from gbserver, so
 * a container key (`command_config`, `docker`, …) may hold a string or array;
 * treating that as an object and spreading it yields index-keyed junk, so every
 * `config[key]` access below is gated on this first.
 */
function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

interface ConfigGroup {
  label: string
  rows: { key: string; value: string }[]
}

/**
 * Reshape the config into the drawer's Configuration section: named subgroups
 * of scalar rows. config is free-form from gbserver — a step carries whatever
 * launcher/lifecycle blocks it ran with (`k8s`, `lsf`, `tuning_config`,
 * `download_config`, …), so nothing is allowlisted: every block that holds
 * scalars becomes a subgroup, and every top-level scalar goes under "General".
 * CONFIG_GROUP_LABELS only supplies a nicer heading for the ones we recognise.
 *
 * Only scalars become rows; a value that is itself a nested object/array stays
 * in the raw disclosure, so this never tries to render arbitrary depth as a
 * flat table. `env` is nested inside a launcher block (config.bash.env,
 * config.docker.env — see gbserver/environment/bash.py), so one level down is
 * scanned too.
 */
function buildConfigGroups(config: Record<string, unknown>): ConfigGroup[] {
  const groups: ConfigGroup[] = []

  const addGroup = (label: string, raw: unknown) => {
    if (!isPlainObject(raw)) return
    const rows = Object.entries(raw)
      .filter(([, v]) => hasValue(v))
      .map(([k, v]) => ({ key: humanizeKey(k), value: String(v) }))
    if (rows.length > 0) groups.push({ label, rows })
  }

  const generalRows: { key: string; value: string }[] = []

  for (const [key, value] of Object.entries(config)) {
    if (isScalar(value)) {
      // A bare top-level param (`group`, `is_dry_run_compatible`, …) — collect
      // these into one "General" group so they aren't stranded in the raw blob.
      // Skip blank/null so only keys the step actually set show up.
      if (hasValue(value)) generalRows.push({ key: humanizeKey(key), value: String(value) })
      continue
    }
    addGroup(groupLabel(key), value)
    // Launcher blocks (bash/docker/skypilot/…) carry their own `env`.
    if (isPlainObject(value) && value.env) addGroup(CONFIG_GROUP_LABELS.env, value.env)
  }

  if (generalRows.length > 0) groups.unshift({ label: 'General', rows: generalRows })

  return groups
}

/**
 * The step's runtime metadata as scalar rows. This is StoredStepRun.metadata —
 * key/values the step pushed at execution time via the LLMB_STEP_METADATA hook
 * (a resolved git `commit_hash` is the documented example), distinct from the
 * declared `config`.
 */
function metadataRows(
  metadata: Record<string, unknown> | undefined
): { key: string; value: string }[] {
  if (!metadata) return []
  return Object.entries(metadata)
    .filter(([, value]) => hasValue(value))
    .map(([key, value]) => ({ key: humanizeKey(key), value: String(value) }))
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
  const started = step.started_at
  const finished = finishedAt(step)
  return (
    <dl className={styles.stepExecution}>
      <div className={styles.stepExecutionCell}>
        <dt className={styles.stepExecutionLabel}>Started</dt>
        {/* The cell shows a bare clock; the full date + timezone lives in the
            tooltip so it resolves the "which zone / which day" ambiguity without
            widening the compact row. */}
        <dd className={styles.stepExecutionValue} title={formatDateTime(started)}>
          {formatClock(started)}
        </dd>
      </div>
      <div className={styles.stepExecutionCell}>
        <dt className={styles.stepExecutionLabel}>Finished</dt>
        <dd className={styles.stepExecutionValue} title={formatDateTime(finished)}>
          {formatClock(finished)}
        </dd>
      </div>
      {/* The launcher is read straight off config.launcher, but the container
          image is resolved inside the compute environment at launch time and is
          usually not persisted — so this is best-effort and falls back to the
          same "Not recorded" placeholder the Source field uses. See
          docs/builds/lineage.md. */}
      <div className={styles.stepExecutionCell}>
        <dt className={styles.stepExecutionLabel}>Launcher</dt>
        <dd className={styles.stepExecutionValue}>
          {step.launcher ?? <span className={styles.stepMuted}>{NOT_RECORDED}</span>}
        </dd>
      </div>
      <div className={styles.stepExecutionCell}>
        <dt className={styles.stepExecutionLabel}>Container image</dt>
        <dd className={styles.stepExecutionValue} title={step.image}>
          {step.image ?? <span className={styles.stepMuted}>{NOT_RECORDED}</span>}
        </dd>
      </div>
    </dl>
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
  const extracted = React.useMemo(() => extractCommand(step.config), [step.config])
  // remainingConfig spreads a fresh object whenever a command was extracted (the
  // common case), so memoize on its actual inputs — otherwise the configGroups
  // memo below never hits cache and buildConfigGroups reruns on every poll tick.
  const rest = React.useMemo(
    () => remainingConfig(step.config, extracted?.sourceKey),
    [step.config, extracted?.sourceKey]
  )
  const configGroups = React.useMemo(() => buildConfigGroups(rest), [rest])
  const restKeys = Object.keys(rest)
  const metaRows = metadataRows(step.metadata)
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
              <CopyButton
                autoAlign
                feedback="Copied!"
                iconDescription="Copy command"
                onClick={() => navigator.clipboard.writeText(extracted.command)}
                size="sm"
              />
            </div>
          </details>
        )}
      </Section>

      {/* Config blocks become labelled subgroups of scalar rows; the full
          verbatim config lives in a nested "Raw parameters" subsection so it
          reads as part of Configuration rather than a peer of it — the escape
          hatch for anything the grouped rows didn't promote (nested objects,
          long value dumps). Always rendered — with a placeholder when no scalar
          groups were recognised — so the section's absence never reads as
          "config missing". */}
      <Section title="Configuration">
        {configGroups.length > 0 ? (
          configGroups.map((group, groupIndex) => (
            // Key on index, not label: a top-level `env` and a launcher block's
            // nested `env` both label as "Environment", so labels aren't unique.
            <div key={`${group.label}-${groupIndex}`} className={styles.stepConfigGroup}>
              <div className={styles.stepConfigGroupLabel}>{group.label}</div>
              {group.rows.map((row) => (
                <Field key={row.key} label={row.key}>
                  <code className={styles.stepCode}>{row.value}</code>
                </Field>
              ))}
            </div>
          ))
        ) : (
          <span className={styles.stepMuted}>{NOT_RECORDED}</span>
        )}

        {restKeys.length > 0 && (
          <details
            className={styles.stepSubsection}
            open={showRaw}
            onToggle={(e) => setShowRaw((e.currentTarget as HTMLDetailsElement).open)}
          >
            <summary className={styles.stepSubsectionSummary}>Raw parameters</summary>
            <pre className={styles.stepParams}>{JSON.stringify(rest, null, 2)}</pre>
          </details>
        )}
      </Section>

      <Section title="Details">
        <Field label="Definition URI">
          {step.uri ? <code className={styles.stepCode}>{step.uri}</code> : '—'}
        </Field>
        <Field label="Step ID">
          {step.uuid ? <code className={styles.stepCode}>{step.uuid}</code> : '—'}
        </Field>
        <Field label="Build ID">
          {buildId ? <code className={styles.stepCode}>{buildId}</code> : '—'}
        </Field>
      </Section>

      {/* Runtime key/values the step reported at execution time. Persisted in
          StoredStepRun.metadata. Always rendered — with a placeholder when the
          step emitted none — so the section is a stable part of the drawer
          rather than appearing only for the steps that happen to push metadata. */}
      <Section title="Metadata">
        {metaRows.length > 0 ? (
          metaRows.map((row) => (
            <Field key={row.key} label={row.key}>
              <code className={styles.stepCode}>{row.value}</code>
            </Field>
          ))
        ) : (
          <span className={styles.stepMuted}>{NOT_RECORDED}</span>
        )}
      </Section>

      {/* Only rendered when the message carries something beyond the ids and
          status already shown above — otherwise it was pure duplication. */}
      {message && (
        <Section title="Message">
          <p className={styles.stepMessage}>{message}</p>
        </Section>
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

  // Span the whole target: first start to last finish. Steps arrive sorted by
  // start time (adaptTargetRun), so steps[0] is the earliest start — but take the
  // latest finish explicitly, since a step that started earlier may finish later.
  const started = steps[0]?.started_at
  // A step still running has no finish time, so the latest finish among the
  // *finished* steps understates the target: a target whose second step has run
  // for 40m would otherwise report only the 5m its first step took. While any
  // step is unfinished the target is still elapsing, so measure to now instead.
  const isRunning = steps.some((s) => !finishedAt(s))
  const finishTimes = steps
    .map((s) => finishedAt(s))
    .filter((t): t is string => Boolean(t) && Number.isFinite(Date.parse(t as string)))
  const lastFinished = finishTimes.length
    ? finishTimes.reduce((latest, t) => (Date.parse(t) > Date.parse(latest) ? t : latest))
    : undefined
  const finished = isRunning ? undefined : lastFinished
  const duration = formatDurationBetween(
    started,
    isRunning ? new Date().toISOString() : finished,
  )
  const stamp = formatDateTime(finished ?? started)
  // A failed or cancelled target did not "complete" — say how long it ran instead,
  // or the header reads "Completed in 2m 4s" directly under a red Failed badge.
  // An in-flight target has not completed either; it is still running.
  const durationLabel =
    status === 'failed' || status === 'cancelled'
      ? 'Ran for'
      : isRunning
        ? 'Running for'
        : 'Completed in'
  const summary = [duration ? `${durationLabel} ${duration}` : undefined, stamp]
    .filter(Boolean)
    .join(' · ')

  return { status, subtitle, summary: summary || undefined }
}
