import type { Build, BuildStatus, BuildTargetRun } from '@granite-build/ui-core/types'
import { formatDurationBetween } from '@granite-build/ui-core/lib/duration'

// Pure header logic for the step drawer, kept out of StepDetailsPanel.tsx so
// it can be unit-tested without React (tests/step-drawer-summary.test.js).

/** Build statuses that mean the build is still in flight. Mirrors LineagePanel. */
export const ACTIVE_STATUSES = new Set(['running', 'submitted', 'pending', 'cancel_requested'])

/**
 * Target statuses that are not terminal — the target has not reached an outcome
 * yet, so it may still be elapsing. Targets and builds draw from the same
 * backend `Status` enum (src/gbserver/types/status.py), so this currently has
 * the same members as ACTIVE_STATUSES. It is kept separate because it answers a
 * different question — is this TARGET still elapsing, versus is the BUILD still
 * in flight — and the two diverge as soon as either gains a status the other
 * cannot report.
 *
 * Terminal by omission: success, failed, invalid, cancelled.
 *
 * `retry_pending` is in the BuildStatus union and BuildStatusBadge renders it as
 * a yellow "Retrying", so it belongs here even though src/gbserver/types/status.py
 * has no RETRY_PENDING member and nothing can emit it today — a target awaiting a
 * retry is still elapsing, and omitting it would print "Ran for 2m 4s" under a
 * "Retrying" badge the moment the backend gains the status.
 *
 * `planned` is deliberately absent: it is frontend-only (synthesised for targets
 * read from the build definition) and never reaches a target's `status`.
 */
export const UNFINISHED_TARGET_STATUSES = new Set([
  'running',
  'submitted',
  'pending',
  'cancel_requested',
  'retry_pending',
])

/**
 * `Aug 22, 2026 at 10:51:28 PDT` — the one place a full timestamp is spelled
 * out. The timezone abbreviation is included because the viewer and the build
 * launcher are often in different zones, so a bare wall-clock time is ambiguous.
 */
export function formatDateTime(value: string | undefined): string | undefined {
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

/**
 * Header metadata for the drawer — derived here so the header and body agree.
 *
 * `build` is optional but should be passed whenever it is known: a target left
 * non-terminal under a stopped build has no `finished_at` of its own, and the
 * build's finish time bounds its duration.
 */
export function stepDrawerSummary(
  target: BuildTargetRun | undefined,
  build?: Pick<Build, 'status' | 'finished_at'>,
): {
  status: BuildStatus | undefined
  subtitle: string
  summary: string | undefined
} {
  const steps = target?.steps ?? []
  // No target at all — a planned target from the build definition, which has no
  // runtime row yet. Nothing to report.
  if (!target) {
    return { status: undefined, subtitle: 'Target', summary: undefined }
  }

  // The runner sets `target.status` from its own target-level status events
  // (buildrunner.py, __update_stored_target_run), so it is the authoritative
  // outcome — unlike ranking over `steps`, which is incomplete: a step only
  // gets a row once it emits its first status event, so a target whose second
  // step hasn't started yet can't be judged from its steps alone.
  //
  // Deliberately NOT gated on `steps.length`: the status is exactly what we can
  // still report for a target run that exists but whose step rows have not
  // arrived yet — the very gap that motivated reading it from the target. An
  // early return on an empty `steps` would blank the badge in that window and
  // then pop it in on a later poll.
  //
  // Unconfirmed: whether the server moves a target to a terminal status when a
  // build is cancelled mid-target. If it does not, such a target keeps a stale
  // `running` status and the header reads "Running for" while the build is
  // active, then "Ran for" once it stops — wrong in the badge but never wrong
  // in the label, since "Ran for" makes no claim about the outcome. Fixing that
  // belongs on the server, not here.
  const status = target.status

  // The step list drives only the subtitle; with no step rows yet there are no
  // names to list, but the target's own status and timing still stand.
  const subtitle =
    steps.length === 0
      ? 'Target'
      : steps.length === 1
        ? `Step · ${steps[0].step_name}`
        : `${steps.length} steps · ${steps.map((s) => s.step_name).join(' → ')}`

  // Timing comes from the target run itself: the runner stamps `started_at` on
  // its first entry into RUNNING and `finished_at` on its first terminal status
  // (buildrunner.py, _apply_run_timestamps). Rebuilding the span from step rows
  // is the incomplete-list trap again — a step has no row until its first event.
  //
  // A stopped build overrides a non-terminal target status: nothing under it can
  // still run. That is also the one case with no target `finished_at`, so the
  // build's own finish time bounds the span there — but only for a target that
  // actually started. One that never ran (skipped, or a retry still queued when
  // the build was cancelled) has no span to bound, and borrowing the build's
  // finish would stamp it with a time at which it did nothing.
  const buildStopped = Boolean(build && !ACTIVE_STATUSES.has(build.status))
  const isRunning = UNFINISHED_TARGET_STATUSES.has(status) && !buildStopped
  const started = target.started_at
  const finished = isRunning
    ? undefined
    : (target.finished_at ?? (started ? build?.finished_at : undefined))
  const duration = formatDurationBetween(
    started,
    isRunning ? new Date().toISOString() : finished,
  )
  // The timestamp slot means "when it finished" for a target that has stopped.
  // A running target has no finish time, so it shows its start instead — label
  // that explicitly, or the same slot silently means two different instants.
  const stamp = finished
    ? formatDateTime(finished)
    : started
      ? `started ${formatDateTime(started)}`
      : undefined
  // A target that ended badly, or never finished, did not "complete" — say how
  // long it ran instead, or the header reads "Completed in 2m 4s" directly
  // under a red Failed (or Pending) badge. Keyed off `target.status`, the
  // authoritative outcome, rather than `isRunning`: a step left `running`
  // under a build that has since stopped is not still elapsing (`isRunning`
  // is false) but it never completed either, so it must still read "Ran for".
  const durationLabel = isRunning ? 'Running for' : status === 'success' ? 'Completed in' : 'Ran for'
  const summary = [duration ? `${durationLabel} ${duration}` : undefined, stamp]
    .filter(Boolean)
    .join(' · ')

  return { status, subtitle, summary: summary || undefined }
}
