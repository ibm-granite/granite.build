// Pure helpers for the Start Tuning wizard's "My tuned models" source.
// Type-only imports: tests require() this file under plain `node --test`.
import type { TunedModel, TuningAsset } from '@granite-build/ui-core/types'

export type TunedModelCheck = { ok: true } | { ok: false; reason: string; retryable: boolean }

const ADAPTER_REASON = 'This tuning produced a LoRA adapter, not full model weights. Select a full-weight model.'
const LEGACY_REASON = 'This model predates the current output format and cannot be loaded as a base model.'
const BUILD_FAILED_REASON = 'The tuning job for this model did not succeed.'
const MISSING_REPO_REASON = 'The repository for this model no longer exists. It may have been deleted or renamed.'
const UNREACHABLE_REASON = 'Hugging Face could not be reached to verify this model. Retry, or try again later.'

/** ComboBox label: the experiment name alone; the repo id stands in when a job has none. */
export function tunedModelLabel(m: TunedModel): string {
  return m.experiment_name || m.repo_id
}

/**
 * ComboBox labels for a list, keyed by `job_id`. Experiment names are not unique,
 * so only the items whose name collides with another item in the list get the
 * short repo id appended (`grpo-math · a69082b7`) — enough to tell them apart
 * while keeping unique names bare.
 */
export function tunedModelLabels(models: TunedModel[]): Map<string, string> {
  const counts = new Map<string, number>()
  for (const m of models) counts.set(tunedModelLabel(m), (counts.get(tunedModelLabel(m)) ?? 0) + 1)
  return new Map(
    models.map((m) => {
      const label = tunedModelLabel(m)
      if ((counts.get(label) ?? 0) < 2) return [m.job_id, label]
      const repoName = m.repo_id.split('/').pop() ?? m.repo_id
      return [m.job_id, `${label} · ${repoName.replace(/^autotunex_/, '')}`]
    }),
  )
}

/** What the picked model is, shown below the ComboBox: base model, tuning method, repository. */
export function tunedModelDetails(m: TunedModel): { label: string; value: string }[] {
  const kind = m.rl_tuner_type && m.rl_tuner_type !== 'none' ? m.rl_tuner_type : m.tuning_type
  return [
    { label: 'Base model', value: m.base_model },
    { label: 'Tuning method', value: kind?.toUpperCase() ?? '' },
    { label: 'Repository', value: m.repo_id },
  ].filter((d) => d.value)
}

/**
 * Whether a tuned output's HF repo (its result-report listing, repo-relative
 * paths) can be loaded as a base model. A job reads its base model from the
 * repo root, so a root `config.json` is required; an `adapter_config.json`
 * marks a PEFT adapter, which fm-tune cannot use as a base.
 */
export function checkTunedModelAssets(assets: Pick<TuningAsset, 'path'>[]): TunedModelCheck {
  const paths = new Set(assets.map((a) => a.path))
  if (paths.has('adapter_config.json')) return { ok: false, reason: ADAPTER_REASON, retryable: false }
  if (!paths.has('config.json')) return { ok: false, reason: LEGACY_REASON, retryable: false }
  return { ok: true }
}

/** The check's outcome when the result-report request itself failed (409 = build not successful). */
export function tunedModelCheckFailure(status: number | undefined): TunedModelCheck {
  if (status === 409) return { ok: false, reason: BUILD_FAILED_REASON, retryable: false }
  if (status === 404) return { ok: false, reason: MISSING_REPO_REASON, retryable: false }
  return { ok: false, reason: UNREACHABLE_REASON, retryable: true }
}
