// Pure helpers for the Start Tuning wizard's "My tuned models" source.
// Type-only imports: tests require() this file under plain `node --test`.
import type { TunedModel, TuningAsset } from '@granite-build/ui-core/types'

export type TunedModelCheck = { ok: true } | { ok: false; reason: string; retryable: boolean }

const ADAPTER_REASON = 'This output is a LoRA/PEFT adapter; only full-weight models can be tuned further.'
const LEGACY_REASON = "This model's output predates model-at-root packaging and can't be used as a base."
const BUILD_FAILED_REASON = "This job's build did not succeed."
const MISSING_REPO_REASON = "This model's repository no longer exists."
const UNREACHABLE_REASON = "Couldn't reach HuggingFace to verify this model."

/** One-line ComboBox label: experiment · base model · tuning kind · repo. */
export function tunedModelLabel(m: TunedModel): string {
  const kind = m.rl_tuner_type && m.rl_tuner_type !== 'none' ? m.rl_tuner_type : m.tuning_type
  return [m.experiment_name, m.base_model, kind, m.repo_id].filter(Boolean).join(' · ')
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
