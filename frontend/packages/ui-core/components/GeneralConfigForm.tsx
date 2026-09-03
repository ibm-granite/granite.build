'use client'

import { NumberInput, Tile } from '@carbon/react'
import type { ConfigData } from '../types'
import { maxConcurrentTrialsCap } from '../lib/autotunex/hyperparamValues'
import { TimeInput } from './TimeInput'
import layoutStyles from './layout.module.scss'

interface GeneralConfigFormProps {
  config: ConfigData
  onConfigChange: (updater: (prev: ConfigData) => ConfigData) => void
}

/**
 * The "General" tab of the Step 2 config editor: the handful of headline
 * settings pulled out of the real `training_config`/`tune_config` sections.
 *
 * All four fields live in ONE two-column grid rather than two independent flex
 * rows, so column edges line up across rows exactly as they do in the source
 * AutoTuneX form (which gets that for free from Carbon's `Row`/`Column`).
 *
 * Note: the source's `isTuning` branch (an aLoRA "Invocation string" field) is
 * deliberately not ported. `GeneralConfigForm` has exactly one call site in
 * AutoTuneX — `<GeneralConfigForm bind:config />` — which never passes
 * `isTuning`, so that block is unreachable dead code there and the field does
 * not belong on this tab (aLoRA's invocation_string is a `tuners_config`
 * hyperparameter, editable under "Tuners").
 */
export function GeneralConfigForm({ config, onConfigChange }: GeneralConfigFormProps) {
  const numGpusPerTrial = config.training_config.num_gpus_per_trial
  const maxConcurrentTrials = config.tune_config.max_concurrent_trials
  const numSamples = config.tune_config.num_samples
  const timeBudget = config.tune_config.time_budget_s
  // Clearing "Num GPUs per trial" reports 0, so this divisor is guarded — see
  // maxConcurrentTrialsCap.
  const concurrentCap = maxConcurrentTrialsCap(numGpusPerTrial.max_val, numGpusPerTrial.default)

  return (
    <Tile>
      <div className={layoutStyles.fieldGrid}>
        <NumberInput
          id="num_gpus_per_trial"
          label="Num GPUs per trial"
          helperText={numGpusPerTrial.description}
          value={numGpusPerTrial.default}
          min={numGpusPerTrial.min_val}
          max={numGpusPerTrial.max_val}
          invalidText={`Value must be between ${numGpusPerTrial.min_val} and ${numGpusPerTrial.max_val}`}
          step={1}
          onChange={(_e, { value }) => {
            const num = typeof value === 'number' ? value : Number(value)
            onConfigChange((prev) => ({
              ...prev,
              training_config: { ...prev.training_config, num_gpus_per_trial: { ...prev.training_config.num_gpus_per_trial, default: num } },
              tune_config: {
                ...prev.tune_config,
                max_concurrent_trials: {
                  ...prev.tune_config.max_concurrent_trials,
                  default: maxConcurrentTrialsCap(prev.training_config.num_gpus_per_trial.max_val, num),
                },
              },
            }))
          }}
        />

        <NumberInput
          id="max_concurrent_trials"
          label="Max concurrent trials"
          helperText={maxConcurrentTrials.description}
          value={maxConcurrentTrials.default}
          min={maxConcurrentTrials.min_val}
          max={concurrentCap}
          invalidText={`Value must be between ${maxConcurrentTrials.min_val} and ${concurrentCap}`}
          step={1}
          onChange={(_e, { value }) => {
            const num = typeof value === 'number' ? value : Number(value)
            onConfigChange((prev) => ({
              ...prev,
              tune_config: { ...prev.tune_config, max_concurrent_trials: { ...prev.tune_config.max_concurrent_trials, default: num } },
            }))
          }}
        />

        <NumberInput
          id="num_samples"
          label="Num samples (trials)"
          helperText={numSamples.description}
          value={numSamples.default}
          min={numSamples.min_val}
          max={numSamples.max_val}
          invalidText={`Value must be between ${numSamples.min_val} and ${numSamples.max_val}`}
          step={1}
          onChange={(_e, { value }) => {
            const num = typeof value === 'number' ? value : Number(value)
            onConfigChange((prev) => ({ ...prev, tune_config: { ...prev.tune_config, num_samples: { ...prev.tune_config.num_samples, default: num } } }))
          }}
        />

        {timeBudget && (
          <TimeInput
            label="Time Budget"
            value={timeBudget}
            onChange={(next) => onConfigChange((prev) => ({ ...prev, tune_config: { ...prev.tune_config, time_budget_s: next } }))}
          />
        )}
      </div>
    </Tile>
  )
}
