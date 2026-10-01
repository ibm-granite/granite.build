import type { ConfigData, ConfigForm, TuningGoal } from '../../types'

// Canonical tab order for advanced mode (ensures consistent ordering regardless of Object.keys order).
export const SECTION_ORDER = [
  'general_config',
  'tune_config',
  'training_config',
  'tokenizer_config',
  'tuners_config',
  'training_rl_config',
  'tuners_rl_config',
]

export interface SectionNamesParams {
  /** false = Basic, true = Advanced */
  mode: boolean
  trainingMode: 'offline_tuning' | 'online_tuning'
  /** The wizard's Step-0 goal ('sft' hides RL tuners). */
  presetGoal: TuningGoal | null
  /** The real top-level object keys present in the fetched template. */
  allSectionKeys: string[]
}

/**
 * Compute the ContentSwitcher tabs for the Step 2 config editor, mirroring the
 * source AutoTuneX Svelte form (CreateConfigForm.svelte).
 *
 * `general_config` is a *synthetic* UI-only section: it is rendered by
 * `GeneralConfigForm` from fields that live in the real `training_config`
 * (num_gpus_per_trial) and `tune_config` (max_concurrent_trials, num_samples,
 * time_budget_s) sections, and is never a top-level key of the template returned
 * by `getConfigurationTemplate()`. It therefore must NOT be filtered against the
 * template's real keys (`allSectionKeys`) — doing so drops the "General" tab and
 * lands the user straight on "Tuners", diverging from AutoTuneX which lists
 * "General" first and unconditionally in basic mode.
 */
export function computeSectionNames({ mode, trainingMode, presetGoal, allSectionKeys }: SectionNamesParams): string[] {
  if (!mode) {
    const basic = ['general_config']
    if (trainingMode === 'offline_tuning') {
      basic.push('tuners_config')
      if (presetGoal !== 'sft') basic.push('tuners_rl_config')
    } else {
      basic.push('tuners_rl_config')
    }
    // Keep the synthetic 'general_config' regardless of the template's real keys;
    // filter the remaining (real) sections so one the template omits stays hidden.
    return basic.filter((s) => s === 'general_config' || allSectionKeys.includes(s))
  }

  let advanced = [...allSectionKeys]
  if (trainingMode !== 'online_tuning') advanced = advanced.filter((k) => k !== 'training_rl_config')
  if (trainingMode === 'online_tuning') advanced = advanced.filter((k) => k !== 'tuners_config')
  if (presetGoal === 'sft') advanced = advanced.filter((k) => k !== 'tuners_rl_config')
  return advanced.sort((a, b) => {
    const ai = SECTION_ORDER.indexOf(a)
    const bi = SECTION_ORDER.indexOf(b)
    return (ai === -1 ? Infinity : ai) - (bi === -1 ? Infinity : bi)
  })
}

/**
 * The `config_data` an edited configuration is saved with.
 *
 * `PUT /configurations/{id}` is a full replacement, so this keeps every section
 * the form holds -- the form is seeded from the stored `config_data`, so a
 * section is never dropped just because no list here names it. Rebuilding the
 * payload from five hard-named sections deleted the stored `tokenizer_config`
 * on every edit. Only the form's own name/tuner fields are removed, plus an RL
 * section the form left empty, which the edit path has always omitted.
 */
export function configDataFromForm(form: ConfigForm): ConfigData {
  const { name: _name, tuner_type: _tunerType, rl_tuner_type: _rlTunerType, ...sections } = form
  if (!sections.training_rl_config) delete sections.training_rl_config
  if (!sections.tuners_rl_config) delete sections.tuners_rl_config
  return sections
}
