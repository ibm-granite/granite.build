'use client'

import { Fragment, useMemo, useRef, useState } from 'react'
import { Checkbox, ContentSwitcher, Dropdown, FormLabel, MultiSelect, NumberInput, Select, SelectItem, Switch, TextInput, Toggle } from '@carbon/react'
import type { Configuration, ConfigForm, TuningGoal } from '../types'
import { getOption, parseCommaList, toUpperCase } from '../lib/autotunex/wizardUtils'
import { computeSectionNames } from '../lib/autotunex/configSections'
import { formatValues, parseValuesInput } from '../lib/autotunex/hyperparamValues'
import { GeneralConfigForm } from './GeneralConfigForm'
import { TimeInput } from './TimeInput'
import styles from './CreateConfigForm.module.scss'
import layoutStyles from './layout.module.scss'

const SFT_ALGORITHMS = ['lora', 'sft', 'alora', 'lokr', 'loha', 'vera']
const RL_ALGORITHMS = ['dpo', 'kto', 'ppo', 'grpo', 'dapo']

const SECTION_LABELS: Record<string, string> = { tuners_rl_config: 'RL Tuners', training_rl_config: 'RL Training' }

/** One selectable candidate in a string hyperparameter's "Values" MultiSelect. */
interface ValueItem {
  id: string
  text: string
  disabled: boolean
}

function isObject(item: any): boolean {
  return item && typeof item === 'object' && !Array.isArray(item)
}

/** Hides fields explicitly marked `required: false`, or gated by a non-matching search_alg/scheduler selector. */
function shouldShowField(section: any, value: any): boolean {
  if (!isObject(value)) return true
  if (value.required === false) return false
  if (Array.isArray(value.search_alg)) {
    const current = section?.search_alg?.default
    if (!current || !value.search_alg.includes(current)) return false
  }
  if (Array.isArray(value.scheduler)) {
    const current = section?.scheduler?.default
    if (!current || !value.scheduler.includes(current)) return false
  }
  return true
}

interface CreateConfigFormProps {
  config: ConfigForm
  setConfig: (updater: (prev: ConfigForm) => ConfigForm) => void
  configurations: Configuration[]
  editMode?: boolean
  existingConfig?: Configuration | null
  hideNameField?: boolean
  presetGoal: TuningGoal | null
  presetAlgorithm?: string | null
  /** When false (HPO off), the search-space controls (Strategy/Values/Min/Max) are disabled — only Default is used. */
  hpoEnabled?: boolean
}

/**
 * Hyperparameter/config editor used by Step 2's create & edit flows.
 *
 * Note on scope vs. the source Svelte component: the source refetches its own
 * config template on mount even though its parent already fetched one before
 * rendering it (the parent's fetch result is bound in, then immediately
 * overwritten) — a redundant double-fetch. Here the parent (Step2Configure)
 * is the sole owner of fetching `config`; this component only derives UI
 * state from it and edits it in place via `setConfig`. The manual "Tuning
 * Mode" radio group from the source is also dropped: this wizard always
 * supplies a `presetGoal` (from Step 0), so that branch is unreachable here.
 */
export function CreateConfigForm({ config, setConfig, configurations, editMode = false, existingConfig, presetGoal, presetAlgorithm, hpoEnabled = true }: CreateConfigFormProps) {
  const trainingMode: 'offline_tuning' | 'online_tuning' = presetGoal === 'online_rl' ? 'online_tuning' : 'offline_tuning'
  const [mode, setMode] = useState(false) // false = Basic, true = Advanced

  const tuners = useMemo(() => (config.tuners_config ? Object.keys(config.tuners_config).sort() : []), []) // eslint-disable-line react-hooks/exhaustive-deps
  const rlTuners = useMemo(() => (config.tuners_rl_config ? Object.keys(config.tuners_rl_config).sort() : []), []) // eslint-disable-line react-hooks/exhaustive-deps

  const availableRlTuners = useMemo(
    () =>
      trainingMode === 'offline_tuning'
        ? ['none', ...rlTuners.filter((t) => ['dpo', 'kto'].includes(t))]
        : rlTuners.filter((t) => ['ppo', 'grpo', 'dapo'].includes(t)),
    [trainingMode, rlTuners]
  )

  const [selectedTuner, setSelectedTuner] = useState<string>(() => {
    if (editMode && existingConfig?.tuner_type) return existingConfig.tuner_type
    if (presetAlgorithm && SFT_ALGORITHMS.includes(presetAlgorithm) && tuners.includes(presetAlgorithm)) return presetAlgorithm
    if (trainingMode === 'offline_tuning') return tuners.includes('lora') ? 'lora' : tuners[0] || ''
    return ''
  })

  const [selectedRlTuner, setSelectedRlTuner] = useState<string>(() => {
    if (editMode && existingConfig?.rl_tuner_type) return existingConfig.rl_tuner_type
    if (presetAlgorithm && RL_ALGORITHMS.includes(presetAlgorithm) && rlTuners.includes(presetAlgorithm)) return presetAlgorithm
    if (presetGoal === 'sft' || trainingMode === 'offline_tuning') return 'none'
    return availableRlTuners[0] || rlTuners[0] || ''
  })

  // Stable set of top-level config sections present at mount (editing hyperparam
  // values never adds/removes a top-level section, so this only needs deriving once).
  const allSectionKeys = useMemo(
    () => Object.keys(config).filter((key) => key !== 'name' && key !== 'tuner_type' && key !== 'rl_tuner_type' && isObject((config as any)[key])),
    [] // eslint-disable-line react-hooks/exhaustive-deps
  )

  const sectionNames = useMemo(
    () => computeSectionNames({ mode, trainingMode, presetGoal, allSectionKeys }),
    [mode, trainingMode, presetGoal, allSectionKeys]
  )

  const [selectedSection, setSelectedSection] = useState<string>(() => sectionNames[0] ?? 'general_config')
  const selectedIndex = Math.max(0, sectionNames.indexOf(selectedSection))

  function handleSectionSwitch(index: number) {
    const name = sectionNames[index]
    if (name) setSelectedSection(name)
  }

  // Reflect the section list if it changed shape (e.g. Basic/Advanced toggle) and the
  // previously-selected section is no longer present.
  if (!sectionNames.includes(selectedSection) && sectionNames.length > 0) {
    setSelectedSection(sectionNames[0])
  }

  const [errorFields, setErrorFields] = useState<Record<string, { error: boolean; message: string }>>({})

  // Raw text the user is currently typing in a "Values" field, keyed by field id.
  // A "Values" input has to accept transient states that don't parse yet ("8,16,",
  // "0.0000"), so the text can't be re-derived from the committed number array on
  // every keystroke — doing so makes a controlled input snap back and the field
  // read-only. An entry lives here only while a field is being edited; on a
  // successful commit it is dropped so the canonical (sorted) values show again.
  const [valueDrafts, setValueDrafts] = useState<Record<string, string>>({})

  // Pristine copy of the template, mirroring the source form's `configCopy`.
  // A hyperparameter's MultiSelect needs the *full* original option list for its
  // items: using the live (current) values would make a deselected option vanish
  // from the menu permanently, so it could never be re-added.
  const pristineConfig = useRef<ConfigForm | null>(null)
  if (pristineConfig.current === null) pristineConfig.current = structuredClone(config)

  /**
   * Validate and commit a "Values" field, mirroring the source form's `on:change`
   * — which is the *native* change event, i.e. blur, not every keystroke.
   * On failure the draft text is kept so the user still sees what they typed
   * alongside the error, exactly as the Svelte version does.
   */
  function commitValues(fieldId: string, raw: string, paramConfig: any, update: (patch: Record<string, any>) => void) {
    const { values, error } = parseValuesInput(raw, paramConfig.min_val, paramConfig.max_val)

    setErrorFields((prev) => ({ ...prev, [fieldId]: { error, message: `Value must be between ${paramConfig.min_val} and ${paramConfig.max_val}` } }))
    if (error || !values) return

    update({ values })
    setValueDrafts((prev) => {
      const next = { ...prev }
      delete next[fieldId]
      return next
    })
  }

  function updateHyperparam(sectionKey: 'tuners_config' | 'tuners_rl_config', tunerKey: string, paramName: string, patch: Record<string, any>) {
    setConfig((prev) => {
      const section: any = (prev as any)[sectionKey] ?? {}
      const tuner = section[tunerKey]
      if (!tuner) return prev
      return {
        ...prev,
        [sectionKey]: {
          ...section,
          [tunerKey]: { ...tuner, hyperparams: { ...tuner.hyperparams, [paramName]: { ...tuner.hyperparams[paramName], ...patch } } },
        },
      } as ConfigForm
    })
  }

  function updateGenericField(sectionKey: string, fieldKey: string, patch: Record<string, any>) {
    setConfig((prev) => {
      const section: any = (prev as any)[sectionKey] ?? {}
      return { ...prev, [sectionKey]: { ...section, [fieldKey]: { ...section[fieldKey], ...patch } } } as ConfigForm
    })
  }

  function renderHyperparamField(sectionKey: 'tuners_config' | 'tuners_rl_config', tunerKey: string, paramName: string, paramConfig: any) {
    const update = (patch: Record<string, any>) => updateHyperparam(sectionKey, tunerKey, paramName, patch)
    const fieldId = `${sectionKey}-${tunerKey}-${paramName}`

    if (paramConfig.values && (paramConfig.type === 'int' || paramConfig.type === 'float')) {
      return (
        <div className={styles.hyperparamRow} key={paramName}>
          <div>
            <Select id={`${fieldId}-strategy`} labelText="Strategy" value={paramConfig.strategy} disabled={!hpoEnabled} onChange={(e) => update({ strategy: e.target.value })}>
              {paramConfig.options.map((option: string) => (
                <SelectItem key={option} value={option} text={getOption(option as any)} />
              ))}
            </Select>
          </div>
          <div>
            <NumberInput
              id={`${fieldId}-default`}
              label="Default"
              min={paramConfig.min_val}
              max={paramConfig.max_val}
              invalidText={`Value must be between ${paramConfig.min_val} and ${paramConfig.max_val}`}
              step={paramConfig.type === 'float' ? 0.01 : 1}
              value={paramConfig.default}
              onChange={(_e, { value }) => update({ default: typeof value === 'number' ? value : Number(value) })}
            />
          </div>
          {paramConfig.strategy === 'uniform' ? (
            <>
              <div>
                <NumberInput
                  id={`${fieldId}-min`}
                  label="Min value"
                  step={paramConfig.type === 'float' ? 0.01 : 1}
                  value={paramConfig.min_val}
                  disabled={!hpoEnabled}
                  onChange={(_e, { value }) => update({ min_val: typeof value === 'number' ? value : Number(value) })}
                />
              </div>
              <div>
                <NumberInput
                  id={`${fieldId}-max`}
                  label="Max value"
                  step={paramConfig.type === 'float' ? 0.01 : 1}
                  value={paramConfig.max_val}
                  disabled={!hpoEnabled}
                  onChange={(_e, { value }) => update({ max_val: typeof value === 'number' ? value : Number(value) })}
                />
              </div>
            </>
          ) : (
            <div className={styles.hyperparamValues}>
              <TextInput
                id={`${fieldId}-values`}
                labelText="Values"
                // Typing updates only the local draft; parsing/validation happens on blur
                // (the source form validates on the native change event). Validating per
                // keystroke and bailing out on a transient parse error would leave this
                // controlled input pinned to the last committed array — unable to be edited.
                value={valueDrafts[fieldId] ?? formatValues(paramConfig.values)}
                invalid={errorFields[fieldId]?.error}
                invalidText={errorFields[fieldId]?.message}
                disabled={!hpoEnabled}
                onChange={(e) => setValueDrafts((prev) => ({ ...prev, [fieldId]: e.target.value }))}
                onBlur={(e) => commitValues(fieldId, e.target.value, paramConfig, update)}
              />
            </div>
          )}
        </div>
      )
    }

    if (paramConfig.options?.length === 1 && paramConfig.strategy === 'string') {
      return (
        <div key={paramName}>
          <TextInput id={fieldId} labelText="" value={paramConfig.default} onChange={(e) => update({ default: e.target.value })} />
        </div>
      )
    }

    if (paramConfig.options?.length === 1 && paramConfig.type === 'str') {
      // Options come from the pristine template, not the live selection — otherwise
      // deselecting an option would remove it from the menu for good.
      const pristineParam = (pristineConfig.current as any)?.[sectionKey]?.[tunerKey]?.hyperparams?.[paramName]
      const allOptions: string[] = pristineParam?.values ?? paramConfig.values ?? []
      const selectedValues: string[] = paramConfig.values ?? []
      // Disable the last remaining selected option (when it is the default) so the
      // candidate list can never be emptied — the same per-item guard the source uses.
      const valueItems: ValueItem[] = allOptions.map((option) => ({
        id: option,
        text: option,
        disabled: selectedValues.length === 1 && option === paramConfig.default,
      }))
      return (
        <div className={styles.hyperparamRow} key={paramName}>
          <div>
            <Select id={`${fieldId}-strategy`} labelText="Strategy" value={paramConfig.strategy} disabled={!hpoEnabled} onChange={(e) => update({ strategy: e.target.value })}>
              {paramConfig.options.map((option: string) => (
                <SelectItem key={option} value={option} text={getOption(option as any)} />
              ))}
            </Select>
          </div>
          <div>
            <Select id={`${fieldId}-default`} labelText="Default" value={paramConfig.default} onChange={(e) => update({ default: e.target.value })}>
              {(paramConfig.values || []).map((option: string) => (
                <SelectItem key={option} value={option} text={option} />
              ))}
            </Select>
          </div>
          <div className={styles.hyperparamValues}>
            <MultiSelect
              id={`${fieldId}-values`}
              // `titleText` is the field label in Carbon React; `label` is only the
              // placeholder shown when nothing is selected (the source's Svelte
              // `labelText` maps to the former).
              titleText="Values"
              label=""
              items={valueItems}
              selectedItems={valueItems.filter((item) => selectedValues.includes(item.id))}
              itemToString={(item: ValueItem | null) => item?.text ?? ''}
              disabled={!hpoEnabled}
              onChange={({ selectedItems }) => {
                const next = (selectedItems ?? []).flatMap((item) => (item ? [item.id] : []))
                // Belt-and-braces alongside the per-item guard above: a hyperparameter
                // must always keep at least one candidate value.
                if (next.length === 0) return
                update({ values: next, default: next.includes(paramConfig.default) ? paramConfig.default : next[0] })
              }}
            />
          </div>
        </div>
      )
    }

    if (paramConfig.type === 'bool') {
      return (
        <div key={paramName}>
          <Checkbox id={fieldId} labelText={toUpperCase(paramName) ?? paramName} checked={!!paramConfig.default} onChange={(_e, { checked }) => update({ default: checked })} />
        </div>
      )
    }

    return null
  }

  function renderHyperparamSection(sectionKey: 'tuners_config' | 'tuners_rl_config', tunerKey: string) {
    const tuner = (config as any)[sectionKey]?.[tunerKey]
    if (!tuner) return null
    return (
      <div className={styles.hyperparamsSection}>
        <h5>{sectionKey === 'tuners_config' ? 'Hyperparameter search space settings' : 'RL Hyperparameter search space settings'}</h5>
        {!hpoEnabled && (
          <p className={styles.hpoDisabledNote}>HPO off — only Default values are used.</p>
        )}
        {Object.entries(tuner.hyperparams).map(([paramName, paramConfig]) => (
          <div className={styles.configItem} key={paramName}>
            <FormLabel className={styles.paramLabel}>
              <span style={{ fontSize: '14px', fontWeight: 600 }}>{toUpperCase((paramConfig as any).description) ?? (paramConfig as any).description}</span>
            </FormLabel>
            {renderHyperparamField(sectionKey, tunerKey, paramName, paramConfig)}
          </div>
        ))}
      </div>
    )
  }

  function renderGenericSection(sectionKey: string) {
    const section: any = (config as any)[sectionKey]
    if (!section) return null
    return (
      <div className={styles.genericSectionInset}>
        <div className={layoutStyles.fieldGrid}>
          {Object.entries(section).map(([key, value]: [string, any]) => {
            // bool-typed fields aren't editable via this generic section (the source has an
            // identical, unreachable Checkbox branch further down gated behind this same
            // exclusion — dropped here rather than porting dead code).
            if (value?.type === 'bool' || key === 'resource_name' || !shouldShowField(section, value)) return null
            if (!isObject(value)) {
              return (
                <TextInput
                  key={key}
                  id={`${sectionKey}-${key}`} labelText={toUpperCase(key) ?? key} value={String(section[key])} onChange={(e) => updateGenericField(sectionKey, key, { default: e.target.value })} />
              )
            }

            const fieldId = `${sectionKey}-${key}`
            let control: React.ReactNode
            if (key === 'max_concurrent_trials' && config.training_config?.num_gpus_per_trial) {
              const gpu = config.training_config.num_gpus_per_trial as any
              control = (
                <NumberInput
                  id={fieldId}
                  label={toUpperCase(key) ?? key}
                  helperText={value.description}
                  value={value.default}
                  min={value.min_val}
                  max={Math.floor(gpu.max_val / gpu.default)}
                  step={value.type === 'float' ? 0.01 : 1}
                  onChange={(_e, { value: v }) => updateGenericField(sectionKey, key, { default: typeof v === 'number' ? v : Number(v) })}
                />
              )
            } else if (key === 'num_gpus_per_trial') {
              control = (
                <NumberInput
                  id={fieldId}
                  label={toUpperCase(key) ?? key}
                  helperText={value.description}
                  value={value.default}
                  min={value.min_val}
                  max={value.max_val}
                  step={value.type === 'float' ? 0.01 : 1}
                  onChange={(_e, { value: v }) => {
                    const num = typeof v === 'number' ? v : Number(v)
                    updateGenericField(sectionKey, key, { default: num })
                    if (config.tune_config?.max_concurrent_trials) {
                      updateGenericField('tune_config', 'max_concurrent_trials', { default: Math.floor(value.max_val / num) })
                    }
                  }}
                />
              )
            } else if (key === 'time_budget_s') {
              control = <TimeInput label={key} value={value} onChange={(next) => updateGenericField(sectionKey, key, next)} />
            } else if (value.type === 'int' || value.type === 'float') {
              control = (
                <NumberInput
                  id={fieldId}
                  label={toUpperCase(key) ?? key}
                  helperText={value.description}
                  value={value.default}
                  min={value.min_val}
                  max={value.max_val}
                  step={value.type === 'float' ? 0.01 : 1}
                  onChange={(_e, { value: v }) => updateGenericField(sectionKey, key, { default: typeof v === 'number' ? v : Number(v) })}
                />
              )
            } else if (value.type === 'str' && value.values?.length > 0) {
              control = (
                <Select id={fieldId} labelText={toUpperCase(key) ?? key} helperText={value.description} value={value.default} onChange={(e) => updateGenericField(sectionKey, key, { default: e.target.value })}>
                  {value.values.map((option: string) => (
                    <SelectItem key={option} value={option} text={option} />
                  ))}
                </Select>
              )
            } else if (value.type === 'list') {
              control = (
                <TextInput
                  id={fieldId}
                  labelText={toUpperCase(key) ?? key}
                  helperText={`${value.description} (comma-separated)`}
                  placeholder="tok_a, tok_b, tok_c"
                  value={Array.isArray(value.default) ? value.default.join(', ') : value.default ?? ''}
                  onChange={(e) => updateGenericField(sectionKey, key, { default: e.target.value })}
                  onBlur={(e) => updateGenericField(sectionKey, key, { default: parseCommaList(e.target.value) })}
                />
              )
            } else {
              control = (
                <TextInput id={fieldId} labelText={toUpperCase(key) ?? key} helperText={value.description} placeholder={`Enter ${key}`} value={value.default ?? ''} onChange={(e) => updateGenericField(sectionKey, key, { default: e.target.value })} />
              )
            }

            return <Fragment key={key}>{control}</Fragment>
          })}
        </div>
      </div>
    )
  }

  // Carbon React's Dropdown renders `selectedItem` verbatim through
  // `itemToString`, so it has to be the *item object* from `items` — passing a
  // synthesised `{ id, text: selectedTuner }` displayed the raw key ("lora")
  // instead of the label. The source binds `selectedId`, which resolves the
  // item for you; these lookups are the React equivalent.
  const tunerItems = tuners.map((name) => ({
    id: name,
    text: `${(config.tuners_config as any)?.[name]?.description} (${toUpperCase(name)})`,
  }))
  const rlTunerItems = availableRlTuners.map((name) => ({
    id: name,
    text: name === 'none' ? 'No RL Algorithm (NONE)' : `${(config.tuners_rl_config as any)?.[name]?.description} (${toUpperCase(name)})`,
  }))

  const switcherItems = sectionNames.map((name) => ({ id: name, text: SECTION_LABELS[name] || toUpperCase(name.replace('_config', '')) || name }))

  return (
    <div>
      <header className={styles.header}>
        <div className={styles.modeRow}>
          <Toggle id="config-mode-toggle" labelA="Basic" labelB="Advanced" labelText="Configuration Mode" toggled={mode} onToggle={setMode} size="sm" />
        </div>
        <div className={styles.sectionSwitcherRow}>
          <ContentSwitcher selectedIndex={selectedIndex} onChange={({ index }) => handleSectionSwitch(index ?? 0)}>
            {switcherItems.map((item) => (
              <Switch key={item.id} name={item.id} text={item.text} />
            ))}
          </ContentSwitcher>
        </div>
      </header>

      <main className={selectedSection === 'tuners_config' || selectedSection === 'tuners_rl_config' ? styles.mainNoScroll : styles.main}>
        {selectedSection === 'general_config' ? (
          <GeneralConfigForm config={config as any} onConfigChange={setConfig as any} />
        ) : selectedSection === 'tuners_rl_config' && availableRlTuners.length > 0 ? (
          <div>
            <div style={{ marginBottom: '1rem' }}>
              <Dropdown
                id="rl-tuner-dropdown"
                label=""
                titleText="RL Algorithm type:"
                selectedItem={rlTunerItems.find((item) => item.id === selectedRlTuner) ?? null}
                items={rlTunerItems}
                itemToString={(item) => item?.text ?? ''}
                onChange={({ selectedItem }) => selectedItem && setSelectedRlTuner(selectedItem.id)}
              />
            </div>
            <div className={styles.configSection}>
              {selectedRlTuner === 'none' ? (
                <div className={styles.centeredHint}>
                  <p>No RL algorithm selected. Using default finetuning approach.</p>
                </div>
              ) : (
                renderHyperparamSection('tuners_rl_config', selectedRlTuner)
              )}
            </div>
          </div>
        ) : selectedSection === 'training_rl_config' && config.training_rl_config ? (
          <div className={styles.configSection}>{renderGenericSection('training_rl_config')}</div>
        ) : (
          <div>
            {selectedSection === 'tuners_config' && tuners.length > 0 && (
              <div style={{ marginBottom: '1rem' }}>
                <Dropdown
                  id="tuner-dropdown"
                  label=""
                  titleText="Tuner type:"
                  selectedItem={tunerItems.find((item) => item.id === selectedTuner) ?? null}
                  items={tunerItems}
                  itemToString={(item) => item?.text ?? ''}
                  onChange={({ selectedItem }) => selectedItem && setSelectedTuner(selectedItem.id)}
                />
              </div>
            )}
            <div className={styles.configSection}>
              {selectedSection === 'tuners_config' && (config.tuners_config as any)?.[selectedTuner]
                ? renderHyperparamSection('tuners_config', selectedTuner)
                : mode
                  ? renderGenericSection(selectedSection)
                  : null}
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
