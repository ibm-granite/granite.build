'use client'

import { useEffect, useMemo, useRef, useState, type ComponentType } from 'react'
import {
  FormGroup,
  RadioButtonGroup,
  RadioButton,
  ComboBox,
  Modal,
  Button,
  Accordion,
  AccordionItem,
  Tile,
  Link,
  InlineLoading,
  TextInput,
} from '@carbon/react'
import {
  Education,
  Compare,
  Growth,
  View,
  Launch,
  CheckboxCheckedFilled,
  Checkbox as CheckboxOutline,
} from '@carbon/icons-react'
import ReactMarkdown from 'react-markdown'
import remarkBreaks from 'remark-breaks'
import type { HuggingFaceModel, ModelSource, TuningGoal } from '@granite-build/ui-core/types'
import { GOAL_OPTIONS } from '@granite-build/ui-core/config/autotunexAlgorithms'
import { getDefaultAlgorithmForGoal } from '@granite-build/ui-core/lib/autotunex/wizardUtils'
import { MODEL_SOURCE_LABELS, MODEL_SOURCE_OPTIONS } from '../../modelSources'
import { getHFModelCard, getHFModels } from '@granite-build/ui-core/api/autotunex'
import { resolveModelComboItem, type ModelSuggestion } from '../modelComboSelection'
import styles from './Step0GetStarted.module.scss'
import layoutStyles from '@granite-build/ui-core/components/layout.module.scss'

const GOAL_ICONS: Record<TuningGoal, ComponentType<{ size?: number }>> = {
  sft: Education,
  offline_rl: Compare,
  online_rl: Growth,
}

const GOAL_TILE_CLASS: Record<TuningGoal, string> = {
  sft: styles.tileSft,
  offline_rl: styles.tileOfflineRl,
  online_rl: styles.tileOnlineRl,
}

const RESOURCE_PANEL_CLASS: Record<TuningGoal, string> = {
  sft: styles.resourcePanelSft,
  offline_rl: styles.resourcePanelOfflineRl,
  online_rl: styles.resourcePanelOnlineRl,
}

/** Strips YAML front matter from a HuggingFace README so it doesn't render as visible text. */
function stripFrontMatter(raw: string): string {
  const lines = raw.split('\n')
  let inFrontMatter = false
  let contentStarted = false
  const out: string[] = []
  for (const line of lines) {
    if (line.trim() === '---') {
      if (!inFrontMatter) {
        inFrontMatter = true
      } else {
        inFrontMatter = false
        contentStarted = true
      }
      continue
    }
    if (inFrontMatter) continue
    if (contentStarted || line.trim() !== '') {
      contentStarted = true
      out.push(line)
    }
  }
  return out.join('\n').trim()
}

interface Step0GetStartedProps {
  selectedAlgorithm: string
  setSelectedAlgorithm: (v: string) => void
  selectedGoal: TuningGoal | null
  setSelectedGoal: (v: TuningGoal) => void
  selectedModel: string
  setSelectedModel: (v: string) => void
  modelSource: ModelSource
  setModelSource: (v: ModelSource) => void
  prefetchedModels: HuggingFaceModel[] | null
}

export function Step0GetStarted({
  selectedAlgorithm,
  setSelectedAlgorithm,
  selectedGoal,
  setSelectedGoal,
  selectedModel,
  setSelectedModel,
  modelSource,
  setModelSource,
  prefetchedModels,
}: Step0GetStartedProps) {
  const [models, setModels] = useState<HuggingFaceModel[]>([])
  const [suggestions, setSuggestions] = useState<ModelSuggestion[]>([])
  const [modelCard, setModelCard] = useState<string | null>(null)
  const [comboBoxReady, setComboBoxReady] = useState(false)
  const [showModelCardModal, setShowModelCardModal] = useState(false)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const previousModelSource = useRef(modelSource)

  // Must not be derived from `suggestions` — see resolveModelComboItem.
  const comboSelectedItem = useMemo(() => resolveModelComboItem(selectedModel), [selectedModel])

  function selectGoal(goal: TuningGoal) {
    setSelectedGoal(goal)
    setSelectedAlgorithm(getDefaultAlgorithmForGoal(goal))
  }

  async function fetchSuggestions(term: string) {
    if (!term.trim()) {
      setSuggestions(models.map((m) => ({ id: m.id, text: m.id })))
      return
    }
    try {
      const response = await getHFModels(term.replace(/(\w+)[-/]\1(?=[-/])/g, '$1'))
      setSuggestions(response.map((model) => ({ id: model.id, text: model.id })))
    } catch {
      setSuggestions([])
    }
  }

  async function fetchModelCard(modelId: string) {
    try {
      // Model cards are a HuggingFace-only concept — neither a registry entry
      // nor a filesystem path has one.
      if (modelSource !== 'huggingface' || !modelId) {
        setModelCard(null)
        return
      }
      const rawContent = await getHFModelCard(modelId)
      setModelCard(stripFrontMatter(rawContent))
    } catch {
      setModelCard(null)
    }
  }

  // Reset dependent state when the model source changes (HuggingFace <-> Local)
  useEffect(() => {
    if (modelSource === previousModelSource.current) return
    previousModelSource.current = modelSource

    if (modelSource === 'custom_path') {
      // No default and nothing to search — the user types a path. Clearing the
      // model also disables Next until they do (see isModelSelectionValid).
      setSelectedModel('')
      setSuggestions([])
      setModelCard(null)
    } else {
      setSelectedModel('ibm-granite/granite-4.0-h-micro')
      setSuggestions(models.map((m) => ({ id: m.id, text: m.id })))
      fetchModelCard('ibm-granite/granite-4.0-h-micro')
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelSource])

  // Initial mount: use prefetched models if available, else fetch + a 5s readiness fallback
  useEffect(() => {
    let alive = true

    function withSelectedModel(base: ModelSuggestion[]): ModelSuggestion[] {
      return selectedModel && !base.some((s) => s.id === selectedModel)
        ? [{ id: selectedModel, text: selectedModel }, ...base]
        : base
    }

    if (prefetchedModels && prefetchedModels.length > 0) {
      setModels(prefetchedModels)
      setSuggestions(withSelectedModel(prefetchedModels.map((m) => ({ id: m.id, text: m.id }))))
      setComboBoxReady(true)
      return
    }

    const fallbackTimeout = setTimeout(() => {
      if (!alive) return
      setSuggestions((prev) => (prev.length > 0 ? prev : withSelectedModel([])))
      setComboBoxReady(true)
    }, 5000)
    ;(async () => {
      try {
        const data = await getHFModels('ibm-granite/granite-4.0-h-micro', 20)
        if (!alive) return
        setModels(data)
        setSuggestions(withSelectedModel(data.map((m) => ({ id: m.id, text: m.id }))))
      } catch {
        if (alive) setSuggestions(withSelectedModel([]))
      }
      if (alive) setComboBoxReady(true)
      clearTimeout(fallbackTimeout)
    })()

    return () => {
      alive = false
      clearTimeout(fallbackTimeout)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  function handleComboBoxInputChange(inputValue: string) {
    clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => {
      if (inputValue) fetchSuggestions(inputValue)
    }, 500)
  }

  function handleComboBoxChange(selectedItem: ModelSuggestion | null | undefined) {
    if (!selectedItem?.id) {
      // Cleared
      setSelectedModel('')
      setSuggestions(models.map((m) => ({ id: m.id, text: m.id })))
      return
    }
    setSelectedModel(selectedItem.id)
    fetchModelCard(selectedItem.id)
  }

  return (
    <div>
      <div className={styles.intro}>
        <h4>Choose Your Fine-Tuning Approach</h4>
        <p className={styles.resourcePanelText}>Select what you want to achieve and choose a base model.</p>
      </div>

      <div className={styles.sectionLabel}>
        <span className={styles.sectionNumber}>1</span> What do you want to achieve?
      </div>
      <div className={layoutStyles.rowWrap} role="radiogroup" aria-label="Select tuning goal">
        {GOAL_OPTIONS.map((goal) => {
          const Icon = GOAL_ICONS[goal.id]
          const isSelected = selectedGoal === goal.id
          return (
            <div className={styles.goalTileColumn} key={goal.id}>
              {/* Radio semantics: exactly one goal is always selected. Re-clicking the
                  selected tile is a no-op (never deselects). A plain Tile — not
                  SelectableTile — keeps selection fully controlled by `selectedGoal`. */}
              <Tile
                role="radio"
                aria-checked={isSelected}
                aria-label={goal.title}
                tabIndex={0}
                onClick={() => selectGoal(goal.id)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault()
                    selectGoal(goal.id)
                  }
                }}
                className={`${styles.goalSelectableTile} ${GOAL_TILE_CLASS[goal.id]}`}
                style={{ minHeight: 120, padding: '1.25rem' }}
              >
                <span className={styles.goalCheckmark} aria-hidden="true">
                  {isSelected ? <CheckboxCheckedFilled size={20} /> : <CheckboxOutline size={20} />}
                </span>
                <div className={styles.goalTile}>
                  <div className={`${styles.goalIcon} ${isSelected ? 'active' : ''}`}>
                    <Icon size={32} />
                  </div>
                  <div className={styles.goalTileContent}>
                    <h6 className={styles.goalHeading}>{goal.title}</h6>
                    <p className={styles.goalSubtitle}>{goal.sub_title}</p>
                    <p className={styles.goalHelperText}>{goal.description}</p>
                    <div className={styles.goalFooter}>
                      <span className={styles.goalFooterLabel}>Data:</span>
                      <span>{goal.dataDescription}</span>
                    </div>
                  </div>
                </div>
              </Tile>
            </div>
          )
        })}
      </div>

      {selectedGoal && (
        <>
          <div className={styles.sectionRow}>
            <ResourcePanel goal={selectedGoal} />
          </div>

          <div className={styles.sectionRow}>
            <div className={styles.sectionLabel}>
              <span className={styles.sectionNumber}>2</span> Select a base model
            </div>
          </div>
          <div className={styles.modelSection}>
            <FormGroup legendText="">
                <RadioButtonGroup
                  legendText="Model source"
                  name="model_source_wizard"
                  valueSelected={modelSource}
                  onChange={(value) => setModelSource(value as ModelSource)}
                >
                  {MODEL_SOURCE_OPTIONS.map((o) => (
                    <RadioButton key={o.value} labelText={MODEL_SOURCE_LABELS[o.value]} value={o.value} id={o.id}  disabled={o.disabled}/>
                  ))}
                </RadioButtonGroup>
              </FormGroup>

              <FormGroup legendText="">
                <div className={styles.modelComboRow}>
                  <div className={styles.modelComboField}>
                    {modelSource === 'custom_path' ? (
                      <TextInput
                        id="model-path"
                        labelText="Model path"
                        placeholder="/mnt/models/granite-4.0-micro"
                        helperText="Absolute path to a model directory on the tuning host."
                        value={selectedModel}
                        onChange={(e) => setSelectedModel(e.target.value)}
                        invalid={selectedModel.trim() !== '' && !selectedModel.trim().startsWith('/')}
                        invalidText="Must be an absolute path"
                      />
                    ) : comboBoxReady ? (
                      <ComboBox
                        id="model-combo"
                        titleText="Model"
                        placeholder="Search for a model..."
                        items={suggestions}
                        itemToString={(item) => item?.text ?? ''}
                        selectedItem={comboSelectedItem}
                        shouldFilterItem={() => true}
                        onInputChange={handleComboBoxInputChange}
                        onChange={({ selectedItem }) => handleComboBoxChange(selectedItem)}
                      />
                    ) : (
                      <InlineLoading description="Loading models..." />
                    )}
                  </div>
                  {selectedModel && modelSource === 'huggingface' && (
                    <Button
                      kind="ghost"
                      size="md"
                      renderIcon={View}
                      iconDescription="View model details"
                      hasIconOnly={false}
                      onClick={() => {
                        if (!modelCard) fetchModelCard(selectedModel)
                        setShowModelCardModal(true)
                      }}
                    >
                      View details
                    </Button>
                  )}
                </div>
              </FormGroup>
            </div>
        </>
      )}

      <Modal
        open={showModelCardModal}
        onRequestClose={() => setShowModelCardModal(false)}
        modalHeading="Model Card"
        passiveModal
        size="lg"
      >
        {modelCard ? (
          <div className={styles.modelCardContent}>
            <ReactMarkdown remarkPlugins={[remarkBreaks]}>{modelCard}</ReactMarkdown>
          </div>
        ) : (
          <InlineLoading description="Loading model card..." />
        )}
      </Modal>
    </div>
  )
}

function ResourcePanel({ goal }: { goal: TuningGoal }) {
  if (goal === 'sft') {
    return (
      <Tile className={`${styles.resourcePanel} ${RESOURCE_PANEL_CLASS.sft}`}>
        <p className={styles.resourcePanelIntro}>
          Model Customization uses <strong>PEFT</strong> (Parameter-Efficient Fine-Tuning) by HuggingFace under the hood for
          all SFT training. Understanding PEFT&apos;s adapters and data format will help you prepare your datasets.
        </p>
        <Accordion size="sm">
          <AccordionItem title="What is PEFT?" open>
            <p className={styles.resourcePanelText}>
              A library for efficiently adapting large language models to downstream tasks by training only a
              small number of extra parameters. It supports methods like LoRA, QLoRA, Prefix Tuning, P-Tuning,
              Prompt Tuning, IA3, and more.
            </p>
          </AccordionItem>
          <AccordionItem title="Data Format">
            <p className={styles.resourcePanelText}>
              PEFT expects standard instruction-following datasets in <strong>JSON</strong>, <strong>JSONL</strong>,
              or <strong>CSV</strong> format with input/output pairs:
            </p>
            <div className={styles.dataExample}>{'{\n  "input": "Summarize the following article: ...",\n  "output": "The article discusses ..."\n}'}</div>
          </AccordionItem>
          <AccordionItem title="Documentation & Resources">
            <div className={styles.resourceLinks}>
              <Link renderIcon={Launch} href="https://huggingface.co/docs/peft" target="_blank">PEFT Documentation</Link>
              <Link renderIcon={Launch} href="https://huggingface.co/docs/peft/quicktour" target="_blank">Quickstart Guide</Link>
              <Link renderIcon={Launch} href="https://huggingface.co/docs/peft/conceptual_guides/lora" target="_blank">LoRA Conceptual Guide</Link>
              <Link renderIcon={Launch} href="https://github.com/huggingface/peft" target="_blank">GitHub Repository</Link>
            </div>
          </AccordionItem>
        </Accordion>
      </Tile>
    )
  }

  if (goal === 'offline_rl') {
    return (
      <Tile className={`${styles.resourcePanel} ${RESOURCE_PANEL_CLASS.offline_rl}`}>
        <p className={styles.resourcePanelIntro}>
          Model Customization uses <strong>TRL</strong> (Transformer Reinforcement Learning) by HuggingFace under the hood
          for all Offline RL training.
        </p>
        <Accordion size="sm">
          <AccordionItem title="What is TRL?" open>
            <p className={styles.resourcePanelText}>
              A library for training language models with reinforcement learning techniques, including Direct
              Preference Optimization (DPO) and Kahneman-Tversky Optimization (KTO).
            </p>
          </AccordionItem>
          <AccordionItem title="Data Format">
            <p className={styles.resourcePanelText}>
              TRL expects preference datasets with chosen/rejected response pairs in <strong>JSON</strong> or{' '}
              <strong>JSONL</strong> format:
            </p>
            <div className={styles.dataExample}>{'{\n  "prompt": "Explain quantum computing in simple terms.",\n  "chosen": "Quantum computing uses quantum bits ...",\n  "rejected": "Quantum computing is a type of ..."\n}'}</div>
          </AccordionItem>
          <AccordionItem title="Documentation & Resources">
            <div className={styles.resourceLinks}>
              <Link renderIcon={Launch} href="https://huggingface.co/docs/trl" target="_blank">TRL Documentation</Link>
              <Link renderIcon={Launch} href="https://huggingface.co/docs/trl/dpo_trainer" target="_blank">DPO Trainer Guide</Link>
              <Link renderIcon={Launch} href="https://huggingface.co/docs/trl/dataset_formats" target="_blank">Dataset Format Guide</Link>
              <Link renderIcon={Launch} href="https://github.com/huggingface/trl" target="_blank">GitHub Repository</Link>
            </div>
          </AccordionItem>
        </Accordion>
      </Tile>
    )
  }

  return (
    <Tile className={`${styles.resourcePanel} ${RESOURCE_PANEL_CLASS.online_rl}`}>
      <p className={styles.resourcePanelIntro}>
        Model Customization uses <strong>VERL</strong> (Volcano Engine RL) under the hood for all Online RL training.
      </p>
      <Accordion size="sm">
        <AccordionItem title="What is VERL?" open>
          <p className={styles.resourcePanelText}>
            A flexible, efficient, production-ready reinforcement learning training library for large language
            models, supporting a range of algorithms and distributed backends.
          </p>
        </AccordionItem>
        <AccordionItem title="Data Format">
          <p className={styles.resourcePanelText}>
            VERL expects datasets in <strong>Parquet</strong> format with the following fields:
          </p>
          <div className={styles.dataExample}>{'{\n  "data_source": "gsm8k",\n  "prompt": [{ "role": "user", "content": "What is 2 + 2?" }],\n  "ability": "math",\n  "reward_model": { "style": "rule", "ground_truth": "4" },\n  "extra_info": { "split": "train", "index": 0 }\n}'}</div>
        </AccordionItem>
        <AccordionItem title="Documentation & Resources">
          <div className={styles.resourceLinks}>
            <Link renderIcon={Launch} href="https://verl.readthedocs.io/en/latest/index.html" target="_blank">VERL Documentation</Link>
            <Link renderIcon={Launch} href="https://verl.readthedocs.io/en/latest/preparation/prepare_data.html" target="_blank">Data Preparation</Link>
            <Link renderIcon={Launch} href="https://verl.readthedocs.io/en/latest/preparation/reward_function.html" target="_blank">Reward Functions</Link>
            <Link renderIcon={Launch} href="https://github.com/verl-project/verl" target="_blank">GitHub Repository</Link>
          </div>
        </AccordionItem>
      </Accordion>
    </Tile>
  )
}
