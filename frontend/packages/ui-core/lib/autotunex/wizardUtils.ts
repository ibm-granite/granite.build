import type {
  ColumnMapping,
  ColumnMetadata,
  DatasetFormatType,
  ModelSource,
  ParsedDataRow,
  TuningGoal,
} from '../../types'
import { ALGORITHM_DETAILS, ALGORITHM_OPTIONS, ALGORITHM_TO_DATASET_TYPE, DATASET_EXAMPLES } from '../../config/autotunexAlgorithms'

export function toUpperCase(text: string): string | undefined {
  if (!text) return undefined
  const cleaned = text.replaceAll('_', ' ').trim()
  return cleaned.charAt(0).toUpperCase() + cleaned.substring(1)
}

export function getOption(option: 'uniform' | 'loguniform' | 'choice'): string {
  if (option === 'uniform') return 'Uniform sampling'
  if (option === 'loguniform') return 'Logarithmic sampling'
  return toUpperCase(option) ?? option
}

export function parseCommaList(value: unknown): string[] | null {
  if (Array.isArray(value)) {
    const cleaned = value.map((v) => String(v).trim()).filter((v) => v.length > 0)
    return cleaned.length > 0 ? cleaned : null
  }
  if (typeof value !== 'string') return null
  const parts = value
    .split(',')
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
  return parts.length > 0 ? parts : null
}

// Mutates and returns configData: comma-separated tokenizer list fields need
// to be arrays before being sent to the backend.
export function normalizeTokenizerListFields(configData: any): any {
  const section = configData?.tokenizer_config
  if (!section || typeof section !== 'object') return configData
  for (const key of Object.keys(section)) {
    const field = section[key]
    if (field && typeof field === 'object' && field.type === 'list') {
      field.default = parseCommaList(field.default)
    }
  }
  return configData
}

/** Detect dataset format from column names — matches known RL/SFT dataset signatures. */
export function detectDatasetFormat(columns: string[]): DatasetFormatType {
  const cols = new Set(columns.map((c) => c.toLowerCase()))

  if (cols.has('prompt') && cols.has('chosen') && cols.has('rejected')) return 'preference_pairs'
  if (cols.has('prompt') && cols.has('completion') && cols.has('label')) return 'kto_format'

  const hasInput =
    cols.has('input') || cols.has('prompt') || cols.has('question') || cols.has('instruction') || cols.has('text')
  const hasOutput =
    cols.has('output') || cols.has('response') || cols.has('answer') || cols.has('completion') || cols.has('target') || cols.has('label')
  if (hasInput && hasOutput) return 'standard_pairs'

  if ((cols.has('prompt') || cols.has('input') || cols.has('question')) && !hasOutput) return 'prompt_only'

  return 'unknown'
}

/** Extract per-column type/sample/null/uniqueness metadata from parsed rows. */
export function extractColumnMetadata(rows: ParsedDataRow[], maxSampleRows = 100): ColumnMetadata[] {
  if (!rows || rows.length === 0) return []

  const sampleRows = rows.slice(0, maxSampleRows)
  const allColumns = new Set<string>()
  for (const row of sampleRows) {
    for (const key of Object.keys(row)) allColumns.add(key)
  }

  return Array.from(allColumns).map((colName) => {
    const values = sampleRows.map((row) => row[colName])
    const nonNullValues = values.filter((v) => v !== null && v !== undefined)
    const uniqueValues = new Set(nonNullValues.map((v) => JSON.stringify(v)))

    let detectedType: ColumnMetadata['detectedType'] = 'null'
    for (const val of nonNullValues) {
      if (Array.isArray(val)) detectedType = 'array'
      else if (typeof val === 'object') detectedType = 'object'
      else if (typeof val === 'number') detectedType = 'number'
      else if (typeof val === 'boolean') detectedType = 'boolean'
      else detectedType = 'string'
      break
    }

    const sampleValues = nonNullValues.slice(0, 3).map((v) => {
      const str = typeof v === 'string' ? v : JSON.stringify(v)
      return str.length > 80 ? str.substring(0, 80) + '...' : str
    })

    return {
      name: colName,
      detectedType,
      sampleValues,
      nullCount: values.length - nonNullValues.length,
      uniqueCount: uniqueValues.size,
    }
  })
}

export function getCompatibleMethods(format: DatasetFormatType): string[] {
  switch (format) {
    case 'preference_pairs':
      return ['DPO']
    case 'kto_format':
      return ['KTO']
    case 'standard_pairs':
      return ['SFT', 'LoRA', 'aLoRA', 'LoKR', 'LoHA', 'VeRA']
    case 'prompt_only':
      return ['PPO', 'GRPO', 'DAPO']
    default:
      return ['All methods']
  }
}

/** Normalize raw parsed rows for upload: standard pairs collapse to {input, output}; RL formats pass through. */
export function formatDatasetForUpload(rows: ParsedDataRow[], format: DatasetFormatType): any[] {
  if (format === 'standard_pairs') {
    return rows.map((row) => ({
      input: row.input ?? row.prompt ?? row.question ?? row.instruction ?? '',
      output: row.output ?? row.response ?? row.answer ?? row.completion ?? row.target ?? '',
    }))
  }
  return rows
}

export function getRequiredColumns(algorithmId: string): string[] {
  const detail = ALGORITHM_DETAILS.find((a) => a.id === algorithmId)
  if (detail) return detail.requiredColumns
  const algo = ALGORITHM_OPTIONS.find((a) => a.id === algorithmId)
  return algo?.requiredColumns || ['input', 'output']
}

export function getColumnsFromTypes(
  algorithmId: string,
  types: Record<string, any>
): { name: string; desc: string; required: boolean }[] {
  const typeKey = ALGORITHM_TO_DATASET_TYPE[algorithmId]
  if (!typeKey || !types[typeKey]) return []
  const columns = types[typeKey].columns || {}
  return Object.values(columns).map((col: any) => ({
    name: col.name as string,
    desc: (col.desc || '') as string,
    required: col.required !== false,
  }))
}

/** Required columns from the backend's dataset-types response, falling back to the hardcoded table. */
export function getRequiredColumnsFromTypes(algorithmId: string, types: Record<string, any>): string[] {
  const allCols = getColumnsFromTypes(algorithmId, types)
  if (allCols.length === 0) return getRequiredColumns(algorithmId)
  return allCols.filter((c) => c.required).map((c) => c.name)
}

export function getAlgorithmsForGoal(goal: TuningGoal) {
  return ALGORITHM_DETAILS.filter((a) => a.category === goal)
}

export function getDefaultAlgorithmForGoal(goal: TuningGoal): string {
  const recommended = ALGORITHM_DETAILS.find((a) => a.category === goal && a.recommended)
  return recommended?.id || 'lora'
}

export function getDatasetExamples(algorithmId: string): Record<string, string>[] {
  if (['lora', 'sft', 'alora', 'lokr', 'loha', 'vera'].includes(algorithmId)) return DATASET_EXAMPLES.sft
  if (['ppo', 'grpo', 'dapo'].includes(algorithmId)) return DATASET_EXAMPLES.online_rl
  return DATASET_EXAMPLES[algorithmId] || DATASET_EXAMPLES.sft
}

export function getDatasetExamplesFromTypes(
  algorithmId: string,
  types: Record<string, any>
): Record<string, string>[] {
  const typeKey = ALGORITHM_TO_DATASET_TYPE[algorithmId]
  if (!typeKey || !types[typeKey]) return getDatasetExamples(algorithmId)

  const columns = types[typeKey].columns || {}
  const exampleRow: Record<string, string> = {}
  for (const col of Object.values(columns) as any[]) {
    exampleRow[col.name] = col.desc || `<${col.name}>`
  }
  return [exampleRow]
}

export function generateFormatExamples(
  columns: { name: string; desc: string; type?: string }[]
): { jsonl: string; csv: string; json: string } {
  const cols = columns.map((col) => ({ name: col.name, placeholder: col.desc || `<${col.name}>` }))

  const jsonlRow = Object.fromEntries(cols.map((c) => [c.name, c.placeholder]))
  const jsonl = [jsonlRow, jsonlRow].map((r) => JSON.stringify(r)).join('\n')

  const header = cols.map((c) => c.name).join(', ')
  const csvRow = cols.map((c) => `"${c.placeholder}"`).join(', ')
  const csv = [header, csvRow, csvRow].join('\n')

  const json = JSON.stringify([jsonlRow, jsonlRow], null, 2)

  return { jsonl, csv, json }
}

/** Auto-suggest a column mapping: exact match first, then a small alias table. */
export function suggestColumnMapping(userColumns: string[], requiredColumns: string[]): ColumnMapping {
  const aliases: Record<string, string[]> = {
    prompt: ['prompt', 'question', 'instruction', 'input', 'text', 'query'],
    input: ['input', 'prompt', 'question', 'instruction', 'text', 'query'],
    output: ['output', 'response', 'answer', 'completion', 'target', 'label'],
    chosen: ['chosen', 'preferred', 'accepted', 'positive', 'output'],
    rejected: ['rejected', 'dispreferred', 'negative'],
    completion: ['completion', 'output', 'response', 'answer'],
    label: ['label', 'preference', 'rating', 'score'],
    documents: ['documents', 'document', 'context', 'source', 'reference'],
  }

  const mapping: ColumnMapping = {}
  const usedColumns = new Set<string>()
  const lowerUserColumns = userColumns.map((c) => c.toLowerCase())

  for (const required of requiredColumns) {
    const candidateAliases = aliases[required] || [required]
    let matched = false
    for (const alias of candidateAliases) {
      const idx = lowerUserColumns.findIndex((c, i) => c === alias && !usedColumns.has(userColumns[i]))
      if (idx !== -1) {
        mapping[required] = userColumns[idx]
        usedColumns.add(userColumns[idx])
        matched = true
        break
      }
    }
    if (!matched) mapping[required] = ''
  }

  return mapping
}

export function suggestAlgorithm(columns: string[]): string {
  switch (detectDatasetFormat(columns)) {
    case 'preference_pairs':
      return 'dpo'
    case 'kto_format':
      return 'kto'
    case 'standard_pairs':
      return 'lora'
    case 'prompt_only':
      return 'grpo'
    default:
      return 'lora'
  }
}

/** Warn (non-blocking) when the detected dataset format doesn't match the selected tuning goal. */
export function validateDatasetForGoal(
  detectedFormat: DatasetFormatType,
  goal: TuningGoal
): { valid: boolean; message: string } {
  const goalLabels: Record<TuningGoal, string> = {
    sft: 'Supervised Fine-Tuning',
    offline_rl: 'Preference Learning (Offline RL)',
    online_rl: 'Reinforcement Learning (Online RL)',
  }
  const formatLabels: Record<DatasetFormatType, string> = {
    preference_pairs: 'preference pairs (prompt + chosen + rejected)',
    kto_format: 'KTO format (prompt + completion + label)',
    standard_pairs: 'standard input/output pairs',
    prompt_only: 'prompt-only format',
    unknown: 'unknown format',
  }
  const expected: Record<TuningGoal, DatasetFormatType[]> = {
    sft: ['standard_pairs'],
    offline_rl: ['preference_pairs', 'kto_format'],
    online_rl: ['prompt_only'],
  }

  if (detectedFormat === 'unknown') return { valid: true, message: '' }
  if (expected[goal]?.includes(detectedFormat)) return { valid: true, message: '' }

  return {
    valid: false,
    message: `The uploaded dataset appears to be in ${formatLabels[detectedFormat]}, which is typically used for a different tuning approach. You selected "${goalLabels[goal]}". Please verify the column mapping below.`,
  }
}

export function getConfigSummary(config: { tuner_type?: string | null; rl_tuner_type?: string | null }): string {
  const displayNames: Record<string, string> = {
    lora: 'LoRA',
    alora: 'aLoRA',
    qlora: 'QLoRA',
    lokr: 'LoKR',
    loha: 'LoHA',
    vera: 'VeRA',
    sft: 'SFT',
  }
  const formatName = (name: string) => displayNames[name.toLowerCase()] ?? name.toUpperCase()
  const parts: string[] = []
  if (config.tuner_type && config.tuner_type !== 'none') parts.push(formatName(config.tuner_type))
  if (config.rl_tuner_type && config.rl_tuner_type !== 'none') parts.push(formatName(config.rl_tuner_type))
  return parts.join(' + ') || 'Default'
}

/** Rename columns from the user's names to the required names, dropping anything unmapped. */
export function applyColumnMapping(rows: ParsedDataRow[], mapping: ColumnMapping): ParsedDataRow[] {
  return rows.map((row) => {
    const mapped: ParsedDataRow = {}
    for (const [requiredCol, userCol] of Object.entries(mapping)) {
      if (userCol && row[userCol] !== undefined) mapped[requiredCol] = row[userCol]
    }
    return mapped
  })
}

/**
 * Like applyColumnMapping, but overlays the mapped canonical columns onto the
 * original row instead of replacing it — used wherever downstream code needs
 * extra columns applyColumnMapping would otherwise drop (e.g. the Reward
 * Function step's test-case generation needs `data_source`/`reward_model`/
 * `extra_info` alongside the mapped `prompt` column).
 */
export function overlayColumnMapping(rows: ParsedDataRow[], mapping: ColumnMapping): ParsedDataRow[] {
  return rows.map((row) => {
    const overlay: ParsedDataRow = { ...row }
    for (const [requiredCol, userCol] of Object.entries(mapping)) {
      if (userCol && row[userCol] !== undefined) overlay[requiredCol] = row[userCol]
    }
    return overlay
  })
}

/**
 * Whether the Step 0 model selection is complete enough to proceed.
 *
 * The "Local" source (`custom_path`) additionally requires an absolute path:
 * the value is handed to the tuning runner verbatim as --model_name_or_path, so
 * a relative path would resolve against the runner's working directory rather
 * than against anything the user had in mind.
 */
export function isModelSelectionValid(source: ModelSource, model: string): boolean {
  const trimmed = model.trim()
  if (!trimmed) return false
  if (source === 'custom_path') return trimmed.startsWith('/')
  return true
}
