import type { ColumnMapping } from '../../types'

/** The target vocabulary a suggestion is normalised against. */
export interface AiMappingTarget {
  /**
   * Every target column for the selected algorithm, required and optional --
   * i.e. `getColumnsFromTypes(algorithm, types).map(c => c.name)`. A suggested
   * target outside this list is dropped.
   */
  targetColumns: string[]
  /**
   * The dataset-types `columns` entry for the algorithm's type key: dict key
   * (`input_col`) -> column descriptor. This is the AI's own key vocabulary, so
   * it is tried before any string munging.
   */
  columnsDict: Record<string, { name?: string }>
}

/**
 * Translate an AI `column_mapping` into the wizard's `{target: source}` state.
 *
 * Pure and dependency-free so it can be unit-tested: there is no component-test
 * harness in this app, and this is the part of the AI flow that silently produces
 * a wrong projection rather than an error.
 *
 * Returns an empty `mapping` when every entry filters out. That is not the same
 * as failure and the caller must handle it -- in `Step1DatasetUpload` it means
 * falling back to the heuristic, because by then `setAiSuggestion` has already
 * fired and permanently disables the heuristic effect.
 */
export function aiMappingToColumnMapping(
  aiColumnMapping: Record<string, string>,
  availableColumns: string[],
  target: AiMappingTarget
): { mapping: ColumnMapping; suggestedFields: Set<string> } {
  const mapping: ColumnMapping = {}
  const suggestedFields = new Set<string>()

  const dictKeyToName: Record<string, string> = {}
  for (const [key, col] of Object.entries(target.columnsDict)) {
    if (col?.name) dictKeyToName[key] = col.name
  }

  for (const [aiKey, sourceColumn] of Object.entries(aiColumnMapping)) {
    // A source column the data does not have would pass the wizard's own
    // completeness gate and then project an absent column at import time.
    if (!sourceColumn || !availableColumns.includes(sourceColumn)) continue

    let matched = dictKeyToName[aiKey]
    if (!matched) {
      const normalized = aiKey.replace(/_col$/, '')
      matched =
        target.targetColumns.find(
          (candidate) =>
            candidate === aiKey || candidate === normalized || candidate === sourceColumn
        ) || ''
    }

    if (matched && target.targetColumns.includes(matched)) {
      mapping[matched] = sourceColumn
      suggestedFields.add(matched)
    }
  }

  return { mapping, suggestedFields }
}
