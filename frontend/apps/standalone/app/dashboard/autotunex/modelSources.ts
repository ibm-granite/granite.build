import type { ModelSource } from '@granite-build/ui-core/types'

export const MODEL_SOURCE_LABELS: Record<ModelSource, string> = {
  huggingface: 'Huggingface',
  custom_path: 'Local',
}

/** Render order of the model-source radios in Start Tuning step 1. */
export const MODEL_SOURCE_OPTIONS: { value: ModelSource; id: string, disabled?: boolean }[] = [
  { value: 'huggingface', id: 'model-source-hf', disabled: false },
  { value: 'custom_path', id: 'model-source-local', disabled: true },
]

/**
 * Sources that can no longer be selected but still appear on jobs already in the
 * database. Kept so their detail view reads the way it did when they were
 * launched, rather than falling back to the raw wire value.
 */
const RETIRED_MODEL_SOURCE_LABELS: Record<string, string> = {
  dmf: 'PVC',
}

/**
 * Label for a source read back from the API. Takes `string` rather than
 * `ModelSource` because job records come from the database, which can hold
 * values this build does not offer (retired sources, or a newer backend); those
 * fall back to the raw value instead of rendering blank.
 *
 * Indexing MODEL_SOURCE_LABELS with a ModelSource-typed value yields `string`,
 * so a `??` fallback at the call site would be dead code by the type system
 * while still being necessary at runtime. This does the widening in one place.
 */
export function modelSourceLabel(source: string): string {
  return MODEL_SOURCE_LABELS[source as ModelSource] ?? RETIRED_MODEL_SOURCE_LABELS[source] ?? source
}
