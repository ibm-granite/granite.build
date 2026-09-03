export type ModelSuggestion = { id: string; text: string }

/**
 * Which item the wizard's model ComboBox should treat as selected.
 *
 * Deliberately independent of the suggestion list. Carbon's ComboBox
 * (@carbon/react 1.108.0, lib/components/ComboBox/ComboBox.js:138-155) reads
 * every *reference* change of its `selectedItem` prop as an instruction to
 * overwrite the text field — with `itemToString(selectedItem)`, or with an empty
 * string when the prop is null. Deriving the prop from `suggestions` therefore
 * wiped whatever the user was mid-way through typing as soon as a search
 * replaced the list, since HuggingFace results for a partial term rarely contain
 * the exact model id that was already selected.
 *
 * Memoize the result on `selectedModel` so the identity only moves when the
 * selection genuinely does.
 */
export function resolveModelComboItem(selectedModel: string): ModelSuggestion | null {
  if (!selectedModel) return null
  return { id: selectedModel, text: selectedModel }
}
