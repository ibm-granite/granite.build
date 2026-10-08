/**
 * Regression test for the model ComboBox on the Start Tuning wizard: editing the
 * model name that was already displayed used to wipe the whole field.
 *
 * Usage: node --test tests/model-combo-selection.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')

const { resolveModelComboItem } = require('../app/dashboard/autotunex/start-tuning/modelComboSelection.ts')

const DEFAULT_HF_MODEL = 'ibm-granite/granite-4.0-h-micro'

function asSuggestions(ids) {
  return ids.map((id) => ({ id, text: id }))
}

// What the wizard has on screen before the user touches anything.
const INITIAL_SUGGESTIONS = asSuggestions([
  DEFAULT_HF_MODEL,
  'ibm-granite/granite-4.0-h-micro-base',
  'ibm-granite/granite-4.0-h-micro-GGUF',
])

// What huggingface.co/api/models actually answers for the partial term
// 'ibm-granite/granite' — ten hits, none of them the exact model above.
const SEARCH_RESULTS = asSuggestions([
  'ibm-granite/granite-4.1-8b',
  'ibm-granite/granite-docling-258M',
  'ibm-granite/granite-4.1-3b',
  'ibm-granite/granite-speech-4.1-2b',
  'ibm-granite/granite-timeseries-ttm-r3',
  'ibm-granite/granite-4.0-h-tiny-GGUF',
  'ibm-granite/granite-4.0-1b',
  'ibm-granite/granite-vision-4.1-4b',
  'ibm-granite/granite-4.1-3b-GGUF',
  'ibm-granite/granite-embedding-97m-multilingual-r2',
])

const itemToString = (item) => item?.text ?? ''

/**
 * Models the parts of @carbon/react 1.108.0 that decide what the ComboBox text
 * field shows (node_modules/@carbon/react/lib/components/ComboBox/ComboBox.js):
 *
 *   - getInputValue (:54)          itemToString(selectedItem), or '' when null
 *   - selectedItem effect (:138)   every *reference* change of the selectedItem
 *                                  prop overwrites the field with getInputValue
 *                                  and fires onChange with that value
 *   - input onChange (:462)        typing only sets inputValue; with
 *                                  allowCustomValue=false it never touches the
 *                                  selection
 */
function mountCarbonComboBox(selectedItemProp) {
  let inputValue = selectedItemProp != null ? itemToString(selectedItemProp) : ''
  let prevSelectedItemProp = selectedItemProp

  return {
    type(text) {
      inputValue = text
    },
    rerender(nextSelectedItemProp) {
      if (prevSelectedItemProp !== nextSelectedItemProp) {
        const restored = nextSelectedItemProp != null ? itemToString(nextSelectedItemProp) : ''
        if (inputValue !== restored) inputValue = restored
        prevSelectedItemProp = nextSelectedItemProp
      }
    },
    get inputValue() {
      return inputValue
    },
  }
}

/** React's useMemo, near enough: recompute only when a dep changes by reference. */
function createMemo() {
  let deps = null
  let value
  return (nextDeps, compute) => {
    if (deps === null || nextDeps.some((dep, i) => dep !== deps[i])) {
      value = compute()
      deps = nextDeps
    }
    return value
  }
}

/**
 * Walks the reported reproduction: the default model is on screen, the user
 * backspaces part of it away, and 500ms later the HuggingFace search resolves
 * and replaces the suggestion list. Returns what the field then holds.
 */
function runEditScenario(deriveSelectedItem) {
  const selectedModel = DEFAULT_HF_MODEL
  let suggestions = INITIAL_SUGGESTIONS

  const combo = mountCarbonComboBox(deriveSelectedItem({ suggestions, selectedModel }))
  combo.type('ibm-granite/granite')
  suggestions = SEARCH_RESULTS
  combo.rerender(deriveSelectedItem({ suggestions, selectedModel }))

  return combo.inputValue
}

describe('model ComboBox selection (Start Tuning wizard)', () => {
  it('reproduces the wipe when selectedItem is derived from the suggestion list', () => {
    // The pre-fix expression, kept here so the harness above is shown to catch
    // the real defect rather than merely agreeing with the new code.
    const fromSuggestions = ({ suggestions, selectedModel }) =>
      suggestions.find((s) => s.id === selectedModel) ?? null

    assert.equal(runEditScenario(fromSuggestions), '', 'expected the old derivation to clear the field')
  })

  it('keeps what the user typed while the search results come back', () => {
    const memo = createMemo()
    const resolved = ({ selectedModel }) => memo([selectedModel], () => resolveModelComboItem(selectedModel))

    assert.equal(runEditScenario(resolved), 'ibm-granite/granite')
  })
})

describe('resolveModelComboItem', () => {
  it('never returns null while a model is selected', () => {
    assert.notEqual(resolveModelComboItem(DEFAULT_HF_MODEL), null)
  })

  it('returns null once the selection is cleared', () => {
    assert.equal(resolveModelComboItem(''), null)
  })

  it('shows the model by its id', () => {
    assert.deepEqual(resolveModelComboItem(DEFAULT_HF_MODEL), { id: DEFAULT_HF_MODEL, text: DEFAULT_HF_MODEL })
  })
})
