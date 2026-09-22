/**
 * Regression test for the `type: list` field in the Step 2 config editor.
 *
 * The field committed through `parseCommaList`, which returns `string[]`. That is
 * right for `tokenizer_config` token lists but corrupts a numeric list: merely
 * tabbing through `tune_config.fidelity_schedule` (`default: [0.1, 0.25, 0.5]`)
 * rewrote it to `["0.1","0.25","0.5"]`, and the backend then raised at BLDS init --
 * `blds.py` does `any(p <= 0.0 or p > 1.0 for p in fidelity_schedule)`, a TypeError
 * between str and float.
 *
 * Usage: node --test tests/numeric-list-fields.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')

const { isNumericList, parseNumericCommaList } = require('../../../packages/ui-core/lib/autotunex/hyperparamValues.ts')

describe('numeric type: list fields', () => {
  const NUMERIC_TEMPLATE = [0.1, 0.25, 0.5]

  it('recognises a numeric template and keeps the list numeric', () => {
    assert.equal(isNumericList(NUMERIC_TEMPLATE), true)
    const parsed = parseNumericCommaList('0.1, 0.25, 0.5')
    assert.deepEqual(parsed, [0.1, 0.25, 0.5])
    for (const v of parsed) assert.equal(typeof v, 'number')
  })

  it('survives the no-op round-trip that used to corrupt the field', () => {
    // Tabbing through re-commits the rendered text without editing it.
    const rendered = NUMERIC_TEMPLATE.join(', ')
    assert.deepEqual(parseNumericCommaList(rendered), NUMERIC_TEMPLATE)
  })

  it('re-parses an already-committed array without stringifying it', () => {
    const once = parseNumericCommaList('0.1,0.25')
    assert.deepEqual(parseNumericCommaList(once), [0.1, 0.25])
  })

  it('does not claim a string template — tokenizer lists keep the string parser', () => {
    assert.equal(isNumericList(['<pad>', '<eos>']), false)
    // A tokenizer list whose entries happen to look numeric is still string-typed,
    // because the decision comes from the template and not from the typed text.
    assert.equal(isNumericList(['<pad>']), false)
  })

  it('drops entries that are not finite numbers rather than emitting NaN', () => {
    // NaN would serialise to null, and a mixed array fails the same way the
    // all-string one did.
    assert.deepEqual(parseNumericCommaList('0.1, abc, 0.5'), [0.1, 0.5])
  })

  it('returns null when nothing usable is left, matching parseCommaList', () => {
    assert.equal(parseNumericCommaList(''), null)
    assert.equal(parseNumericCommaList('abc'), null)
    assert.equal(parseNumericCommaList('   ,  '), null)
  })

  it('treats an absent, empty or mixed template as non-numeric', () => {
    // The call site then uses parseCommaList, i.e. the previous behaviour.
    assert.equal(isNumericList(undefined), false)
    assert.equal(isNumericList(null), false)
    assert.equal(isNumericList([]), false)
    assert.equal(isNumericList([0.1, '0.25']), false)
  })

  it('ignores trailing commas and surrounding whitespace', () => {
    assert.deepEqual(parseNumericCommaList(' 0.1 , 0.25 , '), [0.1, 0.25])
  })
})
