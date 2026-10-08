/**
 * The Reward Function step pre-fills test cases from verl rows, reading
 * `Array.isArray(row.prompt)` and `row.reward_model.ground_truth`. Datasets
 * exported to JSONL/CSV frequently store those fields as json.dumps'd STRINGS,
 * which makes both checks fail and leaves every test case blank.
 *
 * normalizeVerlRows() is the preprocessing pass that coerces those fields back
 * to native array/object before the (verl-strict) step reads them. These tests
 * reproduce the string-encoded failure and prove the fix, and guard the
 * idempotent / no-op / safe behaviours.
 *
 * Usage: node --test tests/verl-normalize.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')

const { normalizeVerlRow, normalizeVerlRows } = require('../../../packages/ui-core/lib/autotunex/verlNormalize.ts')

// The exact field access the step's buildTestCasesFromRows performs.
function stepSeesFilled(row) {
  const promptOk = Array.isArray(row.prompt) && row.prompt.length > 0
  const gt = row.reward_model?.ground_truth
  return { promptOk, groundTruth: gt != null ? String(gt) : '' }
}

const stringEncodedRow = {
  data_source: 'gsm8k',
  // json.dumps output — note the ", " / ": " spacing, stored as a STRING.
  prompt: '[{"role": "system", "content": "You are a helpful math tutor."}, {"role": "user", "content": "If a train travels 60 mph for 2.5 hours?"}]',
  ability: 'math',
  reward_model: '{"style": "rule", "ground_truth": "150 miles"}',
  extra_info: '{"split": "train", "index": "0"}',
}

describe('normalizeVerlRow — string-encoded verl fields', () => {
  it('reproduces the bug: raw string-encoded row fills nothing', () => {
    const before = stepSeesFilled(stringEncodedRow)
    assert.equal(before.promptOk, false, 'string prompt must not be an array')
    assert.equal(before.groundTruth, '', 'ground_truth must be empty on a string reward_model')
  })

  it('parses prompt / reward_model / extra_info into native structures', () => {
    const row = normalizeVerlRow(stringEncodedRow)
    assert.ok(Array.isArray(row.prompt), 'prompt is now an array')
    assert.equal(row.prompt.length, 2)
    assert.equal(row.prompt[0].role, 'system')
    assert.equal(typeof row.reward_model, 'object')
    assert.equal(row.reward_model.ground_truth, '150 miles')
    assert.equal(typeof row.extra_info, 'object')
    assert.equal(row.extra_info.split, 'train')
  })

  it('after normalization the step sees fully-populated fields', () => {
    const after = stepSeesFilled(normalizeVerlRow(stringEncodedRow))
    assert.equal(after.promptOk, true)
    assert.equal(after.groundTruth, '150 miles')
  })

  it('does not mutate the input row', () => {
    const input = { ...stringEncodedRow }
    normalizeVerlRow(input)
    assert.equal(typeof input.reward_model, 'string', 'original row is untouched')
  })
})

describe('normalizeVerlRow — idempotent / safe', () => {
  it('is a no-op on already-native verl rows (same reference back)', () => {
    const native = {
      data_source: 'gsm8k',
      prompt: [{ role: 'user', content: 'What is 2+2?' }],
      reward_model: { style: 'rule', ground_truth: '4' },
      extra_info: { split: 'train', index: 0 },
    }
    assert.equal(normalizeVerlRow(native), native, 'unchanged rows return same reference')
  })

  it('leaves a plain-text prompt (SFT/DPO) alone', () => {
    const dpo = { prompt: 'Explain quantum computing', chosen: 'a', rejected: 'b' }
    const out = normalizeVerlRow(dpo)
    assert.equal(out.prompt, 'Explain quantum computing')
    assert.equal(out, dpo, 'no verl JSON fields → same reference')
  })

  it('leaves an unparseable JSON-looking string as-is', () => {
    const bad = { reward_model: "{'style': 'rule'}" } // python repr, invalid JSON
    const out = normalizeVerlRow(bad)
    assert.equal(out.reward_model, "{'style': 'rule'}")
  })

  it('tolerates null/non-object input', () => {
    assert.equal(normalizeVerlRow(null), null)
    assert.equal(normalizeVerlRow(undefined), undefined)
  })
})

describe('normalizeVerlRows — array wrapper', () => {
  it('normalizes each row and fixes the whole preview', () => {
    const rows = [stringEncodedRow, stringEncodedRow]
    const out = normalizeVerlRows(rows)
    assert.equal(out.length, 2)
    for (const r of out) {
      const seen = stepSeesFilled(r)
      assert.equal(seen.promptOk, true)
      assert.equal(seen.groundTruth, '150 miles')
    }
  })

  it('returns same array reference when nothing changed', () => {
    const native = [{ prompt: [{ role: 'user', content: 'hi' }], reward_model: { ground_truth: '1' } }]
    assert.equal(normalizeVerlRows(native), native)
  })

  it('handles empty / nullish input', () => {
    assert.deepEqual(normalizeVerlRows([]), [])
    assert.deepEqual(normalizeVerlRows(null), [])
    assert.deepEqual(normalizeVerlRows(undefined), [])
  })
})
