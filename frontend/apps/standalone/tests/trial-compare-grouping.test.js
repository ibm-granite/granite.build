/**
 * Tests for the trial-comparison grouping logic behind the labelled sections in
 * the Hyperparameters → Compare view.
 *
 * The compare view splits rows into three labelled blocks — Results,
 * Differences, and Similarities — and the section headings show a count for
 * each.
 * Those counts are only meaningful if the partition is exhaustive and
 * non-overlapping, so the split lives in a pure module rather than inline JSX
 * (the frontend test harness has no jsdom and cannot render Carbon components).
 *
 * getOddOnesOut drives the bold "differs from most trials" convention, and the
 * legend explaining it renders only when the map is non-empty — which is what
 * keeps the two-trial case (where no value can be in the minority) honest.
 *
 * Usage: node --test tests/trial-compare-grouping.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')

const {
  groupCompareKeys,
  findDifferingKeys,
  getOddOnesOut,
  labelForCompareKey,
} = require('../../../packages/ui-core/components/trialCompareGrouping.ts')

// Three flattened compare rows, shaped like toCompareRow() output: id +
// flattened config + rounded metrics. Ordered best-loss-first as the view does.
const THREE_TRIALS = [
  {
    id: '23042_00002',
    loss: 4.24333,
    train_loss: 4.23741,
    total_time: '21s',
    'training_config.learning_rate': 0.000005,
    'training_config.r': 16,
    'training_config.bias': 'none',
    'training_config.gradient_accumulation_steps': 1,
    'tune_config.experiments': ['exp-a'],
    'training_config.model_name_or_path': '/models/granite-3b',
    'training_config.notes': '',
  },
  {
    id: '23042_00001',
    loss: 4.59698,
    train_loss: 4.56085,
    total_time: '18s',
    'training_config.learning_rate': 0.000003,
    'training_config.r': 8,
    'training_config.bias': 'none',
    'training_config.gradient_accumulation_steps': 1,
    'tune_config.experiments': ['exp-a'],
    'training_config.model_name_or_path': '/models/granite-3b',
    'training_config.notes': '',
  },
  {
    id: '23042_00003',
    loss: 4.71184,
    train_loss: 4.73526,
    total_time: '21s',
    'training_config.learning_rate': 0.000003,
    'training_config.r': 8,
    'training_config.bias': 'none',
    'training_config.gradient_accumulation_steps': 1,
    'tune_config.experiments': ['exp-a'],
    'training_config.model_name_or_path': '/models/granite-3b',
    'training_config.notes': '',
  },
]

describe('groupCompareKeys', () => {
  it('puts a metric that varies across trials in resultKeys', () => {
    const { resultKeys } = groupCompareKeys(THREE_TRIALS)
    assert.deepEqual(resultKeys, ['loss', 'train_loss', 'total_time'])
  })

  it('puts a hyperparameter that varies across trials in differingKeys', () => {
    const { differingKeys } = groupCompareKeys(THREE_TRIALS)
    assert.deepEqual(differingKeys, ['training_config.learning_rate', 'training_config.r'])
  })

  it('puts a value identical across every trial in sameKeys', () => {
    const { sameKeys } = groupCompareKeys(THREE_TRIALS)
    assert.ok(sameKeys.includes('training_config.bias'))
    assert.ok(sameKeys.includes('training_config.gradient_accumulation_steps'))
  })

  it('treats an identical array value as identical rather than always-differing', () => {
    const { sameKeys, differingKeys } = groupCompareKeys(THREE_TRIALS)
    assert.ok(sameKeys.includes('tune_config.experiments'))
    assert.ok(!differingKeys.includes('tune_config.experiments'))
  })

  it('drops hidden keys, the id key, and keys empty in every trial', () => {
    const groups = groupCompareKeys(THREE_TRIALS)
    const visible = [...groups.resultKeys, ...groups.differingKeys, ...groups.sameKeys]
    assert.ok(!visible.includes('training_config.model_name_or_path'), 'hidden key leaked')
    assert.ok(!visible.includes('id'), 'id key leaked')
    assert.ok(!visible.includes('training_config.notes'), 'all-empty key leaked')
  })

  it('keeps a key that is empty in only some trials', () => {
    const rows = THREE_TRIALS.map((row, i) => ({ ...row, 'training_config.bias': i === 0 ? 'all' : '' }))
    const { differingKeys } = groupCompareKeys(rows)
    assert.ok(differingKeys.includes('training_config.bias'))
  })

  it('partitions every visible key exactly once', () => {
    const { resultKeys, differingKeys, sameKeys } = groupCompareKeys(THREE_TRIALS)
    const all = [...resultKeys, ...differingKeys, ...sameKeys]
    assert.equal(new Set(all).size, all.length, 'a key appears in more than one section')

    const expected = Object.keys(THREE_TRIALS[0]).filter(
      (k) => k !== 'id' && k !== 'training_config.model_name_or_path' && k !== 'training_config.notes'
    )
    assert.deepEqual([...all].sort(), [...expected].sort())
  })

  it('returns empty sections for no trials', () => {
    assert.deepEqual(groupCompareKeys([]), { resultKeys: [], differingKeys: [], sameKeys: [] })
  })
})

describe('findDifferingKeys', () => {
  it('reports only the keys whose value is not identical across rows', () => {
    const differing = findDifferingKeys([
      { a: 1, b: 'x' },
      { a: 2, b: 'x' },
    ])
    assert.deepEqual([...differing], ['a'])
  })
})

describe('getOddOnesOut', () => {
  it('flags the minority value when one trial of three differs', () => {
    const oddOnes = getOddOnesOut(THREE_TRIALS, ['training_config.r'])
    assert.deepEqual([...oddOnes['training_config.r']], ['16'])
  })

  it('finds no odd ones among two trials, where no value can be in the minority', () => {
    const oddOnes = getOddOnesOut(THREE_TRIALS.slice(0, 2), ['training_config.r'])
    assert.deepEqual(Object.keys(oddOnes), [])
  })
})


// Rows are built per-trial (toCompareRow spreads that trial's own config and
// metrics), so two trials in one comparison can genuinely carry different field
// sets — TrialsTable's radar code notes the same thing about `loss`. Both
// functions used to take their field list from rows[0] alone, so anything the
// first row lacked was invisible: not in Results, not in Differences, not in
// Similarities, and absent from the counts the section headings assert are
// exhaustive.
describe('rows with differing field sets', () => {
  it('includes a key that only later rows carry', () => {
    const rows = [
      { id: 't1', learning_rate: 0.001 },
      { id: 't2', learning_rate: 0.002, total_time: '4m' },
    ]
    const { resultKeys, differingKeys, sameKeys } = groupCompareKeys(rows)
    const all = [...resultKeys, ...differingKeys, ...sameKeys]
    assert.ok(all.includes('total_time'), `total_time missing from ${JSON.stringify(all)}`)
  })

  it('reports a differing key that the first row lacks', () => {
    const rows = [
      { id: 't1' },
      { id: 't2', weight_decay: 0.01 },
      { id: 't3', weight_decay: 0.05 },
    ]
    assert.ok(findDifferingKeys(rows).has('weight_decay'))
  })

  it('still partitions every visible key exactly once across the union', () => {
    const rows = [
      { id: 't1', a: 1 },
      { id: 't2', a: 2, b: 'x' },
      { id: 't3', c: true },
    ]
    const { resultKeys, differingKeys, sameKeys } = groupCompareKeys(rows)
    const all = [...resultKeys, ...differingKeys, ...sameKeys]
    assert.equal(new Set(all).size, all.length, 'a key landed in two sections')
    assert.deepEqual([...all].sort(), ['a', 'b', 'c'])
  })
})

// A trial names its primary metric dynamically (`trial.metric`). lossOf() honours
// that when sorting, but RESULT_KEYS is a fixed list — so a job reporting e.g.
// eval_loss sorted correctly and then filed that number under "Differences" as
// though it were a hyperparameter, while Results showed no loss at all. That is
// the same number TrialsTable prints in its Loss column.
describe('dynamic primary metric', () => {
  it('treats the trials own metric key as a result, not a hyperparameter', () => {
    const rows = [
      { id: 't1', eval_loss: 0.4, learning_rate: 0.001 },
      { id: 't2', eval_loss: 0.6, learning_rate: 0.002 },
    ]
    const { resultKeys, differingKeys } = groupCompareKeys(rows, ['eval_loss'])
    assert.deepEqual(resultKeys, ['eval_loss'])
    assert.deepEqual(differingKeys, ['learning_rate'])
  })

  it('leaves the fixed result keys working when no metric name is given', () => {
    const rows = [
      { id: 't1', loss: 0.4 },
      { id: 't2', loss: 0.6 },
    ]
    assert.deepEqual(groupCompareKeys(rows).resultKeys, ['loss'])
  })
})

describe('labelForCompareKey', () => {
  it('names the loss metric Eval loss, to distinguish it from Train loss', () => {
    // The backend emits `loss` alongside `train_loss` and sets metric="loss", so
    // the bare key really is the evaluation loss; "Loss" next to "Train loss" read
    // as though one of them were unqualified.
    assert.equal(labelForCompareKey('loss'), 'Eval loss')
  })

  it('falls back to humanising any other key', () => {
    assert.equal(labelForCompareKey('learning_rate'), 'Learning rate')
    assert.equal(labelForCompareKey('eval_loss'), 'Eval loss')
  })
})
