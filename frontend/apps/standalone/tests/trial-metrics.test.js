/**
 * Tests for the shaping behind the Hyperparameters tab metric charts.
 *
 * Two things here are easy to get wrong and expensive to get wrong:
 *
 * 1. GET /jobs/{id}/metrics returns three different kinds of row in one stream,
 *    and a null `loss` means "not a step row", not "value missing". Reading
 *    `loss` straight off the array puts a hole in the line at every eval and
 *    every end-of-run summary.
 *
 * 2. A job contains two incomparable phases — the HPO search trials, then one
 *    final run on the winning config over the full data set. The rows carry no
 *    phase marker (we asked for one; it was deferred), so the phase is derived
 *    from the fact that the final run is absent from GET /jobs/{id}/trials.
 *    The dangerous case is the trials query not having resolved yet: the id set
 *    is then empty and the whole job would misclassify as one final run.
 *
 * The fixture mirrors the real shape and per-kind counts of reference job
 * b32d2a29-9af1-4c95-b189-dd1940128d2f: four search trials of 34 step rows +
 * 3 evals + 1 summary, then a final run of 23 step rows + 5 evals + 1 summary.
 * That is 181 rows splitting 159 / 17 / 5.
 *
 * Usage: node --test tests/trial-metrics.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')

const {
  splitMetricRows,
  derivePhases,
  trialColorScale,
  bestTrialId,
  toChartRows,
  emaChartRows,
  METRIC_PALETTE,
  METRIC_DE_EMPHASIS,
  EMPHASIS_THRESHOLD,
} = require('../../../packages/ui-core/components/trialMetrics.ts')

const SEARCH_IDS = ['491c7_00000', '491c7_00001', '491c7_00002', '491c7_00003']
const FINAL_ID = '11517_00000'

let nextId = 245 // the reference job's first metric id

function stepRow(trialId, step, epoch) {
  return {
    id: nextId++,
    trial_id: trialId,
    global_step: step,
    epoch,
    loss: 15 + Math.sin(step) * 0.5,
    grad_norm: 2.8 + (step % 5) * 0.1,
    learning_rate: 1e-6 * (1 + (step % 7)),
    split: 'train',
    extra: {},
  }
}

function evalRow(trialId, step, epoch, evalLoss) {
  return {
    id: nextId++,
    trial_id: trialId,
    global_step: step,
    epoch,
    loss: null,
    grad_norm: null,
    learning_rate: null,
    split: 'eval',
    extra: { eval_loss: evalLoss, eval_runtime: 4.1, eval_samples_per_second: 4.8 },
  }
}

function summaryRow(trialId, step, trainLoss) {
  return {
    id: nextId++,
    trial_id: trialId,
    global_step: step,
    epoch: 3,
    loss: null,
    grad_norm: null,
    learning_rate: null,
    split: 'train',
    extra: { train_loss: trainLoss, train_runtime: 457.1, total_flos: 1.83e14 },
  }
}

/** One run: `steps` step rows, `evalCount` evals, one end-of-run summary. */
function run(trialId, steps, evalCount, epochs) {
  const rows = []
  for (let i = 1; i <= steps; i++) rows.push(stepRow(trialId, i * 2, (i / steps) * epochs))
  for (let e = 1; e <= evalCount; e++) {
    rows.push(evalRow(trialId, Math.round((e / evalCount) * steps * 2), e, 15.2 - e * 0.01))
  }
  rows.push(summaryRow(trialId, steps * 2, 15.1))
  return rows
}

function fixture() {
  nextId = 245
  const rows = []
  for (const id of SEARCH_IDS) rows.push(...run(id, 34, 3, 3))
  rows.push(...run(FINAL_ID, 23, 5, 1))
  return rows
}

function trial(id, loss) {
  return {
    id,
    job_id: 'b32d2a29',
    status: 'completed',
    config: {},
    metric: 'loss',
    metrics: { loss },
    created_at: '2026-09-03T12:16:30Z',
    updated_at: '2026-09-03T12:16:30Z',
  }
}

describe('splitMetricRows', () => {
  it('sorts the reference job into 159 step / 17 eval / 5 summary rows', () => {
    const { trainSteps, evals, summaries } = splitMetricRows(fixture())
    assert.equal(trainSteps.length, 159)
    assert.equal(evals.length, 17)
    assert.equal(summaries.length, 5)
    assert.equal(trainSteps.length + evals.length + summaries.length, 181)
  })

  it('keeps every null-loss row out of trainSteps', () => {
    // The whole point: a hole in the loss line is the bug this prevents.
    const { trainSteps } = splitMetricRows(fixture())
    assert.ok(trainSteps.every((r) => typeof r.loss === 'number'))
  })

  it('routes eval rows by split, not by null-ness, and keeps eval_loss reachable', () => {
    const { evals, summaries } = splitMetricRows(fixture())
    assert.ok(evals.every((r) => r.split === 'eval' && typeof r.extra.eval_loss === 'number'))
    // Summaries are split='train' with a null loss — they must not land in evals.
    assert.ok(summaries.every((r) => r.split === 'train' && r.loss === null))
    assert.equal(summaries.length, 5, 'one summary per run')
  })
})

describe('derivePhases', () => {
  it('attributes the run missing from /trials to the final phase', () => {
    const { search, final, finalTrialIds } = derivePhases(fixture(), SEARCH_IDS, true)
    assert.deepEqual(finalTrialIds, [FINAL_ID])
    assert.equal(final.length, 29, '23 steps + 5 evals + 1 summary')
    assert.equal(search.length, 152, '4 search trials of 38 rows')
    assert.ok(final.every((r) => r.trial_id === FINAL_ID))
    assert.ok(search.every((r) => SEARCH_IDS.includes(r.trial_id)))
  })

  it('treats everything as search until the trials query resolves', () => {
    // The regression guard: with trialsLoaded=false the id set is empty, so a
    // naive implementation would call the entire job one giant final run and
    // draw the search trials on the final-run scale.
    const rows = fixture()
    const { search, final, finalTrialIds } = derivePhases(rows, [], false)
    assert.equal(search.length, rows.length)
    assert.equal(final.length, 0)
    assert.deepEqual(finalTrialIds, [])
  })

  it('does not relabel a row that carries no trial id', () => {
    const rows = [...fixture(), { ...stepRow(null, 2, 0.1), trial_id: null }]
    const { final, search } = derivePhases(rows, SEARCH_IDS, true)
    assert.ok(final.every((r) => r.trial_id === FINAL_ID))
    assert.equal(search.filter((r) => r.trial_id == null).length, 1)
  })

  it('groups several unrecognised run ids under the final phase', () => {
    const rows = [...fixture(), stepRow('99999_00000', 2, 0.1)]
    const { finalTrialIds } = derivePhases(rows, SEARCH_IDS, true)
    assert.deepEqual(finalTrialIds, [FINAL_ID, '99999_00000'])
  })
})

describe('trialColorScale', () => {
  it('gives each run a distinct palette slot at or below the threshold', () => {
    const scale = trialColorScale(SEARCH_IDS, '491c7_00002', 'white')
    const used = Object.values(scale)
    assert.equal(new Set(used).size, SEARCH_IDS.length, 'no two runs share a hue')
    assert.equal(scale[SEARCH_IDS[0]], METRIC_PALETTE.white[0])
  })

  it('keeps a run’s colour when another run is hidden', () => {
    // Colour must follow the run, not its position among the *visible* ones —
    // otherwise hiding a series repaints the survivors and a reader who learned
    // "_00002 is purple" is misled.
    const all = trialColorScale(SEARCH_IDS, '491c7_00002', 'white')
    const stillOrdered = trialColorScale(SEARCH_IDS, '491c7_00002', 'white')
    assert.equal(stillOrdered['491c7_00003'], all['491c7_00003'])
    // And the scale is keyed by id, so a caller filtering its *data* down to two
    // series reuses the same map untouched.
    assert.equal(all['491c7_00002'], METRIC_PALETTE.white[2])
  })

  it('switches to emphasis past the threshold instead of inventing hues', () => {
    const many = Array.from({ length: EMPHASIS_THRESHOLD + 3 }, (_, i) => `t_${i}`)
    const scale = trialColorScale(many, 't_4', 'white')
    assert.equal(scale['t_4'], METRIC_PALETTE.white[0], 'best run takes the accent')
    const others = many.filter((id) => id !== 't_4').map((id) => scale[id])
    assert.ok(
      others.every((c) => c === METRIC_DE_EMPHASIS.white),
      'every other run falls back to the de-emphasis grey'
    )
  })

  it('uses selected dark steps, not the light ones', () => {
    const light = trialColorScale(SEARCH_IDS, undefined, 'white')
    const dark = trialColorScale(SEARCH_IDS, undefined, 'g100')
    // purple-70 and green-60 are too dark on Carbon's g100 layer.
    assert.notEqual(light['491c7_00002'], dark['491c7_00002'])
    assert.notEqual(light['491c7_00003'], dark['491c7_00003'])
    assert.equal(dark['491c7_00002'], '#a56eff')
  })
})

describe('bestTrialId', () => {
  it('picks the lowest reported metric', () => {
    const trials = [trial('a', 15.24), trial('b', 15.16), trial('c', 15.21)]
    assert.equal(bestTrialId(trials), 'b')
  })

  it('ignores runs with no usable metric', () => {
    const noMetric = { ...trial('a', 0), metric: undefined, metrics: {} }
    const nan = { ...trial('b', Number.NaN) }
    assert.equal(bestTrialId([noMetric, nan, trial('c', 15.2)]), 'c')
    assert.equal(bestTrialId([]), undefined)
  })
})

describe('toChartRows', () => {
  it('drops points with no value and keys on the requested x', () => {
    const rows = fixture()
    const byStep = toChartRows(rows, 'global_step', (r) => r.loss)
    assert.equal(byStep.length, 159, 'only step rows carry a loss')
    const byEpoch = toChartRows(rows, 'epoch', (r) => r.loss)
    assert.equal(byEpoch.length, 159)
    assert.ok(byStep.every((r) => Number.isFinite(r.key) && Number.isFinite(r.value)))
  })

  it('reads eval loss out of extra', () => {
    const evalRows = splitMetricRows(fixture()).evals
    const rows = toChartRows(evalRows, 'global_step', (r) => r.extra?.eval_loss)
    assert.equal(rows.length, 17)
  })

  it('orders each group by x so interleaved runs still draw monotonically', () => {
    const rows = toChartRows(
      [stepRow('b', 4, 0.2), stepRow('a', 6, 0.3), stepRow('a', 2, 0.1)],
      'global_step',
      (r) => r.loss
    )
    const groupA = rows.filter((r) => r.group === 'a').map((r) => r.key)
    assert.deepEqual(groupA, [2, 6])
  })
})

describe('emaChartRows', () => {
  it('leaves each group’s first point untouched', () => {
    const raw = [
      { group: 'a', key: 1, value: 10 },
      { group: 'a', key: 2, value: 20 },
      { group: 'b', key: 1, value: 100 },
    ]
    const smoothed = emaChartRows(raw, 0.5)
    assert.equal(smoothed[0].value, 10)
    assert.equal(smoothed[2].value, 100, 'group b does not inherit group a’s state')
    assert.equal(smoothed[1].value, 15)
  })

  it('reduces variance without moving the level', () => {
    const raw = []
    for (let i = 0; i < 40; i++) raw.push({ group: 'a', key: i, value: 15 + (i % 2 ? 1 : -1) })
    const smoothed = emaChartRows(raw)
    const spread = (xs) => Math.max(...xs) - Math.min(...xs)
    const rawValues = raw.map((r) => r.value)
    const smoothValues = smoothed.slice(5).map((r) => r.value)
    assert.ok(spread(smoothValues) < spread(rawValues) / 2, 'noise band narrows')
    const mean = smoothValues.reduce((a, b) => a + b, 0) / smoothValues.length
    assert.ok(Math.abs(mean - 15) < 0.2, 'and stays on the same level')
  })

  it('does not mutate the input rows', () => {
    const raw = [{ group: 'a', key: 1, value: 10 }, { group: 'a', key: 2, value: 20 }]
    emaChartRows(raw, 0.5)
    assert.equal(raw[1].value, 20)
  })
})
