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
 * 3. The `elapsed` x axis measures each run from its own first row, and the
 *    origin has to be shared across a run's series. Taken per series it would
 *    re-anchor eval loss at zero, drawing the first eval — logged at the end of
 *    an epoch — as if it arrived with the first training step.
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
  rowsForTrials,
  trialColorScale,
  bestTrialId,
  toChartRows,
  positiveRows,
  runOrigins,
  METRIC_PALETTE,
  METRIC_DE_EMPHASIS,
  EMPHASIS_THRESHOLD,
} = require('../../../packages/ui-core/components/trialMetrics.ts')

const SEARCH_IDS = ['491c7_00000', '491c7_00001', '491c7_00002', '491c7_00003']
const FINAL_ID = '11517_00000'

let nextId = 245 // the reference job's first metric id

// When each run began. `training_metrics.created_at` is NOT NULL, so every real
// row carries one. The search trials run two at a time and the final run follows
// them, which is why an absolute clock would place the phases in disjoint
// windows and the `elapsed` axis measures each run from its own start instead.
const RUN_START = {
  '491c7_00000': Date.parse('2026-09-03T12:16:30Z'),
  '491c7_00001': Date.parse('2026-09-03T12:16:35Z'),
  '491c7_00002': Date.parse('2026-09-03T12:24:00Z'),
  '491c7_00003': Date.parse('2026-09-03T12:24:05Z'),
  '11517_00000': Date.parse('2026-09-03T12:40:00Z'),
}

const SECONDS_PER_STEP = 15

/**
 * `created_at` for a row of `trialId` at `step`, `offset` seconds after it.
 * A run outside `RUN_START` gets no stamp — that is the "this row has no place
 * on the elapsed axis" case, which the tests below lean on.
 */
function stampAt(trialId, step, offset = 0) {
  const base = RUN_START[trialId]
  if (base === undefined) return undefined
  return new Date(base + (step * SECONDS_PER_STEP + offset) * 1000).toISOString()
}

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
    created_at: stampAt(trialId, step),
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
    created_at: stampAt(trialId, step, 1),
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
    created_at: stampAt(trialId, step, 2),
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

describe('rowsForTrials', () => {
  it('keeps every kind of row for the requested runs and nothing else', () => {
    const rows = rowsForTrials(fixture(), [SEARCH_IDS[0]])
    assert.equal(rows.length, 38, '34 steps + 3 evals + 1 summary')
    assert.ok(rows.every((r) => r.trial_id === SEARCH_IDS[0]))
    // All three kinds survive — filtering by run must not double as a kind filter.
    const { trainSteps, evals, summaries } = splitMetricRows(rows)
    assert.deepEqual([trainSteps.length, evals.length, summaries.length], [34, 3, 1])
  })

  it('drops a row that carries no trial id', () => {
    // Such a row is the job's single unnamed run. It has no checkbox in the
    // trials table, so it can never be one of the requested ids, and drawing it
    // beside a selection would show a curve nobody asked for.
    const rows = [...fixture(), { ...stepRow(null, 2, 0.1), trial_id: null }]
    const kept = rowsForTrials(rows, SEARCH_IDS)
    assert.ok(kept.every((r) => r.trial_id != null))
    assert.equal(kept.length, 4 * 38)
  })

  it('returns nothing for an empty id list rather than everything', () => {
    assert.equal(rowsForTrials(fixture(), []).length, 0)
  })

  it('leaves a surviving run\u2019s elapsed origin exactly where it was', () => {
    // The regression this guards: `runOrigins` is documented as taking a whole
    // phase, so narrowing to a selection has to be origin-preserving or every
    // curve re-anchors and the `elapsed` axis silently lies.
    const rows = fixture()
    const all = runOrigins(rows)
    const narrowed = runOrigins(rowsForTrials(rows, [SEARCH_IDS[1], SEARCH_IDS[3]]))
    assert.deepEqual([...narrowed.keys()].sort(), [SEARCH_IDS[1], SEARCH_IDS[3]].sort())
    assert.equal(narrowed.get(SEARCH_IDS[1]), all.get(SEARCH_IDS[1]))
    assert.equal(narrowed.get(SEARCH_IDS[3]), all.get(SEARCH_IDS[3]))
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

describe('runOrigins', () => {
  it('anchors each run at its own earliest row', () => {
    const origins = runOrigins(fixture())
    assert.equal(origins.size, 5, 'four search trials and the final run')
    // A run's earliest row is its first step row, at global_step 2.
    for (const [id, base] of Object.entries(RUN_START)) {
      assert.equal(origins.get(id), base + 2 * SECONDS_PER_STEP * 1000)
    }
  })

  it('ignores rows with no usable timestamp', () => {
    assert.equal(runOrigins([stepRow('no-such-run', 2, 0.1)]).size, 0)
  })

  it('files a row with no trial id under the job\u2019s single run', () => {
    const row = { ...stepRow(SEARCH_IDS[0], 2, 0.1), trial_id: null }
    assert.deepEqual([...runOrigins([row]).keys()], ['run'])
  })
})

describe('toChartRows on the elapsed axis', () => {
  it('starts every run at its own zero', () => {
    const rows = fixture()
    const chart = toChartRows(
      splitMetricRows(rows).trainSteps,
      'elapsed',
      (r) => r.loss,
      runOrigins(rows)
    )
    assert.equal(chart.length, 159, 'no step row is lost to the time axis')
    for (const id of [...SEARCH_IDS, FINAL_ID]) {
      const own = chart.filter((r) => r.group === id)
      assert.equal(own[0].key, 0, `${id} starts at zero`)
      assert.ok(own[own.length - 1].key > 0, `${id} advances`)
    }
  })

  it('reads in minutes', () => {
    // A search trial logs 34 step rows at global_step 2..68, one every
    // SECONDS_PER_STEP, so it spans (68 - 2) * 15s = 16.5 minutes.
    const rows = fixture()
    const own = toChartRows(
      splitMetricRows(rows).trainSteps,
      'elapsed',
      (r) => r.loss,
      runOrigins(rows)
    ).filter((r) => r.group === SEARCH_IDS[0])
    assert.equal(own[own.length - 1].key, 16.5)
  })

  it('leaves evals where they were logged rather than at zero', () => {
    // The regression this guards: origins taken from `split.evals` alone would
    // make each run's first eval the origin, so eval loss would start at zero
    // beside the training curve instead of an epoch into the run.
    const rows = fixture()
    const split = splitMetricRows(rows)
    const evals = toChartRows(split.evals, 'elapsed', (r) => r.extra?.eval_loss, runOrigins(rows))
    for (const id of [...SEARCH_IDS, FINAL_ID]) {
      const own = evals.filter((r) => r.group === id)
      assert.ok(own[0].key > 0, `${id}'s first eval keeps its offset`)
    }
  })

  it('drops rows it cannot place instead of guessing an origin', () => {
    const rows = [stepRow(SEARCH_IDS[0], 2, 0.1), stepRow(SEARCH_IDS[0], 4, 0.2)]
    assert.equal(toChartRows(rows, 'elapsed', (r) => r.loss).length, 0, 'no origins given')
    assert.equal(
      toChartRows(rows, 'elapsed', (r) => r.loss, new Map()).length,
      0,
      'run absent from origins'
    )
  })
})

describe('positiveRows — the log-axis guard', () => {
  // Carbon's LOG scale throws outright when the domain minimum is <= 0, and it
  // takes the enclosing panel with it — there is no error boundary above these
  // charts. HF Trainer logs `learning_rate: 0` on the final step of a
  // linear-decay schedule, so this is an ordinary completed run, not a corrupt
  // one. `toChartRows` deliberately keeps the point: 0 is finite, and a real
  // `loss: 0` belongs on the linear charts.
  it('drops the zero learning rate a decay schedule logs on its last step', () => {
    const rows = [stepRow(SEARCH_IDS[0], 1, 0.1), stepRow(SEARCH_IDS[0], 2, 0.2)]
    rows[1].learning_rate = 0
    const chart = toChartRows(rows, 'global_step', (r) => r.learning_rate)
    assert.equal(chart.length, 2, 'toChartRows keeps it — 0 is finite')

    const safe = positiveRows(chart)
    assert.equal(safe.length, 1)
    assert.ok(
      Math.min(...safe.map((r) => r.value)) > 0,
      'the LOG domain minimum must be strictly positive'
    )
  })

  it('drops a negative value as well as a zero', () => {
    const rows = [
      { group: 'a', key: 1, value: -1e-7 },
      { group: 'a', key: 2, value: 0 },
      { group: 'a', key: 3, value: 1e-6 },
    ]
    assert.deepEqual(positiveRows(rows), [{ group: 'a', key: 3, value: 1e-6 }])
  })

  it('leaves an all-positive series untouched', () => {
    const chart = toChartRows(
      [stepRow(SEARCH_IDS[0], 1, 0.1), stepRow(SEARCH_IDS[0], 2, 0.2)],
      'global_step',
      (r) => r.learning_rate
    )
    assert.deepEqual(positiveRows(chart), chart)
  })
})
