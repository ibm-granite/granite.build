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
  rowsForKnownRuns,
  trialColorScale,
  emphasisColorScale,
  selectionSlots,
  toChartRows,
  positiveRows,
  logDomain,
  runOrigins,
  METRIC_PALETTE,
  METRIC_DE_EMPHASIS,
  EMPHASIS_THRESHOLD,
} = require('../../../packages/ui-core/components/autotunex/trials/trialMetrics.ts')

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
    const { search, final, finalTrialIds } = derivePhases(fixture(), SEARCH_IDS, true, true)
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
    const { search, final, finalTrialIds } = derivePhases(rows, [], false, true)
    assert.equal(search.length, rows.length)
    assert.equal(final.length, 0)
    assert.deepEqual(finalTrialIds, [])
  })

  it('treats an unrecognised run as search while the search can still start one', () => {
    // The reported race: the metrics and trials queries are independent polls, so
    // mid-search Ray starts trial #5 and its metric rows arrive before its /trials
    // row. Those rows used to land in the final phase, so the panel announced the
    // winning configuration for a trial that was still searching and hid the search
    // charts until the next trials tick corrected it.
    const rows = fixture()
    const { search, final, finalTrialIds } = derivePhases(rows, SEARCH_IDS, true, false)
    assert.equal(search.length, rows.length)
    assert.equal(final.length, 0)
    assert.deepEqual(finalTrialIds, [])
  })

  it('does not relabel a row that carries no trial id', () => {
    const rows = [...fixture(), { ...stepRow(null, 2, 0.1), trial_id: null }]
    const { final, search } = derivePhases(rows, SEARCH_IDS, true, true)
    assert.ok(final.every((r) => r.trial_id === FINAL_ID))
    assert.equal(search.filter((r) => r.trial_id == null).length, 1)
  })

  it('groups several unrecognised run ids under the final phase', () => {
    const rows = [...fixture(), stepRow('99999_00000', 2, 0.1)]
    const { finalTrialIds } = derivePhases(rows, SEARCH_IDS, true, true)
    assert.deepEqual(finalTrialIds, [FINAL_ID, '99999_00000'])
  })
})

describe('rowsForKnownRuns', () => {
  it('drops a run the trials table has no row for', () => {
    // The reported bug, in the shape that produced it: job
    // 2c6f6aa3-2656-4e89-8436-c9ce0cc3507f declared num_trials=16 but wrote only
    // 10 trial rows, four of them left `running` when the job finished. `resolved`
    // can therefore never reach 16, so `isSearchComplete` is false for good, and
    // `derivePhases` stays in its bail-out branch and hands the final run back as a
    // search row. It then reached the search charts' shared y-scale, where its
    // full-data descent next to the trials' one-epoch stubs is exactly the
    // comparison those charts are built to prevent -- and its legend entry named a
    // run the reader could not find in the table.
    const rows = fixture()
    const { search } = derivePhases(rows, SEARCH_IDS, true, false)
    assert.equal(search.length, rows.length, 'the bail-out branch keeps the final run')

    const kept = rowsForKnownRuns(search, SEARCH_IDS)
    assert.equal(kept.length, 4 * 38)
    assert.ok(kept.every((r) => SEARCH_IDS.includes(r.trial_id)))
    assert.ok(!kept.some((r) => r.trial_id === FINAL_ID))
  })

  it('keeps every kind of row for the runs it does list', () => {
    // Filtering by run must not double as a kind filter -- an eval-only selection
    // still has a curve to draw.
    const { trainSteps, evals, summaries } = splitMetricRows(
      rowsForKnownRuns(fixture(), [SEARCH_IDS[0]])
    )
    assert.deepEqual([trainSteps.length, evals.length, summaries.length], [34, 3, 1])
  })

  it('keeps a row that carries no trial id', () => {
    // Where this parts company with `rowsForTrials`. Such a row is the job's single
    // unnamed run, which owns no trials-table row and so can never be named by one;
    // dropping it would blank the charts for a plain tuning job, whose every row is
    // untagged.
    const rows = [{ ...stepRow(null, 2, 0.1), trial_id: null }]
    assert.equal(rowsForKnownRuns(rows, []).length, 1)
    assert.equal(rowsForKnownRuns(rows, SEARCH_IDS).length, 1)
  })

  it('leaves a surviving run\u2019s elapsed origin exactly where it was', () => {
    // Same contract as `rowsForTrials`: this drops whole runs and never reorders or
    // trims a survivor's rows, so `runOrigins` may be taken after it.
    const rows = fixture()
    const all = runOrigins(rows)
    const narrowed = runOrigins(rowsForKnownRuns(rows, SEARCH_IDS))
    assert.deepEqual([...narrowed.keys()].sort(), [...SEARCH_IDS].sort())
    for (const id of SEARCH_IDS) assert.equal(narrowed.get(id), all.get(id))
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
    const scale = trialColorScale(SEARCH_IDS, 'white')
    const used = Object.values(scale)
    assert.equal(new Set(used).size, SEARCH_IDS.length, 'no two runs share a hue')
    assert.equal(scale[SEARCH_IDS[0]], METRIC_PALETTE.white[0])
  })

  it('keeps a run’s colour when another run is hidden', () => {
    // Colour must follow the run, not its position among the *visible* ones —
    // otherwise hiding a series repaints the survivors and a reader who learned
    // "_00002 is purple" is misled.
    const all = trialColorScale(SEARCH_IDS, 'white')
    const stillOrdered = trialColorScale(SEARCH_IDS, 'white')
    assert.equal(stillOrdered['491c7_00003'], all['491c7_00003'])
    // And the scale is keyed by id, so a caller filtering its *data* down to two
    // series reuses the same map untouched.
    assert.equal(all['491c7_00002'], METRIC_PALETTE.white[2])
  })

  it('cycles the palette past the threshold instead of greying runs out', () => {
    const many = Array.from({ length: EMPHASIS_THRESHOLD + 3 }, (_, i) => `t_${i}`)
    const scale = trialColorScale(many, 'white')
    assert.ok(
      Object.values(scale).every((c) => METRIC_PALETTE.white.includes(c)),
      'no run falls back to the de-emphasis grey'
    )
    assert.equal(scale[`t_${EMPHASIS_THRESHOLD}`], METRIC_PALETTE.white[0], 'the eleventh run wraps to slot 0')
  })

  it('never repaints an existing run when the job gains trials', () => {
    // A running job's trials list grows between polls. Colour follows the run, so
    // the eleventh trial arriving must not move the first ten.
    const ten = Array.from({ length: EMPHASIS_THRESHOLD }, (_, i) => `t_${i}`)
    const before = trialColorScale(ten, 'white')
    const after = trialColorScale([...ten, 't_10', 't_11'], 'white')
    for (const id of ten) assert.equal(after[id], before[id], `${id} kept its colour`)
  })

  it('uses selected dark steps, not the light ones', () => {
    const light = trialColorScale(SEARCH_IDS, 'white')
    const dark = trialColorScale(SEARCH_IDS, 'g100')
    // purple-70 and green-60 are too dark on Carbon's g100 layer.
    assert.notEqual(light['491c7_00002'], dark['491c7_00002'])
    assert.notEqual(light['491c7_00003'], dark['491c7_00003'])
    assert.equal(dark['491c7_00002'], '#a56eff')
  })
})

describe('emphasisColorScale', () => {
  it('keeps the best run in its own colour and greys every other run', () => {
    const many = Array.from({ length: EMPHASIS_THRESHOLD + 3 }, (_, i) => `t_${i}`)
    const scale = trialColorScale(many, 'white')
    const emphasis = emphasisColorScale(scale, 't_4', 'white')
    // Its own slot, not slot 0: the best row's checkbox tint and its Metrics tab
    // use the permanent scale, and the charts must agree with them.
    assert.equal(emphasis['t_4'], scale['t_4'])
    assert.notEqual(emphasis['t_4'], METRIC_PALETTE.white[0])
    const others = many.filter((id) => id !== 't_4').map((id) => emphasis[id])
    assert.ok(others.every((c) => c === METRIC_DE_EMPHASIS.white))
  })

  it('greys everything when there is no best run yet', () => {
    const scale = trialColorScale(SEARCH_IDS, 'g100')
    const emphasis = emphasisColorScale(scale, undefined, 'g100')
    assert.ok(Object.values(emphasis).every((c) => c === METRIC_DE_EMPHASIS.g100))
  })
})

describe('selectionSlots', () => {
  const many = Array.from({ length: EMPHASIS_THRESHOLD + 6 }, (_, i) => `t_${i}`)

  it('gives a ticked run its home slot when no other ticked run holds it', () => {
    assert.deepEqual(selectionSlots(many, ['t_3', 't_12'], {}), { t_3: 3, t_12: 2 })
  })

  it('lends the lowest free slot when a ticked run already holds the home one', () => {
    // t_10 shares t_0's home slot, so it borrows — rather than being refused, which
    // disabled rows while the reader was still under the selection cap.
    assert.deepEqual(selectionSlots(many, ['t_0', 't_10'], {}), { t_0: 0, t_10: 1 })
  })

  it('keeps every ticked run distinct up to the cap, however the homes collide', () => {
    const ticked = ['t_0', 't_10', 't_1', 't_11', 't_2', 't_12', 't_3', 't_13', 't_4', 't_14']
    const slots = selectionSlots(many, ticked, {})
    assert.equal(new Set(Object.values(slots)).size, ticked.length)
  })

  it('leaves a borrowed slot in place when its home frees up', () => {
    // Colour follows the run while it stays ticked: unticking t_0 must not repaint
    // t_10's curve back to slot 0 under the reader.
    const first = selectionSlots(many, ['t_0', 't_10'], {})
    const after = selectionSlots(many, ['t_10'], first)
    assert.deepEqual(after, { t_10: 1 })
  })

  it('drops unticked runs, so their slot is free for the next tick', () => {
    const first = selectionSlots(many, ['t_0', 't_10'], {})
    const after = selectionSlots(many, ['t_10', 't_5'], selectionSlots(many, ['t_10'], first))
    assert.deepEqual(after, { t_10: 1, t_5: 5 })
    // t_0's home is free again, so re-ticking it gets its own colour back.
    assert.equal(selectionSlots(many, ['t_10', 't_0'], selectionSlots(many, ['t_10'], first)).t_0, 0)
  })

  it('never borrows in a job at or below the threshold', () => {
    const slots = selectionSlots(SEARCH_IDS, [...SEARCH_IDS].reverse(), {})
    SEARCH_IDS.forEach((id, i) => assert.equal(slots[id], i))
  })
})

describe('trialColorScale with selection slots', () => {
  it('paints a ticked run from its slot and every other run from its home', () => {
    const many = Array.from({ length: EMPHASIS_THRESHOLD + 3 }, (_, i) => `t_${i}`)
    const scale = trialColorScale(many, 'white', { t_10: 1 })
    assert.equal(scale['t_10'], METRIC_PALETTE.white[1])
    assert.equal(scale['t_11'], METRIC_PALETTE.white[1], 'an unticked run keeps its home colour')
    assert.equal(scale['t_0'], METRIC_PALETTE.white[0])
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

describe('logDomain — the log-axis headroom', () => {
  // Carbon pads an axis domain by `(max - min) * 0.1` (paddingRatio, in
  // configuration-non-customizable, so options cannot change it) and applies that
  // *linear* pad whatever the scale type. On a learning-rate axis spanning
  // 1.2e-9..4.9e-6 that pad is ~10% of the max, which is log10(1.1) = 0.04 of the
  // 3.67 decades on screen — 1.1% of the plot height, less than the stroke is
  // wide, so every schedule's peak came out flat-topped against the plot frame.
  // The floor is worse: the LOG branch clamps the lower bound back to the data
  // minimum exactly, so the lowest point sat *on* the bottom axis.
  //
  // `nn` below is that function, transcribed from
  // node_modules/@carbon/charts/dist/index-CHbrPDmO.mjs, so these tests measure
  // what Carbon will actually draw rather than what we hand it.
  const PADDING_RATIO = 0.1
  function nn([min, max], ratio, isLog) {
    const pad = (max - min) * ratio
    const upper = max <= 0 && max + pad > 0 ? 0 : max + pad
    let lower = min >= 0 && min - pad < 0 ? 0 : min - pad
    if (isLog && lower <= 0) {
      if (min <= 0) throw Error('Data must have values greater than 0 if log scale type is used.')
      lower = min
    }
    return [lower, upper]
  }

  /** Where `value` lands as a fraction of plot height, 0 = bottom edge, 1 = top. */
  function heightFraction(value, [lo, hi]) {
    const span = Math.log10(hi) - Math.log10(lo)
    return (Math.log10(value) - Math.log10(lo)) / span
  }

  const rows = (...values) => values.map((value, i) => ({ group: 'a', key: i, value }))

  it('leaves the peak and the floor clear of the plot frame', () => {
    // The reported symptom, measured after Carbon re-pads what we hand it: an
    // explicit `domain` still goes through `extendsDomain`. A 3px point marker on
    // these charts needs ~3% of a 220px chart's plot area to draw in full.
    const data = rows(1.156e-9, 5e-8, 1e-6, 4.878e-6)
    const drawn = nn(logDomain(data), PADDING_RATIO, true)
    const top = heightFraction(4.878e-6, drawn)
    const bottom = heightFraction(1.156e-9, drawn)
    assert.ok(top < 0.97, `peak sits at ${(top * 100).toFixed(1)}% of plot height, needs < 97%`)
    assert.ok(bottom > 0.03, `floor sits at ${(bottom * 100).toFixed(1)}% of plot height, needs > 3%`)
  })

  it('is what Carbon\u2019s own unpadded domain is not', () => {
    // Guards the fix against being reverted to "Carbon already pads it".
    const data = rows(1.156e-9, 4.878e-6)
    const values = data.map((r) => r.value)
    const carbon = nn([Math.min(...values), Math.max(...values)], PADDING_RATIO, true)
    assert.ok(heightFraction(4.878e-6, carbon) > 0.98, 'unpadded: peak is against the frame')
    assert.equal(heightFraction(1.156e-9, carbon), 0, 'unpadded: floor is exactly on the axis')
  })

  it('pads in decades, so headroom does not depend on how wide the span is', () => {
    // The bug in one line: a pad computed on `max - min` is worth almost nothing
    // in log space once the span covers a few decades. This one is scale-free.
    //
    // Both spans are wider than half a decade, which is where the constant-series
    // floor stops being the binding term — a 0.3-decade span takes the floor and
    // gets proportionally more room, which is deliberate, not a counterexample.
    const headroom = ([lo, hi], max) =>
      (Math.log10(hi) - Math.log10(max)) / (Math.log10(hi) - Math.log10(lo))
    const narrow = headroom(logDomain(rows(1e-6, 1e-5)), 1e-5)
    const wide = headroom(logDomain(rows(1e-9, 1e-5)), 1e-5)
    assert.ok(
      Math.abs(narrow - wide) < 0.005,
      `1 decade gives ${narrow.toFixed(4)}, 4 decades ${wide.toFixed(4)}`
    )
  })

  it('returns a usable domain for a series that never changes value', () => {
    // A constant learning-rate schedule. Carbon's pad is `(max - min) * ratio` = 0
    // here, so its domain collapses to [v, v] and the d3 log scale degenerates.
    const [lo, hi] = logDomain(rows(3e-6, 3e-6, 3e-6))
    assert.ok(lo < 3e-6 && hi > 3e-6, `expected 3e-6 strictly inside [${lo}, ${hi}]`)
  })

  it('declines to guess when there is nothing to measure', () => {
    assert.equal(logDomain([]), undefined)
    // positiveRows runs first at every call site, so this is belt and braces —
    // but a zero here would make Carbon throw, and returning undefined leaves its
    // own (throwing) behaviour exactly as it was rather than hiding it.
    assert.equal(logDomain(rows(0, 1e-6)), undefined)
    assert.equal(logDomain(rows(-1e-7)), undefined)
  })
})
