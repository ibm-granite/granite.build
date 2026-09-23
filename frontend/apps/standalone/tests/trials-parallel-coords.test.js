/**
 * Tests for the trials parallel-coordinates model.
 *
 * Kept in a pure module because the frontend test harness has no jsdom and cannot
 * render Carbon components, the same split trialMetrics.ts, trialsRadar.ts,
 * trialProgress.ts, trialCompareGrouping.ts and trialHyperparams.ts already make.
 *
 * The fixture is the real eight-trial job 581482fe (SmolLM2-135M-Instruct_finance),
 * because two of its properties are exactly what this model has to get right: the
 * three 3e-6 trials scored within 0.0003 of each other, and trials 00006/00007
 * differ in `r` alone.
 *
 * Usage: node --test tests/trials-parallel-coords.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')

const {
  buildParallelCoords,
} = require('../../../packages/ui-core/components/autotunex/trials/trialsParallelCoords.ts')

const LOSS = { name: 'loss', lowerIsBetter: true }

// id, alpha, lr, dropout, scheduler, batch, r, warmup, loss, total_time
const RAW = [
  ['71085_00000', 1, 1e-6, 0, 'linear', 8, 8, 0.1, 16.34198570251465, 4671.058],
  ['71085_00001', 1, 3e-6, 0, 'linear', 4, 8, 0.2, 16.14883804321289, 4429.726],
  ['71085_00002', 1, 5e-6, 0.05, 'linear', 8, 16, 0.2, 15.96214485168457, 5946.631],
  ['71085_00003', 2, 3e-6, 0.05, 'cosine', 8, 8, 0.1, 16.14906120300293, 5994.867],
  ['71085_00004', 2, 3e-6, 0.05, 'linear', 8, 8, 0.1, 16.148996353149414, 5837.767],
  ['71085_00005', 1, 5e-6, 0, 'cosine', 8, 16, 0.2, 15.957751274108887, 4625.44],
  ['71085_00006', 2, 5e-6, 0.05, 'cosine', 4, 16, 0.1, 14.636462211608887, 5581.511],
  ['71085_00007', 2, 5e-6, 0.05, 'cosine', 4, 8, 0.1, 15.543909072875977, 5556.511],
]

const TRIALS = RAW.map(([id, alpha, lr, drop, sched, bs, r, warmup, loss, time]) => ({
  id,
  metric: 'loss',
  config: {
    alpha_ratio: alpha,
    learning_rate: lr,
    lora_dropout: drop,
    lr_scheduler_type: sched,
    per_device_train_batch_size: bs,
    r,
    warmup_ratio: warmup,
  },
  metrics: { loss, total_time: time },
}))

// The order hyperparamColumns would hand in (its COLUMN_PRIORITY), so these tests
// exercise the reordering rather than assuming it already happened.
const KEYS = [
  'learning_rate',
  'per_device_train_batch_size',
  'r',
  'alpha_ratio',
  'warmup_ratio',
  'lr_scheduler_type',
  'lora_dropout',
]

const byKey = (axes, key) => axes.findIndex((a) => a.key === key)
const trial = (suffix) => TRIALS.find((t) => t.id.endsWith(suffix))

describe('buildParallelCoords — axis layout', () => {
  it('seats learning_rate and r immediately left of the outcome axes', () => {
    // The whole legibility argument for this form: a relationship between
    // neighbouring axes reads as a short parallel run, one between distant axes as
    // a long crossing. If these two drift left the plot becomes spaghetti.
    const { axes } = buildParallelCoords(TRIALS, KEYS, LOSS)
    const firstOutcome = axes.findIndex((a) => a.isOutcome)
    assert.equal(axes[firstOutcome - 1].key, 'learning_rate')
    assert.equal(axes[firstOutcome - 2].key, 'r')
  })

  it('puts the metric last and total_time just before it', () => {
    const { axes } = buildParallelCoords(TRIALS, KEYS, LOSS)
    assert.equal(axes[axes.length - 1].key, 'loss')
    assert.equal(axes[axes.length - 2].key, 'total_time')
    assert.equal(axes.filter((a) => a.isOutcome).length, 2)
  })

  it('does not reorder when trials arrive in a different order', () => {
    // The order is fixed, not variance-derived, precisely so it holds still
    // mid-run. Reversing the input must not move an axis.
    const forward = buildParallelCoords(TRIALS, KEYS, LOSS).axes.map((a) => a.key)
    const reversed = buildParallelCoords([...TRIALS].reverse(), KEYS, LOSS).axes.map((a) => a.key)
    assert.deepEqual(reversed, forward)
  })

  it('keeps an unrecognised hyperparameter, ahead of the known ones', () => {
    const keys = ['learning_rate', 'some_new_flag']
    const trials = TRIALS.map((t) => ({ ...t, config: { ...t.config, some_new_flag: t.config.r } }))
    const { axes } = buildParallelCoords(trials, keys, LOSS)
    assert.equal(axes[0].key, 'some_new_flag')
    assert.equal(axes[1].key, 'learning_rate')
  })
})

describe('buildParallelCoords — outcome axes read better-is-up', () => {
  it('puts the lowest loss at the top of the loss axis', () => {
    const { axes, lines } = buildParallelCoords(TRIALS, KEYS, LOSS)
    const i = byKey(axes, 'loss')
    assert.equal(axes[i].topLabel, '14.6365')
    assert.equal(axes[i].bottomLabel, '16.342')
    const best = lines.find((l) => l.id.endsWith('00006'))
    const worst = lines.find((l) => l.id.endsWith('00000'))
    assert.equal(best.positions[i], 1)
    assert.equal(worst.positions[i], 0)
  })

  it('puts the fastest run at the top of the total time axis', () => {
    const { axes, lines } = buildParallelCoords(TRIALS, KEYS, LOSS)
    const i = byKey(axes, 'total_time')
    assert.equal(axes[i].topLabel, '73m 49s')
    assert.equal(lines.find((l) => l.id.endsWith('00001')).positions[i], 1)
  })

  it('flips the axis for a higher-is-better metric', () => {
    // Direction comes from the caller's `lowerIsBetter`, so a job scored on reward
    // or accuracy must not draw its winner at the bottom — the bug that made the
    // radar contradict `bestTrialId` before isLowerBetter existed.
    const trials = TRIALS.map((t) => ({
      ...t,
      metric: 'accuracy',
      metrics: { accuracy: t.metrics.loss, total_time: t.metrics.total_time },
    }))
    const { axes, lines } = buildParallelCoords(trials, KEYS, {
      name: 'accuracy',
      lowerIsBetter: false,
    })
    const i = byKey(axes, 'accuracy')
    assert.equal(axes[i].topLabel, '16.342')
    assert.equal(lines.find((l) => l.id.endsWith('00000')).positions[i], 1)
    assert.equal(lines.find((l) => l.id.endsWith('00000')).goodness, 1)
  })
})

describe('buildParallelCoords — hyperparameter axes', () => {
  it('places a larger value higher, without inverting', () => {
    const { axes, lines } = buildParallelCoords(TRIALS, KEYS, LOSS)
    const i = byKey(axes, 'r')
    assert.equal(axes[i].topLabel, '16')
    assert.equal(axes[i].bottomLabel, '8')
    assert.equal(lines.find((l) => l.id.endsWith('00006')).positions[i], 1)
    assert.equal(lines.find((l) => l.id.endsWith('00007')).positions[i], 0)
  })

  it('renders a small learning rate exponentially, matching the table cell', () => {
    const { axes } = buildParallelCoords(TRIALS, KEYS, LOSS)
    const i = byKey(axes, 'learning_rate')
    assert.equal(axes[i].topLabel, '5e-6')
    assert.equal(axes[i].bottomLabel, '1e-6')
  })

  it('renders a zero-valued hyperparameter as "0", not "0e+0"', () => {
    const { axes } = buildParallelCoords(TRIALS, KEYS, LOSS)
    assert.equal(axes[byKey(axes, 'lora_dropout')].bottomLabel, '0')
  })

  it('sorts a categorical axis rather than using first-seen order', () => {
    const { axes, lines } = buildParallelCoords(TRIALS, KEYS, LOSS)
    const i = byKey(axes, 'lr_scheduler_type')
    assert.equal(axes[i].bottomLabel, 'cosine')
    assert.equal(axes[i].topLabel, 'linear')
    assert.equal(lines.find((l) => l.id.endsWith('00000')).positions[i], 1)
    assert.equal(lines.find((l) => l.id.endsWith('00003')).positions[i], 0)
  })

  it('breaks the line at a hyperparameter the trial does not report', () => {
    // A missing value must not be drawn as the axis minimum — on an axis where the
    // minimum is a real setting, that would assert something the trial never said.
    const partial = [...TRIALS.slice(0, 7), { ...trial('00007'), config: { r: 8 } }]
    const { axes, lines } = buildParallelCoords(partial, KEYS, LOSS)
    const line = lines.find((l) => l.id.endsWith('00007'))
    assert.equal(line.positions[byKey(axes, 'learning_rate')], null)
    assert.equal(line.values[byKey(axes, 'learning_rate')], '—')
    assert.equal(line.positions[byKey(axes, 'r')], 0)
  })
})

describe('buildParallelCoords — colour ramp goodness', () => {
  it('interpolates on the metric value, so near-identical scores read alike', () => {
    // The three 3e-6 trials scored within 0.0003 of each other. Ranking them would
    // spread them across three visibly different colours and imply a separation the
    // numbers do not support.
    const { lines } = buildParallelCoords(TRIALS, KEYS, LOSS)
    const g = ['00001', '00003', '00004'].map((s) => lines.find((l) => l.id.endsWith(s)).goodness)
    assert.ok(Math.max(...g) - Math.min(...g) < 0.001, `spread too wide: ${g}`)
    // ...and they are still clearly separated from the winner.
    const best = lines.find((l) => l.id.endsWith('00006')).goodness
    assert.ok(best - Math.max(...g) > 0.8, `winner not separated: ${best} vs ${g}`)
  })

  it('reports no goodness for a trial that logged no metric', () => {
    const trials = [...TRIALS.slice(0, 7), { ...trial('00007'), metrics: { total_time: 5556.511 } }]
    const { lines } = buildParallelCoords(trials, KEYS, LOSS)
    assert.equal(lines.find((l) => l.id.endsWith('00007')).goodness, null)
  })
})

describe('buildParallelCoords — the invariant the renderer depends on', () => {
  // TrialSearchSpace maps position 0 -> the bottom of the plot and 1 -> the top by
  // linear interpolation, with no clamp. Anything outside [0, 1] therefore draws
  // outside the plot area — over the axis names, or off the viewBox entirely — so
  // this is the one property the SVG silently relies on.
  const inRange = (p) => p === null || (Number.isFinite(p) && p >= 0 && p <= 1)

  it('keeps every position within [0, 1] for the full run', () => {
    const { axes, lines } = buildParallelCoords(TRIALS, KEYS, LOSS)
    for (const line of lines) {
      for (const [i, p] of line.positions.entries()) {
        assert.ok(inRange(p), `${line.id} on ${axes[i].key}: ${p}`)
      }
      assert.ok(line.goodness === null || (line.goodness >= 0 && line.goodness <= 1))
    }
  })

  it('keeps them in range for every single-trial subset against full-run bounds', () => {
    // The case most likely to escape the range: one trial's value scaled against
    // bounds it did not set. Every trial, on every axis.
    for (const one of TRIALS) {
      const { axes, lines } = buildParallelCoords([one], KEYS, LOSS, TRIALS)
      for (const [i, p] of lines[0].positions.entries()) {
        assert.ok(inRange(p), `${one.id} alone on ${axes[i].key}: ${p}`)
      }
    }
  })

  it('emits one position and one value per axis', () => {
    // The renderer indexes positions and values by axis index; a length mismatch
    // would pair a value with the wrong axis rather than throwing.
    const { axes, lines } = buildParallelCoords(TRIALS, KEYS, LOSS)
    for (const line of lines) {
      assert.equal(line.positions.length, axes.length)
      assert.equal(line.values.length, axes.length)
    }
  })
})

describe('buildParallelCoords — degenerate inputs', () => {
  it('returns nothing for no trials', () => {
    assert.deepEqual(buildParallelCoords([], KEYS, LOSS), { axes: [], lines: [] })
  })

  it('draws a single trial at the top of every axis instead of dividing by zero', () => {
    // One trial has no range of its own: min === max on every axis. The radar hit
    // the same case and pinned the blob to the centre; here the guard must produce
    // a finite position, not NaN.
    const { axes, lines } = buildParallelCoords([trial('00002')], KEYS, LOSS)
    assert.equal(lines.length, 1)
    for (const [i, p] of lines[0].positions.entries()) {
      assert.ok(Number.isFinite(p), `axis ${axes[i].key} produced ${p}`)
      assert.equal(p, 1)
    }
    assert.equal(lines[0].goodness, 1)
  })

  it('holds the axis scale still when only a subset is plotted', () => {
    // `boundsFrom` is the whole run, so ticking trials on and off must not reshape
    // the axes under the reader.
    const subset = [trial('00003'), trial('00004')]
    const { axes, lines } = buildParallelCoords(subset, KEYS, LOSS, TRIALS)
    const i = byKey(axes, 'loss')
    assert.equal(axes[i].topLabel, '14.6365')
    // Both are mid-pack against the full run, so neither may sit at an extreme.
    for (const line of lines) {
      assert.ok(line.positions[i] > 0 && line.positions[i] < 1, `pinned to an end: ${line.positions[i]}`)
    }
  })

  it('omits an outcome axis no trial reports', () => {
    const trials = TRIALS.map((t) => ({ ...t, metrics: { loss: t.metrics.loss } }))
    const { axes } = buildParallelCoords(trials, KEYS, LOSS)
    assert.equal(byKey(axes, 'total_time'), -1)
    assert.equal(axes[axes.length - 1].key, 'loss')
  })

  it('handles a boolean hyperparameter as a categorical axis', () => {
    const keys = ['learning_rate', 'use_flash']
    const trials = TRIALS.map((t, i) => ({ ...t, config: { ...t.config, use_flash: i % 2 === 0 } }))
    const { axes, lines } = buildParallelCoords(trials, keys, LOSS)
    const i = byKey(axes, 'use_flash')
    assert.equal(axes[i].bottomLabel, 'false')
    assert.equal(axes[i].topLabel, 'true')
    assert.equal(lines[0].positions[i], 1)
    assert.equal(lines[1].positions[i], 0)
  })
})
