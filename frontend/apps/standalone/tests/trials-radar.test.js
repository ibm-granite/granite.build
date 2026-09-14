/**
 * Tests for the trials radar's normalization.
 *
 * The bug this guards: the axes plot *goodness*, but normalization was a plain
 * min-max with no notion of direction. Live jobs report `loss`, `train_loss` and
 * `total_time` — all three lower-is-better — so every axis read backwards: the
 * worst trial took the outer vertex while the winning trial collapsed toward the
 * centre, on precisely the metric `bestTrialId` minimises.
 *
 * The second thing worth guarding is the missing-metric fallback. It used to be
 * the axis minimum, which is harmless when bigger means better and actively wrong
 * once the axis is inverted — a trial that reported nothing would draw as the
 * winner on that axis.
 *
 * Usage: node --test tests/trials-radar.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')

const {
  toRadarData,
  isLowerBetter,
  toFeatureLabel,
} = require('../../../packages/ui-core/components/trialsRadar.ts')

const trial = (id, metrics) => ({ id, status: 'completed', metrics })
const scoreOf = (rows, id, feature) =>
  rows.find((r) => r.product === id && r.feature === feature)?.score

describe('isLowerBetter', () => {
  it('classifies the metrics live jobs actually report', () => {
    // Verified against GET /jobs/{id}/trials: loss, train_loss, total_time.
    for (const name of ['loss', 'train_loss', 'total_time']) {
      assert.equal(isLowerBetter(name), true, `${name} is lower-is-better`)
    }
  })

  it('classifies other loss/time/error shapes the upstream may add', () => {
    for (const name of ['eval_loss', 'train_runtime', 'perplexity', 'error_rate', 'latency_ms']) {
      assert.equal(isLowerBetter(name), true, name)
    }
  })

  it('defaults to higher-is-better for the accuracy family', () => {
    for (const name of ['accuracy', 'precision', 'recall', 'f1', 'reward', 'samples_per_second']) {
      assert.equal(isLowerBetter(name), false, name)
    }
  })
})

describe('toRadarData direction', () => {
  it('puts the lowest loss at the outer edge, not the centre', () => {
    const best = trial('best', { loss: 0.5 })
    const worst = trial('worst', { loss: 1.5 })
    const rows = toRadarData([best, worst])

    assert.equal(scoreOf(rows, 'best', 'Loss'), 1, 'the winning trial reaches the rim')
    assert.equal(scoreOf(rows, 'worst', 'Loss'), 0, 'the worst trial sits at the centre')
  })

  it('keeps the raw direction for a higher-is-better metric', () => {
    const rows = toRadarData([trial('a', { accuracy: 0.8 }), trial('b', { accuracy: 0.9 })])
    assert.equal(scoreOf(rows, 'b', 'Accuracy'), 1)
    assert.equal(scoreOf(rows, 'a', 'Accuracy'), 0)
  })

  it('inverts total_time as well, so the fastest trial reads best', () => {
    const rows = toRadarData([trial('quick', { total_time: 60 }), trial('slow', { total_time: 600 })])
    assert.equal(scoreOf(rows, 'quick', 'Total Time'), 1)
    assert.equal(scoreOf(rows, 'slow', 'Total Time'), 0)
  })

  it('places a mid-range value symmetrically', () => {
    const rows = toRadarData([
      trial('lo', { loss: 0 }),
      trial('mid', { loss: 1 }),
      trial('hi', { loss: 2 }),
    ])
    assert.equal(scoreOf(rows, 'mid', 'Loss'), 0.5)
  })
})

describe('toRadarData edge cases', () => {
  it('plots a metric with no spread at mid-radius, whatever its direction', () => {
    const rows = toRadarData([trial('a', { loss: 1, accuracy: 0.9 }), trial('b', { loss: 1, accuracy: 0.9 })])
    for (const id of ['a', 'b']) {
      assert.equal(scoreOf(rows, id, 'Loss'), 0.5)
      assert.equal(scoreOf(rows, id, 'Accuracy'), 0.5)
    }
  })

  it('draws a trial that never reported a metric at the centre, not the rim', () => {
    // The inversion hazard: falling back to the axis minimum would score this 1.
    const rows = toRadarData([trial('full', { loss: 0.5, total_time: 60 }), trial('partial', { loss: 1.5 })])
    assert.equal(scoreOf(rows, 'partial', 'Total Time'), 0, 'not reported reads as nothing, not best')
    assert.equal(scoreOf(rows, 'full', 'Total Time'), 0.5, 'the only reporter has no spread to rank against')
  })

  it('scales against the run, not just the plotted trials', () => {
    const run = [trial('a', { loss: 0 }), trial('b', { loss: 1 }), trial('c', { loss: 2 })]
    const rows = toRadarData([run[1]], run)
    // Scaled against itself the lone trial would have no range at all (0.5);
    // against the run it sits where it landed among its siblings.
    assert.equal(scoreOf(rows, 'b', 'Loss'), 0.5)

    const rimmed = toRadarData([run[0]], run)
    assert.equal(scoreOf(rimmed, 'a', 'Loss'), 1, 'the best trial in the run reaches the rim alone')
  })

  it('emits a complete grid, which Carbon requires', () => {
    const rows = toRadarData([trial('a', { loss: 1, accuracy: 0.5 }), trial('b', { total_time: 30 })])
    const features = new Set(rows.map((r) => r.feature))
    assert.deepEqual([...features].sort(), ['Accuracy', 'Loss', 'Total Time'])
    for (const id of ['a', 'b']) {
      for (const f of features) {
        assert.equal(typeof scoreOf(rows, id, f), 'number', `${id} has a value for ${f}`)
      }
    }
  })

  it('returns nothing when no trial carries metrics', () => {
    assert.deepEqual(toRadarData([trial('a', {}), trial('b', null)]), [])
  })

  it('labels axes from the metric key', () => {
    assert.equal(toFeatureLabel('total_time'), 'Total Time')
    assert.equal(toFeatureLabel('loss'), 'Loss')
  })
})
