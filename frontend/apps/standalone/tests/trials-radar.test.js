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
  primaryMetric,
  bestTrialId,
} = require('../../../packages/ui-core/components/autotunex/trials/trialsRadar.ts')

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

// `primaryMetric` and `bestTrialId` moved here from trialMetrics.ts so they sit
// beside the `isLowerBetter` predicate they have to agree with.
describe('primaryMetric', () => {
  const scored = (id, metric, metrics) => ({ id, status: 'completed', metric, metrics })

  it('reads the key the trial says it was scored on', () => {
    assert.deepEqual(primaryMetric(scored('a', 'reward', { reward: 0.8, loss: 2 })), {
      name: 'reward',
      value: 0.8,
    })
  })

  it('falls back to a literal loss when the trial names no metric', () => {
    // This is the divergence that let the table and Compare order the same trials
    // two different ways: the table read only metrics[metric] and printed an em
    // dash, while Compare's lossOf already fell back to metrics.loss.
    assert.deepEqual(primaryMetric({ id: 'a', status: 'completed', metrics: { loss: 15.2 } }), {
      name: 'loss',
      value: 15.2,
    })
  })

  it('falls back when the named metric is absent from metrics', () => {
    assert.deepEqual(primaryMetric(scored('a', 'missing', { loss: 3 })), { name: 'loss', value: 3 })
  })

  it('does not fall back when the named metric is present but unusable', () => {
    // Preserved from the previous lossOf: the trial WAS scored on `reward`, so
    // ranking it by `loss` instead would compare it against the others on a metric
    // it was not judged on. Unusable means unranked.
    assert.equal(primaryMetric(scored('a', 'reward', { reward: Number.NaN, loss: 3 })), null)
  })

  it('returns null when nothing usable is reported', () => {
    assert.equal(primaryMetric({ id: 'a', status: 'completed', metrics: {} }), null)
    assert.equal(primaryMetric({ id: 'a', status: 'completed' }), null)
    assert.equal(primaryMetric(scored('a', 'loss', { loss: Number.POSITIVE_INFINITY })), null)
  })
})

describe('bestTrialId', () => {
  const scored = (id, metric, value) => ({ id, status: 'completed', metric, metrics: { [metric]: value } })

  it('picks the lowest value on a lower-is-better metric', () => {
    const trials = [scored('a', 'loss', 15.24), scored('b', 'loss', 15.16), scored('c', 'loss', 15.21)]
    assert.equal(bestTrialId(trials), 'b')
  })

  it('picks the HIGHEST value on a higher-is-better metric', () => {
    // The reported contradiction: this minimised unconditionally, so on a reward or
    // accuracy job it returned the worst trial — which then took palette slot 0, the
    // "Winning trial" tag and first place in the ascending sort, while the radar drew
    // it collapsed at the centre and the real winner at the rim.
    const trials = [scored('a', 'reward', 0.4), scored('b', 'reward', 0.9), scored('c', 'reward', 0.6)]
    assert.equal(bestTrialId(trials), 'b')
    assert.equal(bestTrialId([scored('a', 'accuracy', 0.71), scored('b', 'accuracy', 0.93)]), 'b')
  })

  it('agrees with the radar on direction for every metric it scores', () => {
    for (const name of ['loss', 'train_loss', 'total_time', 'reward', 'accuracy', 'f1']) {
      const worse = isLowerBetter(name) ? 9 : 1
      const better = isLowerBetter(name) ? 1 : 9
      assert.equal(bestTrialId([scored('w', name, worse), scored('b', name, better)]), 'b', name)
    }
  })

  it('uses the loss fallback, so a trial naming no metric can still win', () => {
    const withLoss = { id: 'a', status: 'completed', metrics: { loss: 1.5 } }
    const worse = { id: 'b', status: 'completed', metrics: { loss: 9.5 } }
    assert.equal(bestTrialId([worse, withLoss]), 'a')
  })

  it('ignores runs with no usable metric', () => {
    const noMetric = { id: 'a', status: 'completed', metric: undefined, metrics: {} }
    const nan = scored('b', 'loss', Number.NaN)
    assert.equal(bestTrialId([noMetric, nan, scored('c', 'loss', 15.2)]), 'c')
    assert.equal(bestTrialId([]), undefined)
  })
})
