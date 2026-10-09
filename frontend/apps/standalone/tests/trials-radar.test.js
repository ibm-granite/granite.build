/**
 * Tests for the trial ranking helpers in trialsRadar.ts: metric direction
 * (`isLowerBetter`), the metric a trial and a job are judged on, and the
 * best-first order the trials table and Compare share.
 *
 * Usage: node --test tests/trials-radar.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')

const {
  isLowerBetter,
  toFeatureLabel,
  primaryMetric,
  bestTrialId,
  jobMetric,
  rankBestFirst,
  scoreOn,
} = require('../../../packages/ui-core/components/autotunex/trials/trialsRadar.ts')

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

describe('toFeatureLabel', () => {
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

  it('agrees with isLowerBetter on direction for every metric it scores', () => {
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

describe('jobMetric', () => {
  it('is the first metric any trial names', () => {
    assert.equal(jobMetric([{ id: 'a', metrics: {} }, { id: 'b', metric: 'reward', metrics: {} }]), 'reward')
  })

  it('falls back to loss when no trial names one', () => {
    assert.equal(jobMetric([{ id: 'a', metrics: { loss: 1 } }]), 'loss')
    assert.equal(jobMetric([]), 'loss')
  })
})

describe('rankBestFirst', () => {
  const scored = (id, metric, value) => ({ id, status: 'completed', metric, metrics: { [metric]: value } })
  const ids = (trials) => rankBestFirst(trials).map((t) => t.id)

  it('puts the winning trial first on a higher-is-better metric', () => {
    // The table and Compare sorted ascending unconditionally, so on a reward job
    // the "Winning trial" sat last under an ascending arrow.
    const trials = [scored('a', 'reward', 0.4), scored('b', 'reward', 0.9), scored('c', 'reward', 0.6)]
    assert.deepEqual(ids(trials), ['b', 'c', 'a'])
    assert.equal(rankBestFirst(trials)[0].id, bestTrialId(trials))
  })

  it('puts the lowest first on a lower-is-better metric', () => {
    const trials = [scored('a', 'loss', 15.24), scored('b', 'loss', 15.16), scored('c', 'loss', 15.21)]
    assert.deepEqual(ids(trials), ['b', 'c', 'a'])
  })

  it('sinks trials with no usable value, keeping their order', () => {
    const none = { id: 'x', status: 'running', metrics: {} }
    const nan = scored('y', 'loss', Number.NaN)
    assert.deepEqual(ids([none, scored('a', 'loss', 2), nan, scored('b', 'loss', 1)]), ['b', 'a', 'x', 'y'])
  })

  it('does not mutate the input', () => {
    const trials = [scored('a', 'loss', 2), scored('b', 'loss', 1)]
    rankBestFirst(trials)
    assert.deepEqual(trials.map((t) => t.id), ['a', 'b'])
  })
})

describe('a trial reporting only the fallback metric', () => {
  // A reward-scored job where one trial has `metric: 'reward'` but only a loss so
  // far. primaryMetric falls back to its loss, and judging direction from that
  // trial made the job lower-is-better -- crowning the WORST reward, and comparing
  // a loss against rewards.
  const trials = [
    { id: 'a', status: 'completed', metric: 'reward', metrics: { loss: 0.9 } },
    { id: 'b', status: 'completed', metric: 'reward', metrics: { reward: 0.4 } },
    { id: 'c', status: 'completed', metric: 'reward', metrics: { reward: 0.8 } },
  ]

  it('is not the winner, and does not flip the direction', () => {
    assert.equal(bestTrialId(trials), 'c')
  })

  it('ranks unscored, after the trials scored on the job metric', () => {
    assert.deepEqual(rankBestFirst(trials).map((t) => t.id), ['c', 'b', 'a'])
  })

  it('has no score on the job metric -- the table cell shows none, not its loss', () => {
    // The trials table fills its metric column with scoreOn, so the cell and the
    // order above agree: 'a' sinks unscored rather than showing 0.9 under "Reward".
    assert.equal(scoreOn(trials[0], jobMetric(trials)), null)
    assert.equal(scoreOn(trials[2], jobMetric(trials)), 0.8)
  })
})

describe('the trials table metric column', () => {
  const fs = require('node:fs')
  const path = require('node:path')
  const src = fs.readFileSync(
    path.join(__dirname, '..', '..', '..', 'packages', 'ui-core', 'components', 'autotunex', 'trials', 'TrialsTable.tsx'),
    'utf8'
  )

  it('is labelled with the job metric, not a fixed "Loss"', () => {
    assert.match(src, /h\.key === 'loss' \? \{ \.\.\.h, header: toFeatureLabel\(metric\) \}/)
    assert.match(src, /headers=\{tableHeaders\}/)
  })

  it('shows the value the ranking used', () => {
    assert.match(src, /loss: scoreOn\(t, metric\) \?\? undefined,/)
  })
})

