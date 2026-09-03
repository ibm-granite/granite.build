/**
 * Pure-logic coverage for the AutoTuneX v0.3.5 pagination + adapter helpers.
 * These live in `api/autotunexAdapters.ts` (re-exported from `api/autotunex.ts`)
 * specifically so this file can `require()` them directly under plain
 * `node --test`, without going through the Next.js/tsc alias resolver.
 *
 * Usage: node --test tests/autotunex-adapters.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')

const { pageQuery, toListResult, collectPages, adaptTrial, adaptJob, adaptConfiguration, adaptSuggestion, adaptAsset } = require(
  '../../../packages/ui-core/api/autotunexAdapters.ts'
)

describe('pageQuery', () => {
  it('maps page/pageSize to offset/limit and defaults scope to own', () => {
    assert.deepEqual(pageQuery({ page: 1, pageSize: 20 }), { limit: 20, offset: 0, scope: 'own' })
  })

  it('computes offset from page > 1', () => {
    assert.deepEqual(pageQuery({ page: 3, pageSize: 20 }), { limit: 20, offset: 40, scope: 'own' })
  })

  it('clamps limit to 100 even when pageSize is larger', () => {
    assert.deepEqual(pageQuery({ page: 2, pageSize: 150 }), { limit: 100, offset: 150, scope: 'own' })
  })

  it('passes through q and an explicit scope when given', () => {
    assert.deepEqual(pageQuery({ page: 1, pageSize: 10, q: 'foo', scope: 'all' }), {
      limit: 10,
      offset: 0,
      scope: 'all',
      q: 'foo',
    })
  })

  it('omits q when not provided', () => {
    const params = pageQuery({ page: 1, pageSize: 10 })
    assert.equal('q' in params, false)
  })
})

describe('toListResult', () => {
  it('reshapes an {items,total,limit,offset} envelope into {items,total}, mapping each item', () => {
    const envelope = {
      items: [{ id: 'a' }, { id: 'b' }],
      total: 2,
      limit: 20,
      offset: 0,
    }
    const result = toListResult(envelope, (raw) => ({ id: raw.id.toUpperCase() }))
    assert.deepEqual(result, { items: [{ id: 'A' }, { id: 'B' }], total: 2 })
  })

  it('defaults to an empty list/zero total when items/total are missing', () => {
    assert.deepEqual(toListResult({}, (raw) => raw), { items: [], total: 0 })
  })
})

describe('adaptTrial', () => {
  it('flattens metric/metrics (no nested score object)', () => {
    const raw = {
      id: 't1',
      job_id: 'j1',
      status: 'completed',
      config: { lr: 0.001 },
      metric: 'accuracy',
      metrics: { accuracy: 0.87, loss: 0.12 },
      created_at: '2026-01-01T00:00:00Z',
      updated_at: '2026-01-02T00:00:00Z',
    }
    const trial = adaptTrial(raw)
    assert.equal(trial.metric, 'accuracy')
    assert.deepEqual(trial.metrics, { accuracy: 0.87, loss: 0.12 })
    assert.equal('score' in trial, false)
  })

  it('defaults missing fields (metric undefined, metrics/config to {})', () => {
    const trial = adaptTrial({ id: 't2', job_id: 'j1', status: 'pending' })
    assert.equal(trial.metric, undefined)
    assert.deepEqual(trial.metrics, {})
    assert.deepEqual(trial.config, {})
  })
})

describe('adaptJob + toListResult (getJobs reshaping)', () => {
  it('reshapes a {items,total} jobs envelope into ListResult<TuningJob>', () => {
    const envelope = {
      items: [
        {
          id: 'j1',
          user_id: 'u1',
          status: 'running',
          seed: 42,
          config_id: 'c1',
          config_name: 'cfg-1',
          dataset_id: 'd1',
          dataset: 'ds-1',
          model: 'granite-3b',
          experiment_name: 'exp-1',
          user: 'alice',
          created_at: '2026-01-01T00:00:00Z',
          updated_at: '2026-01-01T01:00:00Z',
        },
      ],
      total: 1,
      limit: 20,
      offset: 0,
    }
    const result = toListResult(envelope, adaptJob)
    assert.equal(result.total, 1)
    assert.equal(result.items.length, 1)
    assert.equal(result.items[0].id, 'j1')
    assert.equal(result.items[0].status, 'running')
    assert.equal(result.items[0].experiment_name, 'exp-1')
  })
})

describe('adaptConfiguration', () => {
  it('maps associated_jobs refs and defaults rl_tuner_type/config_data to null', () => {
    const raw = {
      id: 'cfg-1',
      user_id: 'u1',
      name: 'My Config',
      tuner_type: 'lora',
      associated_jobs: [{ id: 'j1', experiment_name: 'exp-1', status: 'running' }],
    }
    const config = adaptConfiguration(raw)
    assert.equal(config.rl_tuner_type, null)
    assert.equal(config.config_data, null)
    assert.deepEqual(config.associated_jobs, [{ id: 'j1', experiment_name: 'exp-1', status: 'running' }])
  })

  it('defaults associated_jobs to [] when absent/non-array', () => {
    const config = adaptConfiguration({ id: 'cfg-2', user_id: 'u1', name: 'x', tuner_type: 'lora' })
    assert.deepEqual(config.associated_jobs, [])
  })
})

describe('adaptSuggestion', () => {
  it('renames dataset_type->dataset_format and algorithm->tuning_type', () => {
    const raw = {
      dataset_format: 'preference_pairs',
      tuning_type: 'lora',
      confidence: 0.92,
      column_mapping: { prompt: 'input', chosen: 'output' },
      column_confidence: { prompt: 0.99 },
      reasoning: 'Detected pairwise preference columns.',
    }
    const suggestion = adaptSuggestion(raw)
    assert.equal(suggestion.dataset_format, 'preference_pairs')
    assert.equal(suggestion.tuning_type, 'lora')
    assert.deepEqual(suggestion.column_mapping, { prompt: 'input', chosen: 'output' })
    assert.deepEqual(suggestion.column_confidence, { prompt: 0.99 })
    assert.equal(suggestion.reasoning, 'Detected pairwise preference columns.')
  })

  it('defaults column_mapping to {} and leaves reasoning/column_confidence undefined when absent', () => {
    const suggestion = adaptSuggestion({ dataset_format: 'unknown', tuning_type: 'lora', confidence: 0.1 })
    assert.deepEqual(suggestion.column_mapping, {})
    assert.equal(suggestion.reasoning, undefined)
    assert.equal(suggestion.column_confidence, undefined)
  })
})

describe('adaptAsset', () => {
  it('maps a full result-report row (AssetSummary)', () => {
    const raw = {
      filename: 'adapters.safetensors',
      size: 13_000_000,
      modified: '2026-08-21T10:02:00Z',
      path: 'trial_0/adapters.safetensors',
      file_hash: 'abc123',
      published: true,
    }
    assert.deepEqual(adaptAsset(raw), {
      filename: 'adapters.safetensors',
      size: 13_000_000,
      modified: '2026-08-21T10:02:00Z',
      path: 'trial_0/adapters.safetensors',
      file_hash: 'abc123',
      published: true,
    })
  })

  it('defaults the nullable fields to null and size to 0 when absent', () => {
    const asset = adaptAsset({ filename: 'training.log' })
    assert.equal(asset.filename, 'training.log')
    assert.equal(asset.size, 0)
    assert.equal(asset.modified, null)
    assert.equal(asset.path, null)
    assert.equal(asset.file_hash, null)
    assert.equal(asset.published, null)
  })

  it('preserves published:false (only null/undefined fall through to null)', () => {
    assert.equal(adaptAsset({ filename: 'x', published: false }).published, false)
  })
})

describe('collectPages', () => {
  // Builds a fake page-fetcher over a fixed row array, recording every call so a
  // test can assert how many requests the loop actually spent. `reportedTotal`
  // lets a test lie about the total the way a live server can — trials are being
  // appended while a job runs, so the envelope's count goes stale mid-drain.
  function fakeFetcher(rows, reportedTotal = rows.length) {
    const calls = []
    const fetchPage = async (limit, offset) => {
      calls.push({ limit, offset })
      return { items: rows.slice(offset, offset + limit), total: reportedTotal }
    }
    return { fetchPage, calls }
  }

  const identity = (r) => r

  it('returns a single short page and stops after one call', async () => {
    const { fetchPage, calls } = fakeFetcher([{ id: 'a' }, { id: 'b' }])
    const out = await collectPages(fetchPage, identity, 100)
    assert.deepEqual(out, [{ id: 'a' }, { id: 'b' }])
    assert.equal(calls.length, 1)
    assert.deepEqual(calls[0], { limit: 100, offset: 0 })
  })

  it('stops on the total check after one call when the page is exactly full', async () => {
    // 3 rows, limit 3, total 3: the short-page condition would not fire, so
    // without the total check this would spend a second request to learn the
    // collection is exhausted.
    const { fetchPage, calls } = fakeFetcher([{ id: 'a' }, { id: 'b' }, { id: 'c' }])
    const out = await collectPages(fetchPage, identity, 3)
    assert.equal(out.length, 3)
    assert.equal(calls.length, 1)
  })

  it('fetches the next page when a full page has a larger total', async () => {
    const rows = [{ id: 'a' }, { id: 'b' }, { id: 'c' }, { id: 'd' }, { id: 'e' }]
    const { fetchPage, calls } = fakeFetcher(rows)
    const out = await collectPages(fetchPage, identity, 2)
    assert.deepEqual(out.map((r) => r.id), ['a', 'b', 'c', 'd', 'e'])
    // 2 + 2 + 1: the third page is short, which ends the loop.
    assert.equal(calls.length, 3)
    assert.deepEqual(calls.map((c) => c.offset), [0, 2, 4])
  })

  it('terminates on an empty page even when total is overstated', async () => {
    // A server claiming 500 rows but serving 2 must not spin forever. The
    // short-page condition (0 < limit) is what makes the loop total here —
    // the total check never fires.
    const { fetchPage, calls } = fakeFetcher([{ id: 'a' }, { id: 'b' }], 500)
    const out = await collectPages(fetchPage, identity, 2)
    assert.equal(out.length, 2)
    assert.equal(calls.length, 2)
  })

  it('maps every row through adapt', async () => {
    const { fetchPage } = fakeFetcher([{ id: 'a' }, { id: 'b' }])
    const out = await collectPages(fetchPage, (r) => r.id.toUpperCase(), 100)
    assert.deepEqual(out, ['A', 'B'])
  })

  it('treats a missing items array as an empty page', async () => {
    const out = await collectPages(async () => ({ total: 0 }), identity, 100)
    assert.deepEqual(out, [])
  })

  it('keeps paging when a full page omits total', async () => {
    // `total` is optional on the envelope. A missing one must disable the
    // optimization, not trigger it — otherwise the drain stops after page 1 and
    // silently returns partial data.
    const rows = [{ id: 'a' }, { id: 'b' }, { id: 'c' }]
    const calls = []
    const fetchPage = async (limit, offset) => {
      calls.push({ limit, offset })
      return { items: rows.slice(offset, offset + limit) }
    }
    const out = await collectPages(fetchPage, identity, 2)
    assert.deepEqual(out.map((r) => r.id), ['a', 'b', 'c'])
    assert.equal(calls.length, 2)
  })
})
