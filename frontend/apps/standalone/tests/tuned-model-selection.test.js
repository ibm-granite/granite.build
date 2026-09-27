/**
 * Pick-time rules for the Start Tuning wizard's "My tuned models" source: a
 * tuned output is usable as a base only when its HF repo root holds a full model.
 *
 * Usage: node --test tests/tuned-model-selection.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')

const {
  checkTunedModelAssets,
  tunedModelCheckFailure,
  tunedModelLabel,
} = require('../app/dashboard/autotunex/start-tuning/tunedModelSelection.ts')

const asset = (path) => ({ path })

describe('checkTunedModelAssets', () => {
  it('accepts a repo with config.json at its root', () => {
    assert.deepEqual(checkTunedModelAssets([asset('config.json'), asset('model.safetensors')]), { ok: true })
  })

  it('rejects an adapter repo even if it also has a root config.json', () => {
    const result = checkTunedModelAssets([asset('adapter_config.json'), asset('config.json')])

    assert.equal(result.ok, false)
    assert.match(result.reason, /adapter/i)
    assert.equal(result.retryable, false)
  })

  it('rejects a repo whose model sits in a subfolder', () => {
    const result = checkTunedModelAssets([asset('sft-granite/config.json'), asset('results/summary.zip')])

    assert.equal(result.ok, false)
    assert.match(result.reason, /predates/)
  })

  it('rejects an empty listing', () => {
    assert.equal(checkTunedModelAssets([]).ok, false)
  })
})

describe('tunedModelCheckFailure', () => {
  it('reports a failed build for a 409, without offering a retry', () => {
    const result = tunedModelCheckFailure(409)

    assert.match(result.reason, /did not succeed/)
    assert.equal(result.retryable, false)
  })

  it('offers a retry for anything else', () => {
    assert.equal(tunedModelCheckFailure(502).retryable, true)
    assert.equal(tunedModelCheckFailure(undefined).retryable, true)
  })
})

describe('tunedModelLabel', () => {
  const base = {
    job_id: 'j1',
    repo_id: 'ibm-research/autotunex_aaaa0001',
    model_source: 'huggingface',
    experiment_name: 'sft-granite',
    base_model: 'ibm-granite/granite-4.0-h-micro',
    tuning_type: 'sft',
    rl_tuner_type: null,
    finished_at: null,
    user: 'u',
  }

  it('names the experiment, base model, tuning kind and repo', () => {
    assert.equal(
      tunedModelLabel(base),
      'sft-granite · ibm-granite/granite-4.0-h-micro · sft · ibm-research/autotunex_aaaa0001',
    )
  })

  it('prefers the RL tuner as the kind', () => {
    assert.match(tunedModelLabel({ ...base, tuning_type: 'none', rl_tuner_type: 'grpo' }), / · grpo · /)
  })
})
