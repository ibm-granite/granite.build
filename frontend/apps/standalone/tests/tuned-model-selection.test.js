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
  tunedModelDetails,
  tunedModelLabel,
  tunedModelLabels,
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

  it('reports a missing repository for a 404, without offering a retry', () => {
    const result = tunedModelCheckFailure(404)

    assert.match(result.reason, /repository no longer exists/)
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

  it('shows only the experiment name', () => {
    assert.equal(tunedModelLabel(base), 'sft-granite')
  })

  it('falls back to the repo id when the job has no experiment name', () => {
    assert.equal(tunedModelLabel({ ...base, experiment_name: '' }), 'ibm-research/autotunex_aaaa0001')
  })
})

describe('tunedModelDetails', () => {
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

  it('lists the base model, tuning kind and repo id, in that order', () => {
    assert.deepEqual(tunedModelDetails(base), [
      { label: 'Base model', value: 'ibm-granite/granite-4.0-h-micro' },
      { label: 'Tuning kind', value: 'sft' },
      { label: 'Repo id', value: 'ibm-research/autotunex_aaaa0001' },
    ])
  })

  it('prefers the RL tuner as the tuning kind', () => {
    const kind = tunedModelDetails({ ...base, tuning_type: 'none', rl_tuner_type: 'grpo' })[1]

    assert.deepEqual(kind, { label: 'Tuning kind', value: 'grpo' })
  })

  it('omits the tuning kind when the job recorded none', () => {
    const labels = tunedModelDetails({ ...base, tuning_type: null }).map((d) => d.label)

    assert.deepEqual(labels, ['Base model', 'Repo id'])
  })
})

describe('tunedModelLabels', () => {
  const model = (job_id, experiment_name, repo_id) => ({
    job_id,
    repo_id,
    model_source: 'huggingface',
    experiment_name,
    base_model: 'ibm-granite/granite-4.0-h-micro',
    tuning_type: 'sft',
    rl_tuner_type: null,
    finished_at: null,
    user: 'u',
  })

  it('keeps a unique experiment name as it is', () => {
    const labels = tunedModelLabels([
      model('j1', 'sft-granite', 'ibm-research/autotunex_a69082b7'),
      model('j2', 'grpo-math', 'ibm-research/autotunex_1f3c09aa'),
    ])

    assert.deepEqual([labels.get('j1'), labels.get('j2')], ['sft-granite', 'grpo-math'])
  })

  it('suffixes colliding names with the short repo id', () => {
    const labels = tunedModelLabels([
      model('j1', 'grpo-math', 'ibm-research/autotunex_a69082b7'),
      model('j2', 'grpo-math', 'ibm-research/autotunex_1f3c09aa'),
      model('j3', 'sft-granite', 'ibm-research/autotunex_77aa0001'),
    ])

    assert.deepEqual(
      [labels.get('j1'), labels.get('j2'), labels.get('j3')],
      ['grpo-math · a69082b7', 'grpo-math · 1f3c09aa', 'sft-granite'],
    )
  })

  it('keeps a repo name without the autotunex_ prefix whole in the suffix', () => {
    const labels = tunedModelLabels([
      model('j1', 'grpo-math', 'ibm-research/grpo-run-a'),
      model('j2', 'grpo-math', 'ibm-research/grpo-run-b'),
    ])

    assert.deepEqual([labels.get('j1'), labels.get('j2')], ['grpo-math · grpo-run-a', 'grpo-math · grpo-run-b'])
  })
})
