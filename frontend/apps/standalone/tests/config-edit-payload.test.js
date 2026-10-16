/**
 * Regression test for the payload Step 2 builds when an edited configuration is
 * confirmed.
 *
 * `PUT /configurations/{id}` is a full replacement, and the edit path rebuilt
 * `config_data` from five hard-named sections. Any other stored section -- the
 * editable `tokenizer_config` in particular -- was missing from the PUT body and
 * so deleted from the stored configuration for every future job, and a Save-As
 * copy was created without it too.
 *
 * Usage: node --test tests/config-edit-payload.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')

const { configDataFromForm } = require('../../../packages/ui-core/lib/autotunex/configSections.ts')

const FORM = {
  name: 'my-config',
  tuner_type: 'lora',
  rl_tuner_type: null,
  tune_config: { num_samples: { default: 32 } },
  tuners_config: { lora: {} },
  training_config: { num_train_epochs: { default: 3 } },
  tokenizer_config: { additional_special_tokens: { default: ['<x>'] } },
}

describe('configDataFromForm', () => {
  it('keeps tokenizer_config', () => {
    assert.deepEqual(configDataFromForm(FORM).tokenizer_config, FORM.tokenizer_config)
  })

  it('keeps a section it has no name for', () => {
    const data = configDataFromForm({ ...FORM, future_config: { a: 1 } })
    assert.deepEqual(data.future_config, { a: 1 })
  })

  it('drops the form-only name and tuner fields', () => {
    const data = configDataFromForm(FORM)
    assert.equal('name' in data, false)
    assert.equal('tuner_type' in data, false)
    assert.equal('rl_tuner_type' in data, false)
  })

  it('omits an RL section the form left empty, as before', () => {
    const data = configDataFromForm({ ...FORM, training_rl_config: null, tuners_rl_config: undefined })
    assert.equal('training_rl_config' in data, false)
    assert.equal('tuners_rl_config' in data, false)
  })

  it('keeps an RL section that has content', () => {
    const rl = { grpo: {} }
    assert.deepEqual(configDataFromForm({ ...FORM, tuners_rl_config: rl }).tuners_rl_config, rl)
  })
})
