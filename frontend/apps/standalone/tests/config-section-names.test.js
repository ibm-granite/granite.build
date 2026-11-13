// Copyright Granite.Build Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * Regression test for the Step 2 "Configure" config editor's tab list.
 *
 * The "General" tab is a synthetic UI-only section rendered from the real
 * training_config + tune_config sections; it is never a top-level key of the
 * template returned by getConfigurationTemplate(). A prior version filtered the
 * basic-mode tabs against the template's real keys, which silently dropped
 * "General" and landed the user straight on "Tuners" — diverging from the source
 * AutoTuneX form (which lists "General" first, unconditionally). This locks the
 * behavior back in.
 *
 * Usage: node --test tests/config-section-names.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')

const { computeSectionNames } = require('../../../packages/ui-core/lib/autotunex/configSections.ts')

// Real top-level keys as returned by the autotune core template
// (autotune.utils.get_autotune_config) — note: NO `general_config`.
const TEMPLATE_KEYS = ['tune_config', 'tuners_config', 'training_config', 'tokenizer_config', 'tuners_rl_config', 'training_rl_config']

describe('computeSectionNames — basic mode', () => {
  it('keeps the synthetic "General" tab first for an SFT template that lacks general_config', () => {
    const sections = computeSectionNames({
      mode: false,
      trainingMode: 'offline_tuning',
      presetGoal: 'sft',
      allSectionKeys: TEMPLATE_KEYS,
    })
    // The bug produced ['tuners_config'] — General must be present and first.
    assert.deepEqual(sections, ['general_config', 'tuners_config'])
  })

  it('includes RL tuners for offline non-SFT goals', () => {
    const sections = computeSectionNames({
      mode: false,
      trainingMode: 'offline_tuning',
      presetGoal: 'offline_rl',
      allSectionKeys: TEMPLATE_KEYS,
    })
    assert.deepEqual(sections, ['general_config', 'tuners_config', 'tuners_rl_config'])
  })

  it('shows General + RL tuners (no plain tuners) for online RL', () => {
    const sections = computeSectionNames({
      mode: false,
      trainingMode: 'online_tuning',
      presetGoal: 'online_rl',
      allSectionKeys: TEMPLATE_KEYS,
    })
    assert.deepEqual(sections, ['general_config', 'tuners_rl_config'])
  })

  it('still hides a real section the template genuinely omits, while keeping General', () => {
    const sections = computeSectionNames({
      mode: false,
      trainingMode: 'offline_tuning',
      presetGoal: 'sft',
      allSectionKeys: ['tune_config', 'training_config'], // no tuners_config
    })
    assert.deepEqual(sections, ['general_config'])
  })
})

describe('computeSectionNames — advanced mode', () => {
  it('lists the real template keys in canonical order and never injects general_config (SFT)', () => {
    const sections = computeSectionNames({
      mode: true,
      trainingMode: 'offline_tuning',
      presetGoal: 'sft',
      allSectionKeys: TEMPLATE_KEYS,
    })
    // SFT drops tuners_rl_config; offline drops training_rl_config; no general_config.
    assert.deepEqual(sections, ['tune_config', 'training_config', 'tokenizer_config', 'tuners_config'])
  })

  it('drops plain tuners and keeps RL sections for online RL', () => {
    const sections = computeSectionNames({
      mode: true,
      trainingMode: 'online_tuning',
      presetGoal: 'online_rl',
      allSectionKeys: TEMPLATE_KEYS,
    })
    assert.deepEqual(sections, ['tune_config', 'training_config', 'tokenizer_config', 'training_rl_config', 'tuners_rl_config'])
  })
})
