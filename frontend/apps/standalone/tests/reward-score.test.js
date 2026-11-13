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
 * Regression test for reading a reward function's test-run return value.
 *
 * The backend passes the function's return value through as-is (int, float, str,
 * bool, list, dict), and verl itself accepts a float or a dict with a "score"
 * key. The reward step assumed a number and called `.toFixed(3)` on it, so a
 * `return {"score": 0.9}` -- the form the shipped template's own docstring
 * mentions -- crashed the step, and a string return counted as a passing test.
 *
 * Usage: node --test tests/reward-score.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')

const { rewardScore, rewardScoreError } = require('../../../packages/ui-core/lib/autotunex/rewardScore.ts')

describe('rewardScore', () => {
  it('takes a finite number as is', () => {
    assert.equal(rewardScore(0.9), 0.9)
    assert.equal(rewardScore(0), 0)
  })

  it('takes the numeric score out of a dict, as verl does', () => {
    assert.equal(rewardScore({ score: 0.9, acc: 1 }), 0.9)
  })

  it('reads a bool as 0 or 1, as Python does', () => {
    assert.equal(rewardScore(true), 1)
    assert.equal(rewardScore(false), 0)
  })

  it('rejects anything verl could not score', () => {
    for (const v of ['0.9', [1], { score: 'x' }, {}, null, undefined, Number.NaN, Infinity]) {
      assert.equal(rewardScore(v), null, `expected null for ${JSON.stringify(v)}`)
    }
  })
})

describe('rewardScoreError', () => {
  it('names the Python type that came back', () => {
    assert.match(rewardScoreError('abc'), /got str/)
    assert.match(rewardScoreError([1]), /got list/)
    assert.match(rewardScoreError({ s: 1 }), /got dict/)
    assert.match(rewardScoreError(null), /got None/)
  })
})
