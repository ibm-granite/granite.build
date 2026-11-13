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
 * AUTOTUNEX_ADMIN_DEFAULT_SCOPE picks where an admin's Mine/All list toggle
 * starts. Anything but the exact string "all" must fall back to "own", so a
 * typo can never widen the view.
 *
 * Usage: node --test tests/admin-default-scope.test.js
 */

const { describe, it, afterEach } = require('node:test')
const assert = require('node:assert/strict')

const { adminDefaultScope } = require('../../../packages/ui-core/api/client.ts')

const KEY = 'AUTOTUNEX_ADMIN_DEFAULT_SCOPE'
const saved = process.env[KEY]

describe('adminDefaultScope', () => {
  afterEach(() => {
    if (saved === undefined) delete process.env[KEY]
    else process.env[KEY] = saved
  })

  it('is own when unset', () => {
    delete process.env[KEY]
    assert.equal(adminDefaultScope(), 'own')
  })

  it('is own when empty (next.config.ts bakes unset as "")', () => {
    process.env[KEY] = ''
    assert.equal(adminDefaultScope(), 'own')
  })

  it('is all when set to all', () => {
    process.env[KEY] = 'all'
    assert.equal(adminDefaultScope(), 'all')
  })

  it('is own for any other value', () => {
    for (const v of ['own', 'ALL', ' all', 'everyone']) {
      process.env[KEY] = v
      assert.equal(adminDefaultScope(), 'own', `value ${JSON.stringify(v)}`)
    }
  })
})
