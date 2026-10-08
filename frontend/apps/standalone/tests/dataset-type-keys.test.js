/**
 * ALGORITHM_TO_DATASET_TYPE names keys in the backend's dataset-types response.
 * A key that drifts from the backend silently drops the algorithm's optional
 * columns: required columns fall back to a hardcoded table, optional ones have no
 * fallback. So check every mapped key against the vendored definition.
 *
 * Usage: node --test tests/dataset-type-keys.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const { ALGORITHM_DETAILS, ALGORITHM_TO_DATASET_TYPE } = require('../../../packages/ui-core/config/autotunexAlgorithms.ts')

const CATALOG = path.join(__dirname, '..', '..', '..', '..', 'autotunex', 'src', 'fm-tune', 'autotune', 'catalog.py')

describe('ALGORITHM_TO_DATASET_TYPE', () => {
  it('names only dataset types the backend defines', () => {
    const src = fs.readFileSync(CATALOG, 'utf8')
    const block = src.match(/^AutotuneDatasetTypes = \{([\s\S]*?)^\}/m)
    assert.ok(block, 'AutotuneDatasetTypes should exist in catalog.py')
    const defined = new Set([...block[1].matchAll(/^    "([a-z_]+)": \{/gm)].map((m) => m[1]))
    assert.ok(defined.size > 0, 'the definition should list its dataset types')
    for (const [algorithm, key] of Object.entries(ALGORITHM_TO_DATASET_TYPE)) {
      assert.ok(defined.has(key), `${algorithm} maps to ${key}, which catalog.py does not define (${[...defined].join(', ')})`)
    }
  })

  it('covers every algorithm the wizard offers', () => {
    for (const { id } of ALGORITHM_DETAILS) {
      assert.ok(ALGORITHM_TO_DATASET_TYPE[id], `${id} has no dataset type`)
    }
  })
})
