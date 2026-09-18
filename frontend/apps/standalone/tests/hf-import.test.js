/**
 * Tests for the HuggingFace dataset-import decision logic.
 *
 * These functions exist as a separate module precisely so they can be tested:
 * there is no jsdom harness in this app, so anything left inside
 * HfImportModal.tsx has no automated coverage at all. Each function here is one
 * the flow gets silently wrong if it regresses:
 *
 *  - A name that keeps a '/', a '\' or a '..' is a 422 at import time.
 *  - An empty column_mapping is a 422 on the preview probe, which would kill the
 *    mapping form before it ever renders (the field is min_length=1 server-side).
 *  - A mapping counted as complete when it is not lets the import fire against a
 *    partial projection.
 *
 * Usage: node --test tests/hf-import.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')

const {
  deriveDatasetName,
  isDatasetNameValid,
  suffixWithRevision,
  probeMapping,
  isMappingComplete,
} = require('../app/dashboard/autotunex/start-tuning/steps/hfImport.ts')

describe('deriveDatasetName', () => {
  it('takes the last path segment of a repo id', () => {
    assert.equal(deriveDatasetName('vicgalle/alpaca-gpt4'), 'alpaca-gpt4')
  })

  it('accepts a bare name with no owner', () => {
    assert.equal(deriveDatasetName('alpaca'), 'alpaca')
  })

  it('ignores a trailing slash rather than returning an empty name', () => {
    assert.equal(deriveDatasetName('owner/name/'), 'name')
  })

  it('replaces each run of illegal characters with a single dash', () => {
    assert.equal(deriveDatasetName('owner/na me!!x'), 'na-me-x')
  })

  it('collapses dot runs, because ".." is rejected by the server', () => {
    assert.equal(deriveDatasetName('owner/a..b'), 'a.b')
    assert.ok(!deriveDatasetName('owner/a...b').includes('..'))
  })

  it('falls back to a usable name when nothing survives sanitization', () => {
    assert.equal(deriveDatasetName('owner/...'), 'dataset')
    assert.equal(deriveDatasetName(''), 'dataset')
  })

  it('caps the name at the server limit of 255', () => {
    const derived = deriveDatasetName(`owner/${'a'.repeat(300)}`)
    assert.equal(derived.length, 255)
  })

  it('always produces a name the server will accept', () => {
    for (const repoId of ['vicgalle/alpaca-gpt4', 'owner/a..b', 'owner/na me!!x', 'owner/...', `owner/${'a'.repeat(300)}`]) {
      assert.ok(isDatasetNameValid(deriveDatasetName(repoId)), `invalid name derived from ${repoId}`)
    }
  })
})

describe('isDatasetNameValid', () => {
  it('accepts an ordinary name', () => {
    assert.equal(isDatasetNameValid('alpaca-gpt4'), true)
  })

  it('rejects the three things the server rejects', () => {
    assert.equal(isDatasetNameValid('a/b'), false)
    assert.equal(isDatasetNameValid('a\\b'), false)
    assert.equal(isDatasetNameValid('a..b'), false)
  })

  it('rejects empty and over-long names', () => {
    assert.equal(isDatasetNameValid(''), false)
    assert.equal(isDatasetNameValid('a'.repeat(256)), false)
    assert.equal(isDatasetNameValid('a'.repeat(255)), true)
  })
})

describe('suffixWithRevision', () => {
  it('appends the short SHA', () => {
    assert.equal(suffixWithRevision('alpaca-gpt4', 'abc1234def5678'), 'alpaca-gpt4-abc1234')
  })

  it('leaves the name alone when there is no revision to suffix', () => {
    assert.equal(suffixWithRevision('alpaca', ''), 'alpaca')
  })

  it('keeps the suffixed name within the 255 limit and still valid', () => {
    const suffixed = suffixWithRevision('a'.repeat(255), 'abc1234def')
    assert.ok(suffixed.length <= 255)
    assert.ok(suffixed.endsWith('-abc1234'))
    assert.equal(isDatasetNameValid(suffixed), true)
  })
})

describe('probeMapping', () => {
  it('maps the first required column with a blank source', () => {
    assert.deepEqual(probeMapping(['input', 'output']), { input: '' })
    assert.deepEqual(probeMapping(['prompt', 'chosen']), { prompt: '' })
  })

  it('is never empty, because column_mapping is min_length=1 server-side', () => {
    // An empty dict 422s the probe, which is the request that fetches the very
    // column list the mapping form needs -- the flow would dead-end at stage 3.
    assert.equal(Object.keys(probeMapping([])).length, 1)
    assert.deepEqual(probeMapping([]), { input: '' })
  })
})

describe('isMappingComplete', () => {
  it('is true only when every required column has a source', () => {
    assert.equal(isMappingComplete({ input: 'instruction', output: 'output' }, ['input', 'output']), true)
    assert.equal(isMappingComplete({ input: 'instruction' }, ['input', 'output']), false)
  })

  it('treats a blank source as unmapped', () => {
    assert.equal(isMappingComplete({ input: 'instruction', output: '' }, ['input', 'output']), false)
  })

  it('is false when there are no required columns at all', () => {
    // Not vacuously true: with no known targets there is no valid import request
    // to build (column_mapping is min_length=1), so the import must stay blocked.
    assert.equal(isMappingComplete({}, []), false)
  })
})
