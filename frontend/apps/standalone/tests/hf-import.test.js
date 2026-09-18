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
  survivalSummary,
  defaultConfig,
  defaultTrainSplit,
  truncationNotice,
  problemDetail,
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

describe('survivalSummary', () => {
  it('shows nothing at all until the mapping is complete', () => {
    // A partial mapping's survival count is computed over the mapping's own keys,
    // so it reports a high number for the columns filled in so far. Showing it
    // would undo the reason the mapping screen cannot be skipped.
    const summary = survivalSummary({ sampled: 100, survived: 98, mappingComplete: false })
    assert.equal(summary.kind, 'hidden')
    assert.equal(summary.text, '')
  })

  it('blocks when no row survives the mapping', () => {
    const summary = survivalSummary({ sampled: 100, survived: 0, mappingComplete: true })
    assert.equal(summary.kind, 'blocked')
    assert.match(summary.text, /0 of 100/)
  })

  it('blocks with its own message when the split returned no rows', () => {
    const summary = survivalSummary({ sampled: 0, survived: 0, mappingComplete: true })
    assert.equal(summary.kind, 'blocked')
    assert.match(summary.text, /no rows/i)
  })

  it('warns but allows a partially surviving mapping', () => {
    // The alpaca-gpt4 case: its own column named `input` is empty in 54 of 100
    // rows, so an identity mapping retains 46. This number is the only thing that
    // reveals it, so it must be present and must not read as success.
    const summary = survivalSummary({ sampled: 100, survived: 46, mappingComplete: true })
    assert.equal(summary.kind, 'warning')
    assert.match(summary.text, /46 of 100/)
  })

  it('confirms a fully surviving mapping', () => {
    const summary = survivalSummary({ sampled: 100, survived: 100, mappingComplete: true })
    assert.equal(summary.kind, 'ok')
    assert.match(summary.text, /100/)
  })
})

describe('defaultConfig', () => {
  it('prefers the config named "default"', () => {
    assert.equal(defaultConfig(['en', 'default', 'fr']), 'default')
  })

  it('falls back to the first config', () => {
    assert.equal(defaultConfig(['en', 'fr']), 'en')
  })

  it('returns empty for no configs', () => {
    assert.equal(defaultConfig([]), '')
  })
})

describe('defaultTrainSplit', () => {
  it('prefers the split named "train"', () => {
    assert.equal(defaultTrainSplit(['test', 'train']), 'train')
  })

  it('falls back to the first split', () => {
    assert.equal(defaultTrainSplit(['test', 'validation']), 'test')
  })

  it('returns empty for no splits', () => {
    assert.equal(defaultTrainSplit([]), '')
  })
})

describe('truncationNotice', () => {
  it('is silent when nothing was truncated', () => {
    assert.equal(truncationNotice(null, 50000), null)
    assert.equal(truncationNotice(undefined, 50000), null)
    assert.equal(
      truncationNotice({ train_original_rows: 100, train_retained_rows: 100 }, 50000),
      null
    )
  })

  it('is silent when the counts are absent', () => {
    assert.equal(truncationNotice({ column_mapping: { input: 'instruction' } }, 50000), null)
  })

  it('reports a truncated train split with both counts and the cap', () => {
    const notice = truncationNotice(
      { train_original_rows: 120000, train_retained_rows: 50000 },
      50000
    )
    assert.match(notice, /50,000/)
    assert.match(notice, /120,000/)
    assert.match(notice, /train/)
  })

  it('reports a truncated validation split even when train was untouched', () => {
    // Checking only the train pair would stay silent here.
    const notice = truncationNotice(
      {
        train_original_rows: 100,
        train_retained_rows: 100,
        validation_original_rows: 80000,
        validation_retained_rows: 50000,
      },
      50000
    )
    assert.match(notice, /validation/)
    assert.ok(!notice.includes('train'))
  })

  it('reports both splits when both were truncated', () => {
    const notice = truncationNotice(
      {
        train_original_rows: 120000,
        train_retained_rows: 50000,
        validation_original_rows: 80000,
        validation_retained_rows: 50000,
      },
      50000
    )
    assert.match(notice, /train/)
    assert.match(notice, /validation/)
  })
})

describe('problemDetail', () => {
  it('returns the backend authored detail', () => {
    // The two 503s share a title and differ only here, so this string is the only
    // thing that tells "not converted yet" from "disabled in this deployment".
    const err = { response: { data: { detail: 'HuggingFace has not converted x/y yet.' } } }
    assert.equal(problemDetail(err, 'fallback'), 'HuggingFace has not converted x/y yet.')
  })

  it('falls back when there is no usable detail', () => {
    assert.equal(problemDetail(null, 'fallback'), 'fallback')
    assert.equal(problemDetail({}, 'fallback'), 'fallback')
    assert.equal(problemDetail({ response: { data: {} } }, 'fallback'), 'fallback')
    assert.equal(problemDetail({ response: { data: { detail: '   ' } } }, 'fallback'), 'fallback')
    assert.equal(problemDetail({ response: { data: { detail: { a: 1 } } } }, 'fallback'), 'fallback')
  })
})
