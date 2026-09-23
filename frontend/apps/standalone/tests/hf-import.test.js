/**
 * Tests for the HuggingFace dataset-import decision logic.
 *
 * These functions exist as a separate module precisely so they can be tested:
 * there is no jsdom harness in this app, so anything left inside the
 * HuggingFace import UI has no automated coverage at all. Each function here is one
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
  hfErrorStatus,
  pollStep,
  mappedPreviewKey,
  canImport,
  pruneMapping,
  NO_VALIDATION,
  preselectValidationSplit,
  reconcileValidationSplit,
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

  // The discriminating pair for the shard-boundary case: identical row counts, and
  // the verdict turns entirely on the server's flag. Comparing rows alone -- which
  // is all this did before -- reports the first of these as a clean import.
  it('reports truncation the row counts cannot show, naming the unread shards', () => {
    const notice = truncationNotice(
      {
        train_original_rows: 50000,
        train_retained_rows: 50000,
        train_truncated: true,
        train_unread_shards: 9999,
      },
      50000
    )
    assert.match(notice, /train/)
    assert.match(notice, /9,999/)
    assert.match(notice, /not read/)
    // No "of 50,000": `_original_rows` counts only the shards that were opened, so
    // quoting it as the split total would be a number the server never claimed.
    assert.ok(!notice.includes('of 50,000'))
  })

  it('stays silent on the same counts when the server reported no truncation', () => {
    assert.equal(
      truncationNotice(
        { train_original_rows: 50000, train_retained_rows: 50000, train_unread_shards: 0 },
        50000
      ),
      null
    )
  })

  it('reports a validation split truncated at a shard boundary', () => {
    const notice = truncationNotice(
      {
        train_original_rows: 100,
        train_retained_rows: 100,
        validation_original_rows: 50000,
        validation_retained_rows: 50000,
        validation_truncated: true,
        validation_unread_shards: 1,
      },
      50000
    )
    assert.match(notice, /validation/)
    // Singular, because "1 shards were not read" is the kind of copy people notice.
    assert.match(notice, /1 further shard was not read/)
    assert.ok(!notice.includes('train'))
  })

  it('trusts row counts over a flag that contradicts them', () => {
    // A `false` flag must not be able to hide a shortfall the counts state plainly.
    const notice = truncationNotice(
      { train_original_rows: 120000, train_retained_rows: 50000, train_truncated: false },
      50000
    )
    assert.match(notice, /120,000/)
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

describe('hfErrorStatus', () => {
  it('round-trips the statuses callers branch on', () => {
    assert.equal(hfErrorStatus({ response: { status: 503 } }), 503)
    assert.equal(hfErrorStatus({ response: { status: 422 } }), 422)
    assert.equal(hfErrorStatus({ response: { status: 409 } }), 409)
  })

  it('is undefined when there is no usable status', () => {
    assert.equal(hfErrorStatus({}), undefined)
    assert.equal(hfErrorStatus(null), undefined)
    assert.equal(hfErrorStatus(undefined), undefined)
  })

  it('is undefined for a non-number status, not the string itself', () => {
    // A string '503' would make `=== 503` silently false, so the Retry button
    // would never appear for a real Axios error carrying this shape.
    assert.equal(hfErrorStatus({ response: { status: '503' } }), undefined)
  })
})

describe('pollStep', () => {
  it('is ready when the status says ready, expired or not', () => {
    assert.equal(pollStep({ status: 'ready', expired: false }), 'ready')
    assert.equal(pollStep({ status: 'ready', expired: true }), 'ready')
  })

  it('is error when the status says error, expired or not', () => {
    assert.equal(pollStep({ status: 'error', expired: false }), 'error')
    assert.equal(pollStep({ status: 'error', expired: true }), 'error')
  })

  it('times out on an expired deadline with no terminal status', () => {
    assert.equal(pollStep({ status: undefined, expired: true }), 'timeout')
    assert.equal(pollStep({ status: 'importing', expired: true }), 'timeout')
  })

  it('waits while there is time left and no terminal status', () => {
    assert.equal(pollStep({ status: 'importing', expired: false }), 'wait')
    assert.equal(pollStep({ status: undefined, expired: false }), 'wait')
  })
})

describe('mappedPreviewKey', () => {
  const base = { repoId: 'vicgalle/alpaca-gpt4', config: 'default', trainSplit: 'train', mappingKey: '{"input":"instruction"}' }

  it('is identical for identical inputs', () => {
    assert.equal(mappedPreviewKey(base), mappedPreviewKey({ ...base }))
  })

  it('differs when the mapping differs', () => {
    assert.notEqual(mappedPreviewKey(base), mappedPreviewKey({ ...base, mappingKey: '{"input":"text"}' }))
  })

  it('differs when the config differs', () => {
    assert.notEqual(mappedPreviewKey(base), mappedPreviewKey({ ...base, config: 'other' }))
  })

  it('differs when the repoId differs', () => {
    // Dropping repoId from the key is exactly the stale-read bug this key exists
    // to prevent: two repos sharing a config/split/mapping would otherwise collide.
    assert.notEqual(mappedPreviewKey(base), mappedPreviewKey({ ...base, repoId: 'other/repo' }))
  })

  it('differs when the trainSplit differs', () => {
    assert.notEqual(mappedPreviewKey(base), mappedPreviewKey({ ...base, trainSplit: 'test' }))
  })
})

describe('canImport', () => {
  const base = {
    hasRepo: true,
    hasConfig: true,
    hasTrainSplit: true,
    mappingComplete: true,
    nameValid: true,
    survivalKind: 'ok',
    importing: false,
    splitFromTrain: false,
    validationPercentage: 10,
  }

  it('allows an ok or a warning survival', () => {
    assert.equal(canImport({ ...base, survivalKind: 'ok' }), true)
    assert.equal(canImport({ ...base, survivalKind: 'warning' }), true)
  })

  it('blocks a hidden or a blocked survival', () => {
    assert.equal(canImport({ ...base, survivalKind: 'hidden' }), false)
    assert.equal(canImport({ ...base, survivalKind: 'blocked' }), false)
  })

  it('blocks while a run is already importing', () => {
    assert.equal(canImport({ ...base, importing: true }), false)
  })

  it('blocks an invalid name', () => {
    assert.equal(canImport({ ...base, nameValid: false }), false)
  })

  it('blocks a missing repo, config or train split', () => {
    assert.equal(canImport({ ...base, hasRepo: false }), false)
    assert.equal(canImport({ ...base, hasConfig: false }), false)
    assert.equal(canImport({ ...base, hasTrainSplit: false }), false)
  })

  it('blocks an out-of-range validation percentage when splitting from train', () => {
    assert.equal(canImport({ ...base, splitFromTrain: true, validationPercentage: 0 }), false)
    // Pins the upper boundary at 50: an implementation that let it drift to 51
    // (e.g. `<= 51`) would pass every other assertion in this file and only this
    // one would catch it.
    assert.equal(canImport({ ...base, splitFromTrain: true, validationPercentage: 51 }), false)
    assert.equal(canImport({ ...base, splitFromTrain: true, validationPercentage: 99 }), false)
  })

  it('allows the boundary and mid-range percentages when splitting from train', () => {
    assert.equal(canImport({ ...base, splitFromTrain: true, validationPercentage: 1 }), true)
    assert.equal(canImport({ ...base, splitFromTrain: true, validationPercentage: 10 }), true)
    assert.equal(canImport({ ...base, splitFromTrain: true, validationPercentage: 50 }), true)
  })

  it('blocks a non-integer validation percentage within range', () => {
    // Pins integrality: 10.5 satisfies `>= 1 && <= 50` on its own, so without the
    // Number.isInteger guard it would pass canImport and reach the server's
    // `int | None` field as a 422.
    assert.equal(canImport({ ...base, splitFromTrain: true, validationPercentage: 10.5 }), false)
  })

  it('ignores the percentage when a separate validation split is chosen', () => {
    assert.equal(canImport({ ...base, splitFromTrain: false, validationPercentage: 0 }), true)
    assert.equal(canImport({ ...base, splitFromTrain: false, validationPercentage: 99 }), true)
  })
})

describe('pruneMapping', () => {
  it('drops targets that are no longer required', () => {
    assert.deepEqual(
      pruneMapping({ input: 'instruction', output: 'response' }, ['prompt', 'chosen']),
      {}
    )
  })

  it('keeps the targets that survive, with their sources', () => {
    assert.deepEqual(
      pruneMapping({ input: 'instruction', output: 'response' }, ['input']),
      { input: 'instruction' }
    )
  })

  it('returns the SAME object when nothing needs dropping', () => {
    // The prune effect depends on this: a fresh object every render would set
    // state on every commit and loop forever.
    const mapping = { input: 'instruction' }
    assert.equal(pruneMapping(mapping, ['input', 'output']), mapping)
  })

  it('returns the same object for an empty mapping', () => {
    const mapping = {}
    assert.equal(pruneMapping(mapping, ['input']), mapping)
  })

  it('drops everything when there are no required columns', () => {
    assert.deepEqual(pruneMapping({ input: 'instruction' }, []), {})
  })

  it('preserves an empty-string source for a still-required target', () => {
    // An empty source is an incomplete mapping, not a stale key -- dropping it
    // would silently reset a select the user is part-way through.
    assert.deepEqual(pruneMapping({ input: '' }, ['input']), { input: '' })
  })
})

describe('preselectValidationSplit', () => {
  it('picks a split named exactly "validation"', () => {
    assert.equal(preselectValidationSplit(['test', 'validation']), 'validation')
  })

  it('does not fall back to "test"', () => {
    // Selecting a model against the held-out test split is a methodology error,
    // and it is the "wrong split discovered after a multi-hour run" case the
    // import UI's own comments warn about. An implementation that ranked
    // candidates would return 'test' here and still pass every other assertion
    // in this block.
    assert.equal(preselectValidationSplit(['test']), '')
  })

  it('does not match a near-miss name', () => {
    assert.equal(preselectValidationSplit(['valid', 'dev', 'eval']), '')
  })

  it('returns empty for no candidates', () => {
    assert.equal(preselectValidationSplit([]), '')
  })
})

describe('reconcileValidationSplit', () => {
  it('forces split-from-train when the new train split leaves no candidates', () => {
    // The toggle hides at zero candidates, so any other return would leave it
    // stuck OFF with an unreachable control -- holding a stale name equal to the
    // new train split, which is the bug this function exists to fix.
    assert.equal(reconcileValidationSplit('test', []), NO_VALIDATION)
  })

  it('leaves split-from-train alone when candidates remain', () => {
    // NO_VALIDATION is a sentinel and never a member of nextCandidates, so
    // without an explicit check for it the "still a candidate" branch cannot
    // match and the preselect branch fires -- silently switching the toggle OFF
    // on the default path every time the train split changes.
    assert.equal(reconcileValidationSplit(NO_VALIDATION, ['test', 'validation']), NO_VALIDATION)
  })

  it('keeps a selection that is still a candidate', () => {
    assert.equal(reconcileValidationSplit('validation', ['test', 'validation']), 'validation')
  })

  it('re-preselects when the selection is no longer a candidate', () => {
    assert.equal(reconcileValidationSplit('test', ['train', 'validation']), 'validation')
  })

  it('re-preselects to empty rather than snapping back to split-from-train', () => {
    // Keeps the toggle OFF: forcing it back ON would silently discard a choice
    // the user made explicitly. '' blocks Import visibly instead.
    assert.equal(reconcileValidationSplit('validation', ['train', 'test']), '')
  })
})
