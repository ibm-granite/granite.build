/**
 * The build detail page must decide whether to show its AutoTuneX panels from the
 * linked tuning job, never from build tags.
 *
 * The tag came from AutoTuneX's `gb_tags` setting (autotunex/src/autotunex/core/
 * config.py) — renameable, extendable to a comma-separated list, and empty
 * disables tagging entirely — so matching it against hardcoded literals silently
 * dropped every AutoTuneX panel whenever an operator changed that setting.
 * `GET /jobs/by-build-id/{build_id}` is the authoritative answer, and is now asked
 * for every build.
 *
 * These are static checks on file contents. This workspace's `node --test` harness
 * has no DOM, so a "does not contain" assertion on a MISSING file would pass
 * vacuously and silently stop verifying anything — every such check below is
 * therefore preceded by an assertion that the file was actually read.
 *
 * Usage: node --test tests/autotunex-build-gate.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('fs')
const path = require('path')

const BUILD_PAGE = path.join(__dirname, '..', 'app', 'dashboard', 'builds', '[buildId]')
const UI_CORE = path.join(__dirname, '..', '..', '..', 'packages', 'ui-core')

function read(dir, rel) {
  try {
    return fs.readFileSync(path.join(dir, rel), 'utf8')
  } catch {
    return ''
  }
}

describe('the linked job, not build tags, gates the AutoTuneX panels', () => {
  it('useLinkedTuningJob owns the by-build-id lookup', () => {
    const hook = read(BUILD_PAGE, 'useLinkedTuningJob.ts')
    assert.ok(hook, 'useLinkedTuningJob.ts should exist')
    assert.match(hook, /getJobByBuildId/, 'the hook should own the by-build-id lookup')
  })

  it('the lookup does not retry', () => {
    // A deployment without AutoTuneX answers 502 on every build page. Retrying a
    // 502 that cannot succeed costs three extra requests per page and shows the
    // user nothing either way, since every failure renders silently.
    const hook = read(BUILD_PAGE, 'useLinkedTuningJob.ts')
    assert.ok(hook, 'useLinkedTuningJob.ts should exist')
    assert.match(hook, /retry:\s*false/, 'the lookup should not retry')
  })

  it('BuildDetails does not read build tags', () => {
    const details = read(BUILD_PAGE, 'BuildDetails.tsx')
    assert.ok(details, 'BuildDetails.tsx should exist')
    assert.ok(
      !details.includes('build?.tags'),
      'BuildDetails should gate the AutoTuneX panels on the linked job, not build tags',
    )
    assert.ok(
      !details.includes('model-customisation'),
      'BuildDetails should not match hardcoded tag literals',
    )
  })
})
