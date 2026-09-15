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
    // 502 that cannot succeed costs one extra request per page and shows the user
    // nothing either way, since every failure renders silently. That's one, not
    // react-query's library default of three, because this app's own QueryClient
    // (ClientShell) already sets `retry: 1`.
    const hook = read(BUILD_PAGE, 'useLinkedTuningJob.ts')
    assert.ok(hook, 'useLinkedTuningJob.ts should exist')
    assert.match(hook, /retry:\s*false/, 'the lookup should not retry')
  })

  it('the lookup waits for spaces before firing', () => {
    // `isAdmin` is part of the query key and starts false while `listSpaces` is in
    // flight. Without waiting for it, the lookup fires twice for an admin — once
    // at scope=own, then again at scope=all under a new key. The bug is invisible
    // in the UI (identical render, one extra request), so nothing but this
    // mechanical check catches someone removing `&& !spacesPending` as apparently
    // redundant.
    const hook = read(BUILD_PAGE, 'useLinkedTuningJob.ts')
    assert.ok(hook, 'useLinkedTuningJob.ts should exist')
    assert.match(hook, /!spacesPending/, 'the lookup must wait for spaces so isAdmin is settled before the key is built')
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
    assert.match(details, /useLinkedTuningJob\(buildId\)/, 'BuildDetails should gate on the linked-job lookup')
  })

  it('AutoTuneXPanel takes the job as a prop instead of fetching it', () => {
    const panel = read(BUILD_PAGE, 'AutoTuneXPanel.tsx')
    assert.ok(panel, 'AutoTuneXPanel.tsx should exist')
    assert.ok(
      !panel.includes('getJobByBuildId'),
      'BuildDetails owns the lookup; AutoTuneXPanel should not fetch the job',
    )
    assert.ok(
      !panel.includes('listSpaces'),
      'the scope travels with the job as a prop; AutoTuneXPanel should not resolve its own',
    )
  })

  it('the Trials and Logs panels take the job as a prop instead of fetching it', () => {
    const panels = read(BUILD_PAGE, 'AutoTuneXJobPanels.tsx')
    assert.ok(panels, 'AutoTuneXJobPanels.tsx should exist')
    assert.ok(
      !panels.includes('getJobByBuildId'),
      'BuildDetails owns the lookup; these panels should not fetch the job',
    )
    assert.ok(
      !panels.includes('listSpaces'),
      'the scope travels with the job as a prop; these panels should not resolve their own',
    )
    assert.ok(
      !panels.includes('NoJob'),
      'a panel only mounts when a job exists, so the no-job notice is unreachable',
    )
  })

  it('getJobByBuildId no longer documents a tag-gated caller', () => {
    const api = read(UI_CORE, 'api/autotunex.ts')
    assert.ok(api, 'packages/ui-core/api/autotunex.ts should exist')
    assert.ok(
      !api.includes('merely carry'),
      "the docstring should not describe callers rendering nothing for builds that 'merely carry' the tag",
    )
  })
})
