/**
 * Every AutoTuneX screen must take "is this caller an admin?" from AutoTuneX
 * (`useAutotunexIsAdmin`, backed by GET /api/v1/auth/me), never from gbserver's
 * space roles.
 *
 * Only AutoTuneX decides whether `scope=all` is allowed. Borrowing "admin of any
 * gbserver space" instead sent `scope=own` for a caller AutoTuneX treats as an
 * admin (e.g. standalone AutoTuneX behind a gbserver where they admin no space),
 * and would send `scope=all`, and 403, for the reverse case.
 *
 * Static checks on file contents (no DOM in this harness). Each file is asserted
 * to have been read before its "does not contain" check, so a moved file fails
 * loudly instead of passing vacuously.
 *
 * Usage: node --test tests/autotunex-admin-source.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('fs')
const path = require('path')

const FRONTEND = path.join(__dirname, '..', '..', '..')

// The screens that pick a scope from the admin check.
const SCREENS = [
  'apps/standalone/app/dashboard/autotunex/page.tsx',
  'apps/standalone/app/dashboard/autotunex/[tuningId]/TuningDetailPageClient.tsx',
  'apps/standalone/app/dashboard/autotunex/[tuningId]/TuningDetailTabs.tsx',
  'apps/standalone/app/dashboard/builds/[buildId]/useLinkedTuningJob.ts',
  'packages/ui-core/components/autotunex/settings/ConfigurationsTable.tsx',
  'packages/ui-core/components/autotunex/settings/DatasetsTable.tsx',
  'packages/ui-core/components/autotunex/trials/TrialsTable.tsx',
  'packages/ui-core/components/autotunex/tunings/TuningResultsPanel.tsx',
]

function read(rel) {
  try {
    return fs.readFileSync(path.join(FRONTEND, rel), 'utf8')
  } catch {
    return ''
  }
}

describe('AutoTuneX admin check comes from AutoTuneX', () => {
  for (const rel of SCREENS) {
    it(`${path.basename(rel)} uses useAutotunexIsAdmin, not gbserver spaces`, () => {
      const src = read(rel)
      assert.ok(src, `${rel} should exist`)
      assert.match(src, /useAutotunexIsAdmin\(\)/, 'should read the admin flag from AutoTuneX')
      assert.ok(!src.includes('listSpaces'), 'should not derive the admin flag from gbserver spaces')
    })
  }
})
