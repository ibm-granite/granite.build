/**
 * Verifies the source tree no longer contains dead code that was supposed to be removed.
 * These are static checks on the filesystem and file contents — no build required.
 *
 * Usage: node --test tests/source-cleanup.test.js
 *
 * Checks against app-level code (this workspace) use APP_ROOT; checks against the
 * shared library (components/api/types/etc.) use UI_CORE_ROOT — since the
 * ui-version-consolidation split, those live in the sibling packages/ui-core workspace,
 * not here. Pointing a check at the wrong root makes it pass vacuously (a missing file
 * "does not contain" anything), which silently stops verifying the invariant it names.
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('fs')
const path = require('path')

const APP_ROOT = path.join(__dirname, '..')
const UI_CORE_ROOT = path.join(__dirname, '..', '..', '..', 'packages', 'ui-core')

function exists(root, rel) {
  return fs.existsSync(path.join(root, rel))
}

function fileContains(root, rel, str) {
  try {
    return fs.readFileSync(path.join(root, rel), 'utf8').includes(str)
  } catch {
    return false
  }
}

describe('deleted pages (Phase 1)', () => {
  it('app/dashboard/workloads/ was removed', () => {
    assert.ok(!exists(APP_ROOT, 'app/dashboard/workloads'), 'app/dashboard/workloads/ should be deleted')
  })

  it('app/dashboard/plans/ was removed', () => {
    assert.ok(!exists(APP_ROOT, 'app/dashboard/plans'), 'app/dashboard/plans/ should be deleted')
  })

  it('K8sResourcesPanel.tsx was removed', () => {
    assert.ok(
      !exists(APP_ROOT, 'app/dashboard/builds/[buildId]/K8sResourcesPanel.tsx'),
      'K8sResourcesPanel.tsx should be deleted',
    )
  })
})

describe('nav cleanup (Phase 2)', () => {
  it('AppHeader does not reference Workloads', () => {
    assert.ok(
      !fileContains(UI_CORE_ROOT, 'components/AppHeader.tsx', 'Workloads'),
      'AppHeader.tsx still contains a Workloads nav link',
    )
  })

  it('AppHeader does not import KubernetesPod', () => {
    assert.ok(
      !fileContains(UI_CORE_ROOT, 'components/AppHeader.tsx', 'KubernetesPod'),
      'AppHeader.tsx still imports KubernetesPod icon',
    )
  })
})

describe('orphaned module deletion (Phase 3)', () => {
  it('api/plans.ts was removed', () => {
    assert.ok(!exists(UI_CORE_ROOT, 'api/plans.ts'), 'api/plans.ts should be deleted')
  })

  it('components/PlanStatusBadge.tsx was removed', () => {
    assert.ok(
      !exists(UI_CORE_ROOT, 'components/PlanStatusBadge.tsx'),
      'PlanStatusBadge.tsx should be deleted',
    )
  })

  it('config/environments.ts was removed', () => {
    assert.ok(!exists(UI_CORE_ROOT, 'config/environments.ts'), 'config/environments.ts should be deleted')
  })
})

describe('isSidecarConfigured removal (Phase 4-5)', () => {
  it('api/analytics.ts does not export isSidecarConfigured', () => {
    assert.ok(
      !fileContains(UI_CORE_ROOT, 'api/analytics.ts', 'isSidecarConfigured'),
      'api/analytics.ts should not export isSidecarConfigured',
    )
  })

  it('builds/page.tsx does not import getBuildResources', () => {
    assert.ok(
      !fileContains(APP_ROOT, 'app/dashboard/builds/page.tsx', 'getBuildResources'),
      'builds/page.tsx should not import getBuildResources',
    )
  })
})

describe('dead K8s infrastructure functions removal (Phase 5)', () => {
  it('api/analytics.ts does not contain getQueueCapacity', () => {
    assert.ok(!fileContains(UI_CORE_ROOT, 'api/analytics.ts', 'getQueueCapacity'))
  })

  it('api/analytics.ts does not contain getNodePools', () => {
    assert.ok(!fileContains(UI_CORE_ROOT, 'api/analytics.ts', 'getNodePools'))
  })

  it('api/analytics.ts does not contain getLeaderboard', () => {
    assert.ok(!fileContains(UI_CORE_ROOT, 'api/analytics.ts', 'getLeaderboard'))
  })

  it('api/analytics.ts does not contain getBuildK8sResources', () => {
    assert.ok(!fileContains(UI_CORE_ROOT, 'api/analytics.ts', 'getBuildK8sResources'))
  })
})

describe('dead type removal (Phase 6)', () => {
  it('types/index.ts does not define QueueCapacity', () => {
    assert.ok(!fileContains(UI_CORE_ROOT, 'types/index.ts', 'QueueCapacity'))
  })

  it('types/index.ts does not define K8sResource', () => {
    assert.ok(!fileContains(UI_CORE_ROOT, 'types/index.ts', 'K8sResource'))
  })

  it('types/index.ts does not define NodePool', () => {
    assert.ok(!fileContains(UI_CORE_ROOT, 'types/index.ts', 'NodePool'))
  })

  it('types/index.ts does not define LeaderboardEntry', () => {
    assert.ok(!fileContains(UI_CORE_ROOT, 'types/index.ts', 'LeaderboardEntry'))
  })

  it('types/index.ts does not define Plan interface', () => {
    assert.ok(!fileContains(UI_CORE_ROOT, 'types/index.ts', 'interface Plan '))
  })

  it('types/index.ts does not define LinkedBuild', () => {
    assert.ok(!fileContains(UI_CORE_ROOT, 'types/index.ts', 'LinkedBuild'))
  })
})

describe('minor cleanups (Phase 7)', () => {
  it('ClientShell does not contain PUBLIC_PATHS', () => {
    assert.ok(
      !fileContains(UI_CORE_ROOT, 'components/ClientShell.tsx', 'PUBLIC_PATHS'),
      'ClientShell.tsx should not contain PUBLIC_PATHS dead code',
    )
  })

  it('gbserver.ts does not reference /auth/callback', () => {
    assert.ok(
      !fileContains(UI_CORE_ROOT, 'api/gbserver.ts', '/auth/callback'),
      'api/gbserver.ts should not reference /auth/callback',
    )
  })
})
