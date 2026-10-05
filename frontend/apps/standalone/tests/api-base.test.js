/**
 * In standalone (production) builds with no *_API_URL baked in, API calls must
 * resolve to relative same-origin paths so gbserver serves/proxies them.
 *
 * Usage: node --test tests/api-base.test.js
 */

const { describe, it, afterEach } = require('node:test')
const assert = require('node:assert/strict')

const { apiBase, autotunexApiBase } = require('../../../packages/ui-core/api/client.ts')

const saved = {}
function save(k) { saved[k] = process.env[k] }
function restore(k) {
  if (saved[k] === undefined) delete process.env[k]
  else process.env[k] = saved[k]
}

describe('same-origin API base in standalone', () => {
  afterEach(() => {
    // Only restore keys this test actually saved via save(k) -- restoring an
    // un-saved key would `delete` it, clobbering a dev/CI-exported value that
    // this test never touched.
    for (const k of Object.keys(saved)) {
      restore(k)
      delete saved[k]
    }
  })

  it('autotunexApiBase is relative in production when unset', () => {
    save('NODE_ENV'); save('AUTOTUNEX_API_URL'); save('GBSERVER_API_URL')
    process.env.NODE_ENV = 'production'
    delete process.env.AUTOTUNEX_API_URL
    delete process.env.GBSERVER_API_URL
    assert.equal(autotunexApiBase('/job/x'), '/api/autotunex/job/x')
  })

  it('autotunexApiBase stays relative even when AUTOTUNEX_API_URL is set', () => {
    // A baked absolute URL sent the browser cross-origin to the AutoTuneX API.
    // That needs credentialed CORS (cors_allow_origins) plus
    // session_cookie_same_site="none" configured on AutoTuneX, which nothing in
    // this repo sets -- and next.config.ts's `env:` block inlines the value into
    // the client bundle, so `AUTOTUNEX_API_URL=... make build-frontend` shipped a
    // silently broken app. The gbserver proxy exists precisely to avoid this, so
    // the browser always goes same-origin through it.
    save('NODE_ENV'); save('AUTOTUNEX_API_URL'); save('GBSERVER_API_URL')
    process.env.NODE_ENV = 'production'
    process.env.AUTOTUNEX_API_URL = 'http://example:8000'
    delete process.env.GBSERVER_API_URL
    assert.equal(autotunexApiBase('/job/x'), '/api/autotunex/job/x')
  })

  it('autotunexApiBase follows GBSERVER_API_URL in a split-origin build', () => {
    // The proxy is mounted on gbserver, so when the static export is served from
    // another host the AutoTuneX calls must target gbserver's origin, exactly as
    // apiBase's do -- relative paths would 404 on the static host.
    save('NODE_ENV'); save('AUTOTUNEX_API_URL'); save('GBSERVER_API_URL')
    process.env.NODE_ENV = 'production'
    delete process.env.AUTOTUNEX_API_URL
    process.env.GBSERVER_API_URL = 'https://gb.example.com'
    assert.equal(autotunexApiBase('/job/x'), 'https://gb.example.com/api/autotunex/job/x')
  })

  it('apiBase is relative in production when GBSERVER_API_URL unset', () => {
    save('NODE_ENV'); save('GBSERVER_API_URL')
    process.env.NODE_ENV = 'production'
    delete process.env.GBSERVER_API_URL
    assert.equal(apiBase('/v1/builds'), '/v1/builds')
  })
})

describe('the AutoTuneX client is wired to the host seam', () => {
  const fs = require('node:fs')
  const path = require('node:path')
  const src = fs.readFileSync(path.join(__dirname, '..', '..', '..', 'packages', 'ui-core', 'api', 'autotunex.ts'), 'utf8')

  it('is built with createApiClient, without the gbserver-only base-URL opt-in', () => {
    // gbserver authenticates /api/autotunex/*, so a client the host's headers
    // cannot reach 401s every call. allowHostBaseUrl would point it at /api/v1.
    assert.match(src, /^const client = createApiClient\(autotunexApiBase\(''\)\)$/m)
    assert.doesNotMatch(src, /axios\.create\(/)
  })
})
