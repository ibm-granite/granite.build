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
    save('NODE_ENV'); save('AUTOTUNEX_API_URL')
    process.env.NODE_ENV = 'production'
    delete process.env.AUTOTUNEX_API_URL
    assert.equal(autotunexApiBase('/job/x'), '/api/autotunex/job/x')
  })

  it('autotunexApiBase targets the baked URL when set', () => {
    save('NODE_ENV'); save('AUTOTUNEX_API_URL')
    process.env.NODE_ENV = 'production'
    process.env.AUTOTUNEX_API_URL = 'http://example:8000'
    assert.equal(autotunexApiBase('/job/x'), 'http://example:8000/api/v1/job/x')
  })

  it('apiBase is relative in production when GBSERVER_API_URL unset', () => {
    save('NODE_ENV'); save('GBSERVER_API_URL')
    process.env.NODE_ENV = 'production'
    delete process.env.GBSERVER_API_URL
    assert.equal(apiBase('/v1/builds'), '/v1/builds')
  })
})
