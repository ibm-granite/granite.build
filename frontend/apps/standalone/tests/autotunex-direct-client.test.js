/**
 * configureAutotunexClient: a hosted dashboard calling AutoTuneX directly.
 *
 * Usage: node --test tests/autotunex-direct-client.test.js
 *
 * Unlike most client tests here, these exercise real behaviour: the request is
 * built by axios's own interceptors and captured by a stub adapter just before it
 * would leave, so what is asserted is the request the browser would send.
 *
 * The case that matters most is the bearer token. AutoTuneX routes a bearer
 * credential ahead of its session cookie (autotunex core/auth/routing.py), and a
 * `session`-only deployment has no verifier for it, so a direct request that
 * still carried the host's gbserver token would 401 despite a valid cookie.
 */

const { describe, it, beforeEach, afterEach } = require('node:test')
const assert = require('node:assert/strict')

const {
  autotunexResourceUrl,
  configureApiClient,
  configureAutotunexClient,
  createAutotunexClient,
} = require('../../../packages/ui-core/api/client.ts')

/** A client whose requests are captured instead of sent, answering `status`. */
function capturingClient(status = 200) {
  const client = createAutotunexClient()
  const sent = []
  client.defaults.adapter = async (config) => {
    sent.push(config)
    const response = { data: {}, status, statusText: String(status), headers: {}, config }
    if (status >= 400) {
      const err = new Error(`status ${status}`)
      err.config = config
      err.response = response
      throw err
    }
    return response
  }
  return { client, sent }
}

describe('the AutoTuneX client', () => {
  let hostUnauthorized
  let autotunexUnauthorized

  beforeEach(() => {
    hostUnauthorized = 0
    autotunexUnauthorized = 0
    configureApiClient({
      resolveHeaders: () => ({ Authorization: 'Bearer gbserver-token' }),
      onUnauthorized: () => { hostUnauthorized++ },
    })
  })

  afterEach(() => {
    configureAutotunexClient(null)
    configureApiClient({})
  })

  describe('unconfigured: through the gbserver proxy', () => {
    it('targets the proxy path with the host headers and no credentials', async () => {
      const { client, sent } = capturingClient()
      await client.get('/auth/me')
      assert.equal(sent[0].baseURL, '/api/autotunex')
      assert.equal(sent[0].headers.Authorization, 'Bearer gbserver-token')
      assert.notEqual(sent[0].withCredentials, true)
    })

    it('hands a 401 to the host', async () => {
      const { client } = capturingClient(401)
      await assert.rejects(client.get('/auth/me'))
      assert.equal(hostUnauthorized, 1)
    })

    it('builds download links on the proxy path', () => {
      assert.equal(autotunexResourceUrl('/jobs/j/result-report/archive'), '/api/autotunex/jobs/j/result-report/archive')
    })
  })

  describe('configured: direct to AutoTuneX', () => {
    beforeEach(() => {
      configureAutotunexClient({
        baseUrl: 'https://autotunex.example.com/',
        onUnauthorized: () => { autotunexUnauthorized++ },
      })
    })

    it('targets AutoTuneX /api/v1, credentialed', async () => {
      const { client, sent } = capturingClient()
      await client.get('/auth/me')
      assert.equal(sent[0].baseURL, 'https://autotunex.example.com/api/v1')
      assert.equal(sent[0].withCredentials, true)
    })

    it('never forwards the host gbserver headers', async () => {
      const { client, sent } = capturingClient()
      await client.get('/auth/me')
      assert.equal(sent[0].headers.Authorization, undefined)
    })

    it('hands a 401 to the AutoTuneX hook, not the host login', async () => {
      const { client } = capturingClient(401)
      await assert.rejects(client.get('/auth/me'))
      assert.equal(autotunexUnauthorized, 1)
      assert.equal(hostUnauthorized, 0)
    })

    it('ignores a non-401 error', async () => {
      const { client } = capturingClient(403)
      await assert.rejects(client.get('/jobs/'))
      assert.equal(autotunexUnauthorized, 0)
    })

    it('leaves a request that names its own base alone', async () => {
      const { client, sent } = capturingClient()
      await client.get('https://elsewhere.example.com/x', { baseURL: '' })
      assert.equal(sent[0].baseURL, '')
      assert.notEqual(sent[0].withCredentials, true)
      assert.equal(sent[0].headers.Authorization, undefined)
    })

    it('builds download links on AutoTuneX directly', () => {
      assert.equal(
        autotunexResourceUrl('/jobs/j/result-report/archive?scope=own'),
        'https://autotunex.example.com/api/v1/jobs/j/result-report/archive?scope=own',
      )
    })

    it('applies a configuration made after the client was built', async () => {
      configureAutotunexClient(null)
      const { client, sent } = capturingClient()
      configureAutotunexClient({ baseUrl: 'https://late.example.com' })
      await client.get('/auth/me')
      assert.equal(sent[0].baseURL, 'https://late.example.com/api/v1')
    })
  })
})
