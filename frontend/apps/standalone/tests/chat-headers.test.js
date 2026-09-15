/**
 * Verifies every outbound chat request can carry host-supplied headers.
 *
 * Usage: node --test tests/chat-headers.test.js
 *
 * Why static checks rather than behavioural ones: this workspace's test setup is plain
 * `node --test` with no DOM or module transpiler, so exercising an axios interceptor and a
 * native fetch against TypeScript source is not available here. The failure mode worth
 * guarding is structural anyway — a *new* chat call path that forgets the headers — and a
 * source-level check catches that, where a behavioural test of the existing two would not.
 *
 * Chat is the one API client a host app cannot reach with its own axios interceptor: three
 * routes go through this module's private `statusClient`, and /chat/stream uses native fetch
 * because axios does not stream cleanly in-browser. If a request escapes both hooks it
 * arrives unidentified, and the backend's resolve_identity() collapses every user into one
 * shared "standalone" identity — merging the per-identity rate limit into a single bucket and
 * merging the session scoping that stops one user resolving another's pending confirm_action.
 *
 * Like source-cleanup.test.js, these read from UI_CORE_ROOT, not APP_ROOT — chat.ts lives in
 * the sibling packages/ui-core workspace. The first test exists so that a moved or renamed
 * file fails loudly here instead of making every "source contains ..." check below pass
 * vacuously against an empty string.
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('fs')
const path = require('path')

const UI_CORE_ROOT = path.join(__dirname, '..', '..', '..', 'packages', 'ui-core')
const CHAT_TS = path.join(UI_CORE_ROOT, 'api', 'chat.ts')

function chatSource() {
  return fs.readFileSync(CHAT_TS, 'utf8')
}

function countOccurrences(haystack, needle) {
  return haystack.split(needle).length - 1
}

/**
 * The source of the balanced (...) argument list that follows `marker`.
 *
 * Scoping matters more than it looks: an earlier version of the interceptor check matched
 * /use\(async \(config\) => \{[\s\S]*?resolveChatHeaders\(\)/ against the whole file, which
 * passed as long as resolveChatHeaders() appeared *anywhere* later — including in the
 * unrelated fetch path further down. It therefore still passed against an interceptor whose
 * body had been gutted. Matching only within the call's own parentheses is what makes the
 * check mean what it says.
 */
function callArgumentSource(src, marker) {
  const start = src.indexOf(marker)
  if (start === -1) return ''
  let i = src.indexOf('(', start + marker.length - 1)
  if (i === -1) return ''
  let depth = 0
  const from = i
  for (; i < src.length; i += 1) {
    if (src[i] === '(') depth += 1
    else if (src[i] === ')') {
      depth -= 1
      if (depth === 0) return src.slice(from, i + 1)
    }
  }
  return ''
}

describe('chat header provider', () => {
  it('api/chat.ts is where these checks expect it', () => {
    assert.ok(
      fs.existsSync(CHAT_TS),
      `packages/ui-core/api/chat.ts not found at ${CHAT_TS} — if it moved, update this test; ` +
        'otherwise every check below would pass against an empty string',
    )
    assert.ok(chatSource().length > 500, 'api/chat.ts is unexpectedly small — wrong file?')
  })

  it('exposes a settable header provider', () => {
    const src = chatSource()
    assert.match(src, /export function setChatHeaderProvider\(/)
    assert.match(src, /export type ChatHeaderProvider/)
  })

  it('applies the provider to the statusClient path', () => {
    const src = chatSource()
    assert.match(
      src,
      /statusClient\.interceptors\.request\.use\(/,
      'statusClient has no request interceptor, so /chat/status, /chat/stop and ' +
        '/chat/confirm cannot carry host headers',
    )
    const body = callArgumentSource(src, 'statusClient.interceptors.request.use')
    assert.ok(body.length > 0, 'could not parse the interceptor call — has its shape changed?')
    assert.ok(
      body.includes('resolveChatHeaders()'),
      "statusClient's request interceptor does not call resolveChatHeaders(), so the three " +
        'routes behind statusClient would arrive unidentified',
    )
  })

  it('applies the provider to the streaming fetch path', () => {
    const src = chatSource()
    assert.ok(
      /headers: \{\s*\.\.\.\(await resolveChatHeaders\(\)\)/.test(src),
      'the /chat/stream fetch does not spread resolveChatHeaders() into its headers',
    )
  })

  it('keeps Content-Type authoritative on the streaming body', () => {
    // A provider returning its own Content-Type must not change how the JSON body is read,
    // so the literal has to come after the spread rather than before it.
    const src = chatSource()
    const spreadAt = src.indexOf('...(await resolveChatHeaders())')
    const contentTypeAt = src.indexOf("'Content-Type': 'application/json'", spreadAt)
    assert.ok(spreadAt !== -1 && contentTypeAt > spreadAt, 'Content-Type must be set after the spread')
  })

  it('has no chat request path that bypasses both hooks', () => {
    // The guard that actually earns its keep: a third call path added later.
    const src = chatSource()
    assert.equal(
      countOccurrences(src, 'axios.create('),
      1,
      'a second axios instance was added to chat.ts — it needs the same request interceptor, ' +
        'or its calls will arrive unidentified',
    )
    assert.equal(
      countOccurrences(src, 'fetch('),
      1,
      'chat.ts makes more than one direct fetch() call — each needs resolveChatHeaders() ' +
        'spread into its headers',
    )
  })
})
