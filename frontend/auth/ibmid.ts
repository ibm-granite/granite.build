/**
 * IBMid OIDC authentication — browser-native PKCE + polling.
 *
 * Mirrors granite.build's ibmid_auth.py IBMidOIDCClient, adapted for
 * the browser. The sidecar acts as the confidential client; the browser
 * only ever sees the access token, never the client secret.
 *
 * Flow:
 *  1. generatePKCE() — create verifier + S256 challenge in the browser
 *  2. startIBMidOAuth() — redirect browser to sidecar /authorize
 *  3. Sidecar redirects browser to IBM, user authenticates
 *  4. IBM redirects to sidecar /callback — sidecar exchanges code for tokens
 *  5. pollIBMidStatus() — browser polls sidecar /status with verifier
 *  6. Sidecar verifies PKCE and returns tokens (one-time)
 */
import axios from 'axios'

const STATE_KEY   = 'gb-ui-ibmid-state'
const VERIFIER_KEY = 'gb-ui-ibmid-verifier'
const REDIRECT_KEY = 'gb-ui-pre-auth-url'

// ── PKCE ─────────────────────────────────────────────────────────────────────

async function generatePKCE(): Promise<{ verifier: string; challenge: string }> {
  const verifier = crypto.randomUUID().replace(/-/g, '') + crypto.randomUUID().replace(/-/g, '')
  const encoder = new TextEncoder()
  const data = encoder.encode(verifier)
  const digest = await crypto.subtle.digest('SHA-256', data)
  const challenge = btoa(String.fromCharCode(...new Uint8Array(digest)))
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=/g, '')
  return { verifier, challenge }
}

// ── Start flow ────────────────────────────────────────────────────────────────

export async function startIBMidOAuth(): Promise<void> {
  const { verifier, challenge } = await generatePKCE()
  const state = crypto.randomUUID()

  sessionStorage.setItem(STATE_KEY, state)
  sessionStorage.setItem(VERIFIER_KEY, verifier)
  sessionStorage.setItem(REDIRECT_KEY, window.location.pathname)

  const params = new URLSearchParams({
    code_challenge: challenge,
    code_challenge_method: 'S256',
    state,
  })

  // Redirect to the sidecar's /authorize, which handles the IBM redirect.
  window.location.href = `/api/analytics/auth/ibmid/authorize?${params}`
}

// ── Poll for completion ────────────────────────────────────────────────────────

export interface IBMidTokenResult {
  access_token: string
  username: string
  name?: string
  email: string
}

export async function pollIBMidStatus(
  onDone: (result: IBMidTokenResult) => void,
  onError: (msg: string) => void,
  intervalMs = 3000,
  timeoutMs = 120_000,
): Promise<void> {
  const state = sessionStorage.getItem(STATE_KEY)
  const verifier = sessionStorage.getItem(VERIFIER_KEY)

  if (!state || !verifier) {
    onError('IBMid session not found. Please try signing in again.')
    return
  }

  const deadline = Date.now() + timeoutMs
  let running = true

  async function tick() {
    if (!running || Date.now() > deadline) {
      if (running) {
        running = false
        onError('IBMid authentication timed out. Please try again.')
      }
      return
    }

    try {
      const { data } = await axios.get('/api/analytics/auth/ibmid/status', {
        params: { state, code_verifier: verifier },
        validateStatus: (s) => s < 500,
      })

      if (data.status === 'pending') {
        setTimeout(tick, intervalMs)
        return
      }

      running = false
      sessionStorage.removeItem(STATE_KEY)
      sessionStorage.removeItem(VERIFIER_KEY)

      if (data.status === 'error') {
        onError(data.error || 'IBMid authentication failed.')
        return
      }

      if (data.status === 'complete') {
        onDone({
          access_token: data.access_token,
          username: data.user_info?.preferred_username || data.user_info?.sub || '',
          name: data.user_info?.name || data.user_info?.preferred_username || '',
          email: data.user_info?.email || '',
        })
        return
      }
    } catch {
      // Transient network error — keep polling
      setTimeout(tick, intervalMs)
    }
  }

  setTimeout(tick, intervalMs)
}
