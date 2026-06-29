const STATE_STORAGE_KEY = 'gb-ui-oauth-state'
const POPUP_MESSAGE_TYPE = 'gb-ui-github-oauth'

/**
 * Opens GitHub OAuth in a popup window. Returns a Promise that:
 * - Never resolves on success (the main window navigates to /auth/callback)
 * - Rejects with 'popup_closed' when the popup closes without completing auth
 *   (happens when w3id SSO interrupts the flow and the user ends up on
 *   github.ibm.com — the popup closing establishes the w3id session, so a
 *   second call will succeed without SSO interference)
 *
 * Falls back to a full-page redirect if popups are blocked.
 */
export async function startGitHubOAuth(): Promise<void> {
  const state = crypto.randomUUID()
  sessionStorage.setItem(STATE_STORAGE_KEY, state)

  const resp = await fetch(`/api/analytics/auth/github/authorize?state=${state}`)
  const { url } = await resp.json() as { url: string }

  const width = 600
  const height = 700
  const left = Math.round(window.screenX + (window.outerWidth - width) / 2)
  const top = Math.round(window.screenY + (window.outerHeight - height) / 2)
  const popup = window.open(
    url,
    'gb-ui-github-oauth',
    `width=${width},height=${height},left=${left},top=${top},scrollbars=yes,resizable=yes`,
  )

  if (!popup) {
    // Popup blocked — fall back to full-page redirect
    window.location.href = url
    return new Promise(() => {}) // page navigates away, promise never settles
  }

  return new Promise((_, reject) => {
    let crossOriginSince: number | null = null

    const handleMessage = (event: MessageEvent) => {
      if (event.origin !== window.location.origin) return
      if (event.data?.type !== POPUP_MESSAGE_TYPE) return
      cleanup()
      if (event.data.error) {
        reject(new Error(event.data.error))
        return
      }
      if (event.data.code && event.data.state) {
        window.location.href = `/auth/callback?code=${event.data.code}&state=${event.data.state}`
      } else {
        reject(new Error('missing_code'))
      }
    }

    // Poll to track how long the popup has been on a cross-origin page.
    // This tells us when the user is on a "stable" external page (w3id or
    // github.ibm.com) vs still mid-navigation.
    const checkClosed = setInterval(() => {
      if (popup.closed) {
        cleanup()
        reject(new Error('popup_closed'))
        return
      }
      try {
        void popup.location.href // throws SecurityError if cross-origin
        crossOriginSince = null  // back on same origin
      } catch {
        if (!crossOriginSince) crossOriginSince = Date.now()
      }
    }, 500)

    // When the main window regains focus while the popup has been cross-origin
    // for 5+ seconds, the user completed w3id SSO and the popup is sitting on
    // github.ibm.com. Auto-close it — the w3id session is now active so a
    // retry will succeed without SSO interference.
    const handleFocus = () => {
      if (!crossOriginSince || popup.closed) return
      if (Date.now() - crossOriginSince < 5000) return
      cleanup()
      popup.close()
      reject(new Error('session_established'))
    }

    const cleanup = () => {
      clearInterval(checkClosed)
      window.removeEventListener('message', handleMessage)
      window.removeEventListener('focus', handleFocus)
    }

    window.addEventListener('message', handleMessage)
    window.addEventListener('focus', handleFocus)
  })
}

export const GITHUB_POPUP_MESSAGE_TYPE = POPUP_MESSAGE_TYPE
