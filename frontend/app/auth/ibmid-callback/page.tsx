'use client'

/**
 * IBMid OIDC callback — the sidecar redirects here after the browser
 * completes the IBMid flow and the sidecar has exchanged the code for tokens.
 *
 * Unlike the GitHub callback (which receives the code directly), the IBMid
 * callback is handled entirely server-side by the sidecar. The browser is
 * sent a success/error HTML page by the sidecar. gb-ui never receives the
 * authorization code.
 *
 * This page is only shown if the user navigates directly to /auth/ibmid/done,
 * which shouldn't happen in normal flow — the sidecar serves its own HTML
 * response to the callback URL. This component exists as a fallback landing
 * page in case the user's browser ends up here.
 */

export default function IBMidCallbackPage() {
  return (
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '100vh', background: '#f4f4f4' }}>
      <div style={{ maxWidth: 400, padding: '2rem', textAlign: 'center' }}>
        <p>Completing IBMid sign-in… you can close this tab.</p>
      </div>
    </div>
  )
}