'use client'

import { useState, useEffect } from 'react'
import {
  TextInput,
  Button,
  InlineNotification,
  InlineLoading,
  Tile,
} from '@carbon/react'
import { LogoGithub } from '@carbon/icons-react'
import { useRouter, useSearchParams } from 'next/navigation'
import { useAuth } from '@/auth/useAuth'
import { startGitHubOAuth } from '@/auth/github'
import { startIBMidOAuth, pollIBMidStatus } from '@/auth/ibmid'
import axios from 'axios'

export default function LoginPage() {
  const { login, authProvider: PROVIDER, githubClientId: GITHUB_CLIENT_ID, environment } = useAuth()
  const router = useRouter()
  const searchParams = useSearchParams()

  // Compute once at mount whether the IBMid callback flow should auto-start
  const ibmidState = sessionStorage.getItem('gb-ui-ibmid-state')
  const ibmidVerifier = sessionStorage.getItem('gb-ui-ibmid-verifier')
  const shouldPollIbmid = PROVIDER === 'ibmid' && !!ibmidState && !!ibmidVerifier && !searchParams.get('error')

  const [token, setToken] = useState('')
  const [error, setError] = useState(searchParams.get('error') ?? '')
  const [loading, setLoading] = useState(false)
  const [ibmidPolling, setIbmidPolling] = useState(shouldPollIbmid)
  const [githubPopupOpen, setGithubPopupOpen] = useState(false)
  const [githubInterrupted, setGithubInterrupted] = useState(false)

  // If returning from IBMid flow, start polling automatically
  useEffect(() => {
    if (!shouldPollIbmid) return
    pollIBMidStatus(
      ({ access_token, username, name }) => {
        setIbmidPolling(false)
        login(access_token, username, name)
        const returnTo = sessionStorage.getItem('gb-ui-pre-auth-url') ?? '/dashboard'
        sessionStorage.removeItem('gb-ui-pre-auth-url')
        router.replace(returnTo)
      },
      (msg) => {
        setIbmidPolling(false)
        setError(msg)
      },
    )
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  async function handleTokenSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      await axios.get('/api/v1/spaces/', {
        headers: { Authorization: `Bearer ${token}` },
      })
      login(token, 'user')
      router.push('/builds')
    } catch {
      setError('Invalid token or unable to reach gbserver. Check your token and VITE_GBSERVER_URL.')
    } finally {
      setLoading(false)
    }
  }

  async function handleGitHubSignIn(isRetry = false) {
    setError('')
    setGithubInterrupted(false)
    setGithubPopupOpen(true)
    try {
      await startGitHubOAuth()
      // Never reached on success (main window navigates away)
    } catch (err) {
      const reason = err instanceof Error ? err.message : ''
      if (reason === 'session_established' && !isRetry) {
        // w3id session established in popup — retry immediately, should succeed
        setGithubPopupOpen(false)
        handleGitHubSignIn(true)
        return
      }
      setGithubInterrupted(true)
    } finally {
      setGithubPopupOpen(false)
    }
  }

  async function handleIBMidLogin() {
    setError('')
    await startIBMidOAuth()
    // Page will redirect — no further action needed here
  }

  const showGitHub = PROVIDER === 'github' && GITHUB_CLIENT_ID
  const showIBMid = PROVIDER === 'ibmid'
  const showToken = PROVIDER === 'apikey' || (!showGitHub && !showIBMid)

  // Polling for IBMid completion — show spinner
  if (ibmidPolling) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '100vh', background: 'var(--cds-background)' }}>
        <Tile style={{ width: '100%', maxWidth: '420px', padding: '2rem', textAlign: 'center' }}>
          <InlineLoading description="Waiting for IBMid authentication…" status="active" />
          <p style={{ fontSize: '0.875rem', color: 'var(--cds-text-secondary)', marginTop: '1rem' }}>
            Complete sign-in in the browser tab that opened.
          </p>
        </Tile>
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '100vh', background: 'var(--cds-background)' }}>
      <Tile style={{ width: '100%', maxWidth: '420px', padding: '2rem' }}>
        <h1 style={{ fontSize: '1.5rem', fontWeight: 600, marginBottom: '0.25rem' }}>Granite.build</h1>
        <p style={{ color: 'var(--cds-text-secondary)', marginBottom: '1.5rem', fontSize: '0.875rem' }}>
          Dashboard
        </p>

        {error && (
          <InlineNotification
            kind="error"
            title="Sign in failed"
            subtitle={decodeURIComponent(error)}
            style={{ marginBottom: '1rem' }}
          />
        )}

        {showGitHub && (
          <>
            {githubInterrupted && (
              <InlineNotification
                kind="warning"
                title="Sign-in was interrupted"
                subtitle="Your IBM session has been established. Click Sign in with GitHub to complete sign-in."
                style={{ marginBottom: '1rem' }}
              />
            )}
            <Button
              kind="tertiary"
              renderIcon={githubPopupOpen ? undefined : LogoGithub}
              style={{ width: '100%', justifyContent: 'center', marginBottom: '1rem' }}
              disabled={githubPopupOpen}
              onClick={() => void handleGitHubSignIn()}
            >
              {githubPopupOpen
                ? <InlineLoading description="Waiting for GitHub sign-in…" status="active" />
                : githubInterrupted ? 'Try again' : 'Sign in with GitHub'}
            </Button>
            {showToken && <Divider />}
          </>
        )}

        {showIBMid && (
          <>
            <Button
              kind="tertiary"
              style={{ width: '100%', justifyContent: 'center', marginBottom: '1rem' }}
              onClick={handleIBMidLogin}
            >
              Sign in with IBMid
            </Button>
            {showToken && <Divider />}
          </>
        )}

        {showToken && (
          <form onSubmit={handleTokenSubmit}>
            <TextInput
              id="token"
              type="password"
              labelText="API token"
              placeholder="Paste your gbserver token"
              value={token}
              onChange={(e) => setToken(e.target.value)}
              style={{ marginBottom: '1.5rem' }}
            />
            <Button type="submit" disabled={!token || loading} style={{ width: '100%' }}>
              {loading ? 'Signing in…' : 'Sign in'}
            </Button>
          </form>
        )}

        <p style={{ marginTop: '1.5rem', fontSize: '0.875rem', color: 'var(--cds-text-secondary)' }}>
          Connecting to <code>{environment}</code>
        </p>
      </Tile>
    </div>
  )
}

function Divider() {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', margin: '1rem 0' }}>
      <hr style={{ flex: 1, border: 'none', borderTop: '1px solid var(--cds-border-subtle-00)' }} />
      <span style={{ fontSize: '0.875rem', color: 'var(--cds-text-secondary)', whiteSpace: 'nowrap' }}>or use a token</span>
      <hr style={{ flex: 1, border: 'none', borderTop: '1px solid var(--cds-border-subtle-00)' }} />
    </div>
  )
}
