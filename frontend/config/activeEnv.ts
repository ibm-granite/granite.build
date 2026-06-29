/**
 * Session-level environment store.
 *
 * The active environment override is kept here instead of in the URL so that
 * it propagates automatically to all API clients and nav links without any
 * per-link wiring. sessionStorage persists across page navigations within the
 * same browser tab but is cleared when the tab is closed.
 */

const SESSION_KEY = 'gb_active_env'

// In-memory mirror so reads are synchronous even before sessionStorage is available
let _env: string | null = null

export function getActiveEnv(): string | null {
  if (_env !== null) return _env
  if (typeof window !== 'undefined') {
    const stored = sessionStorage.getItem(SESSION_KEY)
    if (stored) _env = stored
  }
  return _env
}

export function setActiveEnv(envId: string): void {
  _env = envId
  if (typeof window !== 'undefined') {
    sessionStorage.setItem(SESSION_KEY, envId)
  }
}

export function clearActiveEnv(): void {
  _env = null
  if (typeof window !== 'undefined') {
    sessionStorage.removeItem(SESSION_KEY)
  }
}
