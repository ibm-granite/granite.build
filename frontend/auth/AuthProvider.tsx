'use client'

import React, { useState, useCallback, useMemo } from 'react'
import { AuthContext } from './AuthContext'
import type { AuthProvider, AuthState, AuthContextValue } from './AuthContext'

export interface RuntimeConfig {
  environment: string
  authProvider: AuthProvider
  githubClientId: string
}

const STANDALONE_AUTH: AuthState = { token: '', username: 'standalone', provider: 'apikey' }
const STORAGE_KEY = 'gb-ui-auth'

// Updated by AuthProvider on each render — safe to read in non-React
// request interceptors (api/gbserver.ts etc.) which run after mount.
export let IS_STANDALONE = false

function loadStoredAuth(): AuthState | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return null
    const state = JSON.parse(raw) as AuthState
    if (!state.name) {
      state.name = state.username
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state))
    }
    return state
  } catch {
    return null
  }
}

export function AuthProvider({
  children,
  runtimeConfig,
}: {
  children: React.ReactNode
  runtimeConfig: RuntimeConfig
}) {
  const isStandalone = runtimeConfig.authProvider === 'apikey'
  IS_STANDALONE = isStandalone

  const [auth, setAuth] = useState<AuthState | null>(() =>
    isStandalone ? STANDALONE_AUTH : loadStoredAuth(),
  )

  const login = useCallback(
    (token: string, username: string, name?: string) => {
      if (isStandalone) return
      const state: AuthState = {
        token,
        username,
        name: name ?? username,
        provider: runtimeConfig.authProvider,
      }
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state))
      setAuth(state)
    },
    [isStandalone, runtimeConfig.authProvider],
  )

  const logout = useCallback(() => {
    if (isStandalone) return
    localStorage.removeItem(STORAGE_KEY)
    setAuth(null)
  }, [isStandalone])

  const value = useMemo<AuthContextValue>(
    () => ({
      auth,
      login,
      logout,
      isAuthenticated: isStandalone || Boolean(auth?.token),
      provider: auth?.provider ?? runtimeConfig.authProvider,
      isStandalone,
      authProvider: runtimeConfig.authProvider,
      githubClientId: runtimeConfig.githubClientId,
      environment: runtimeConfig.environment,
      configLoaded: true,
    }),
    [auth, login, logout, isStandalone, runtimeConfig],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}
