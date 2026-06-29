'use client'

import { createContext } from 'react'

export type AuthProvider = 'apikey' | 'github' | 'ibmid'

export interface AuthState {
  token: string
  username: string
  name?: string
  provider: AuthProvider
}

export interface AuthContextValue {
  auth: AuthState | null
  login: (token: string, username: string, name?: string) => void
  logout: () => void
  isAuthenticated: boolean
  provider: AuthProvider
  // Runtime config — fetched from /api/config on mount so it can vary per deployment
  // without needing to be baked into the JS bundle at build time.
  isStandalone: boolean
  authProvider: AuthProvider
  githubClientId: string
  environment: string
  configLoaded: boolean
}

export const AuthContext = createContext<AuthContextValue | null>(null)
