'use client'

import React from 'react'
import { type AppRoutes, DEFAULT_ROUTES } from './routeShapes'

// Re-exported so `config/routes` remains the single import path for consumers,
// while the pure builders stay importable from a module without 'use client'.
// See routeShapes.ts for why the split exists.
export { DEFAULT_ROUTES } from './routeShapes'
export type { AppRoutes } from './routeShapes'

const RoutesContext = React.createContext<AppRoutes>(DEFAULT_ROUTES)

/**
 * Override the link shape for everything below. Mount once, near the root.
 *
 * `value` does not need memoising by the caller, including in the form that
 * matters: `<ClientShell routes={{ buildHref: (id) => …, artifactHref: (id) => … }}>`
 * creates two new functions on every render. The context value published here is
 * stable regardless — it is built once and delegates through a ref, so a new
 * `value` changes which builders get called without changing the identity every
 * `useRoutes()` consumer depends on.
 *
 * An earlier version memoised on `[value.buildHref, value.artifactHref]`, which
 * reads like it covers this and does not: those are the identities that change.
 * The version before that documented stability as the caller's obligation. Both
 * failed the same way — every consumer re-rendering on every parent render, a
 * perf cliff rather than an error, so nothing surfaced it.
 *
 * Delegating is safe here specifically because `AppRoutes` is a stateless bag of
 * pure functions: there is no stale closure to capture, and the newest builder is
 * always the one invoked. The ref is written during render rather than in an
 * effect so that the first render already calls the builders it was given.
 */
export function RoutesProvider({
  value,
  children,
}: {
  value: AppRoutes
  children: React.ReactNode
}) {
  const latest = React.useRef(value)
  latest.current = value
  const stable = React.useMemo<AppRoutes>(
    () => ({
      buildHref: (buildId) => latest.current.buildHref(buildId),
      artifactHref: (artifactId) => latest.current.artifactHref(artifactId),
    }),
    [],
  )
  return <RoutesContext.Provider value={stable}>{children}</RoutesContext.Provider>
}

/**
 * Read the active route builders. Returns {@link DEFAULT_ROUTES} when no
 * provider is mounted, so this is safe to call from any shared component
 * without requiring consumers to change.
 *
 * No warning is emitted when the default is in effect, deliberately: the
 * standalone app mounts no provider by design, so warning on it would fire
 * constantly in the primary consumer and train people to ignore the message.
 */
export function useRoutes(): AppRoutes {
  return React.useContext(RoutesContext)
}
