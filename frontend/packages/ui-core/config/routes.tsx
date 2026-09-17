'use client'

import React from 'react'

/**
 * Where shared components should link for builds and artifacts.
 *
 * ui-core is consumed by more than one app, and those apps do not agree on URL
 * shape: this repo's standalone app serves a single detail shell per resource
 * and passes the id as a query param (`/dashboard/builds/_/?id=…`), while the
 * internal gb-ui deployment uses path segments (`/builds/:id`). Components that
 * hardcoded the former could not be shared at all — which is why gb-ui keeps
 * local copies of BuildsTable and several detail panels that are otherwise
 * identical to the ones here.
 *
 * Injecting the link shape instead removes that reason to fork. Consumers that
 * say nothing keep the standalone scheme, so adding this changes no behaviour.
 */
export interface AppRoutes {
  /** Detail page for one build. */
  buildHref(buildId: string): string
  /** Detail page for one artifact. */
  artifactHref(artifactId: string): string
}

/**
 * The standalone app's scheme, and the fallback when no provider is mounted.
 *
 * Keeping this as the default is deliberate: it means introducing the seam is a
 * no-op for every existing call site, and a consumer opts in by mounting a
 * provider rather than by being migrated.
 */
export const DEFAULT_ROUTES: AppRoutes = {
  buildHref: (buildId) => `/dashboard/builds/_/?id=${buildId}`,
  artifactHref: (artifactId) => `/dashboard/artifacts/_/?id=${artifactId}`,
}

const RoutesContext = React.createContext<AppRoutes>(DEFAULT_ROUTES)

/**
 * Override the link shape for everything below. Mount once, near the root.
 *
 * `value` is memoised by the caller or, more usually, a module-level constant —
 * an object literal written inline here would be a new reference every render
 * and would re-render every consumer.
 */
export function RoutesProvider({
  value,
  children,
}: {
  value: AppRoutes
  children: React.ReactNode
}) {
  return <RoutesContext.Provider value={value}>{children}</RoutesContext.Provider>
}

/**
 * Read the active route builders. Returns {@link DEFAULT_ROUTES} when no
 * provider is mounted, so this is safe to call from any shared component
 * without requiring consumers to change.
 */
export function useRoutes(): AppRoutes {
  return React.useContext(RoutesContext)
}
