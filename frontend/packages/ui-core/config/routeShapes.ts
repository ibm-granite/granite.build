/**
 * Where shared components should link for builds and artifacts — the pure half.
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
 *
 * This module is deliberately **not** `'use client'`: the context, provider and
 * hook live in `routes.tsx`, which is. Keeping the interface and the default
 * builders here means a Server Component, `generateMetadata`, or a server-rendered
 * breadcrumb can build a link without pulling in client-only code — and it means
 * the builders are unit-testable without a React harness, which the frontend
 * workspace does not have. `routes.tsx` re-exports both names, so importing from
 * either path works.
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
 *
 * Ids are percent-encoded. `AppRoutes` types them as `string`, not as a UUID, so
 * an id carrying `&`, `#`, `+` or a space would otherwise corrupt on the way back
 * out: `searchParams.get('id')` returns only the text before an `&`, and a bare
 * `+` decodes to a space. Every id in use today is a UUID and unaffected; doing
 * this while there is exactly one implementation is much cheaper than doing it
 * once several consumers have their own.
 */
export const DEFAULT_ROUTES: AppRoutes = {
  buildHref: (buildId) => `/dashboard/builds/_/?id=${encodeURIComponent(buildId)}`,
  artifactHref: (artifactId) =>
    `/dashboard/artifacts/_/?id=${encodeURIComponent(artifactId)}`,
}
