"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { Theme } from "@carbon/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useTheme } from "../hooks/useTheme";
import { type AppRoutes, RoutesProvider, useRoutes } from "../config/routes";
import { AppHeader } from "./AppHeader";
import { ChatWidget } from "./ChatWidget";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 15_000, retry: 1 },
  },
});

// Direct/bookmarked links to a build, artifact, or tuning (e.g. from
// `gb build status <id>`, or a refresh of the cosmetic pretty URL these pages
// push via history.replaceState) hit gbserver's SPA-fallback 404 handler, which
// always serves the dashboard shell — only the literal "_" path is statically
// generated for these dynamic routes. This redirects client-side to the real
// pre-rendered detail route + id query param, matching the convention
// BuildDetailPageClient/ArtifactDetailPageClient/TuningDetailPageClient expect
// (a query param, not a hash, so useSearchParams() picks it up reactively even
// when navigating between two instances of the same "_" route).
//
// The *targets* go through the route seam rather than being spelled out here.
// They were byte-identical to the two builders in config/routeShapes.ts, which
// meant ui-core held two independent copies of the same URL shape while claiming
// one — so a consumer that injected its own scheme still got redirected to the
// standalone one by this hook. The regexes stay standalone-specific on purpose:
// they describe the paths gbserver's SPA fallback serves, and a consumer with real
// path-segment routes simply never matches, which is the correct no-op for it.
//
// The matched segment is decoded before it reaches a builder. `pathname` is
// percent-encoded and the builders encode again, so passing it through raw
// double-encodes: `/dashboard/builds/a%20b` would redirect to `?id=a%2520b`,
// which `searchParams.get('id')` reads back as the literal `a%20b` — the wrong
// id. UUIDs are unaffected, but ids carrying a reserved character are exactly
// the ones the encoding was added for, so the two halves have to agree.
function decodeSegment(segment: string): string {
  try {
    return decodeURIComponent(segment);
  } catch {
    // A stray `%` is not a valid escape and makes decodeURIComponent throw.
    // Redirecting with the raw segment is wrong in the same way it was before
    // this fix; throwing out of an effect would blank the whole shell.
    return segment;
  }
}

// Unlike builds/artifacts, /dashboard/autotunex has sibling static routes
// (settings, start-tuning) sharing the prefix, so those are excluded from the
// id match — otherwise a direct load of e.g. /dashboard/autotunex/settings would
// be wrongly redirected to /dashboard/autotunex/_/?id=settings.
const AUTOTUNEX_RESERVED = new Set(["_", "settings", "start-tuning"]);

function useDeepLinkRedirect() {
  const router = useRouter();
  const routes = useRoutes();
  useEffect(() => {
    const path = window.location.pathname;
    const buildMatch = path.match(/^\/dashboard\/builds\/([^/]+)\/?$/);
    if (buildMatch && buildMatch[1] !== "_") {
      router.replace(routes.buildHref(decodeSegment(buildMatch[1])));
      return;
    }
    const artifactMatch = path.match(/^\/dashboard\/artifacts\/([^/]+)\/?$/);
    if (artifactMatch && artifactMatch[1] !== "_") {
      router.replace(routes.artifactHref(decodeSegment(artifactMatch[1])));
      return;
    }
    const tuningMatch = path.match(/^\/dashboard\/autotunex\/([^/]+)\/?$/);
    if (tuningMatch && !AUTOTUNEX_RESERVED.has(tuningMatch[1])) {
      router.replace(`/dashboard/autotunex/_/?id=${tuningMatch[1]}`);
    }
  }, [router, routes]);
}

function AppShell({ children }: { children: React.ReactNode }) {
  const { theme } = useTheme();
  useDeepLinkRedirect();

  return (
    <>
      <AppHeader />
      <Theme theme={theme}>
        <div style={{ paddingTop: "3rem", paddingLeft: "3rem" }}>
          {children}
        </div>
      </Theme>
      <ChatWidget />
    </>
  );
}

/**
 * The shared app root: query client, theme, header, chat widget.
 *
 * `routes` is the injection point for the link-shape seam. It has to be here
 * because this is the only root ui-core owns — a consumer wrapping ClientShell
 * from outside would sit above the provider but below nothing, leaving every
 * ui-core component inside it (including useDeepLinkRedirect above) reading the
 * standalone default. Omitting it keeps that default, which is what the
 * standalone app wants.
 */
export function ClientShell({
  children,
  routes,
}: {
  children: React.ReactNode;
  routes?: AppRoutes;
}) {
  const shell = (
    <QueryClientProvider client={queryClient}>
      <AppShell>{children}</AppShell>
    </QueryClientProvider>
  );
  return routes ? <RoutesProvider value={routes}>{shell}</RoutesProvider> : shell;
}
