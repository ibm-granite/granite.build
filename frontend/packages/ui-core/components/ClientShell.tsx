"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { Theme } from "@carbon/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useTheme } from "../hooks/useTheme";
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
// Unlike builds/artifacts, /dashboard/autotunex has sibling static routes
// (settings, start-tuning) sharing the prefix, so those are excluded from the
// id match — otherwise a direct load of e.g. /dashboard/autotunex/settings would
// be wrongly redirected to /dashboard/autotunex/_/?id=settings.
const AUTOTUNEX_RESERVED = new Set(["_", "settings", "start-tuning"]);

function useDeepLinkRedirect() {
  const router = useRouter();
  useEffect(() => {
    const path = window.location.pathname;
    const buildMatch = path.match(/^\/dashboard\/builds\/([^/]+)\/?$/);
    if (buildMatch && buildMatch[1] !== "_") {
      router.replace(`/dashboard/builds/_/?id=${buildMatch[1]}`);
      return;
    }
    const artifactMatch = path.match(/^\/dashboard\/artifacts\/([^/]+)\/?$/);
    if (artifactMatch && artifactMatch[1] !== "_") {
      router.replace(`/dashboard/artifacts/_/?id=${artifactMatch[1]}`);
      return;
    }
    const tuningMatch = path.match(/^\/dashboard\/autotunex\/([^/]+)\/?$/);
    if (tuningMatch && !AUTOTUNEX_RESERVED.has(tuningMatch[1])) {
      router.replace(`/dashboard/autotunex/_/?id=${tuningMatch[1]}`);
    }
  }, [router]);
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

export function ClientShell({ children }: { children: React.ReactNode }) {
  return (
    <QueryClientProvider client={queryClient}>
      <AppShell>{children}</AppShell>
    </QueryClientProvider>
  );
}
