"use client";

import { useEffect, useState } from "react";
import { useRouter, usePathname } from "next/navigation";
import { SkeletonText, Theme } from "@carbon/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AuthProvider } from "@/auth/AuthProvider";
import type { RuntimeConfig } from "@/auth/AuthProvider";
import { useAuth } from "@/auth/useAuth";
import { useTheme } from "@/hooks/useTheme";
import { AppHeader } from "@/components/AppHeader";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 15_000, retry: 1 },
  },
});

const PUBLIC_PATHS = ["/login", "/auth/callback", "/auth/trampoline", "/auth/ibmid-callback"];

const DEFAULT_CONFIG: RuntimeConfig = {
  environment: "STANDALONE",
  authProvider: "apikey",
  githubClientId: "",
};

function AuthGate({ children }: { children: React.ReactNode }) {
  const [mounted, setMounted] = useState(false);
  const { isAuthenticated } = useAuth();
  const { theme } = useTheme();
  const router = useRouter();
  const pathname = usePathname();

  const isPublic = PUBLIC_PATHS.some((p) => pathname.startsWith(p));

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!mounted || isPublic) return;
    if (!isAuthenticated) {
      if (pathname !== "/dashboard") {
        sessionStorage.setItem("gb-ui-pre-auth-url", pathname + window.location.search);
      }
      router.replace("/login");
    }
  }, [mounted, isAuthenticated, isPublic, pathname, router]);

  if (!mounted) return <SkeletonText />;

  if (!isAuthenticated && !isPublic) return <SkeletonText />;

  if (isPublic) return <>{children}</>;

  return (
    <>
      <AppHeader />
      <Theme theme={theme}>
        <div style={{ paddingTop: "3rem", paddingLeft: "3rem" }}>
          {children}
        </div>
      </Theme>
    </>
  );
}

export function ClientShell({ children }: { children: React.ReactNode }) {
  const [runtimeConfig, setRuntimeConfig] = useState<RuntimeConfig | null>(null);

  useEffect(() => {
    fetch("/api/config")
      .then((r) => r.json())
      .then((cfg: RuntimeConfig) => setRuntimeConfig(cfg))
      .catch(() => setRuntimeConfig(DEFAULT_CONFIG));
  }, []);

  if (!runtimeConfig) return <SkeletonText />;

  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider runtimeConfig={runtimeConfig}>
        <AuthGate>{children}</AuthGate>
      </AuthProvider>
    </QueryClientProvider>
  );
}
