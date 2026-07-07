"use client";

import { Theme } from "@carbon/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useTheme } from "@/hooks/useTheme";
import { AppHeader } from "@/components/AppHeader";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 15_000, retry: 1 },
  },
});

function AppShell({ children }: { children: React.ReactNode }) {
  const { theme } = useTheme();

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
  return (
    <QueryClientProvider client={queryClient}>
      <AppShell>{children}</AppShell>
    </QueryClientProvider>
  );
}
