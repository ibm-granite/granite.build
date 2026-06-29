"use client";

import { useState, useEffect } from "react";
import { usePathname } from "next/navigation";
import { useRouter } from "next/navigation";
import Link from "next/link";
import {
  Header,
  HeaderGlobalAction,
  HeaderGlobalBar,
  HeaderMenuButton,
  HeaderName,
  HeaderPanel,
  SideNav,
  SideNavItems,
  SideNavLink,
  SkipToContent,
  Switcher,
  SwitcherDivider,
  SwitcherItem,
  Theme,
} from "@carbon/react";
import {
  Analytics,
  Activity,
  Product,
  Asleep,
  Dashboard,
  DataVis_4,
  DeliveryParcel,
  FlowModeler,
  Light,
  Switcher as SwitcherIcon,
  KubernetesPod,
} from "@carbon/icons-react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/auth/useAuth";
import { useTheme } from "@/hooks/useTheme";
import styles from "./AppHeader.module.scss";
import { getActiveEnv, setActiveEnv } from "@/config/activeEnv";
import type { EnvironmentEntry } from "@/types";

export function AppHeader() {
  const { logout, auth, isStandalone, environment } = useAuth();
  const { theme, toggleTheme } = useTheme();
  const pathname = usePathname();
  const router = useRouter();
  const queryClient = useQueryClient();
  const [isSideNavExpanded, setIsSideNavExpanded] = useState(false);
  const [isPanelOpen, setIsPanelOpen] = useState(false);

  // After client-side navigation the focused NavLink stays focused inside the
  // SideNav, so Carbon's onBlur never fires on its own. Blurring the active
  // element on route change triggers the SideNav's onBlur handler, which
  // resets expandedViaHoverState and fires onSideNavBlur so we can collapse.
  useEffect(() => {
    (document.activeElement as HTMLElement)?.blur();
  }, [pathname]);

  const { data: environments = [] } = useQuery<EnvironmentEntry[]>({
    queryKey: ["environments"],
    queryFn: () => fetch("/api/environments").then((r) => r.json()),
    staleTime: Infinity,
  });

  // Tracks the session-stored override; falls back to the server-configured default.
  const [activeEnvId, setActiveEnvId] = useState<string | null>(() =>
    getActiveEnv(),
  );
  const activeEnv =
    activeEnvId && environments.some((e) => e.id === activeEnvId)
      ? activeEnvId
      : environment;
  const activeEnvLabel =
    environments.find((e) => e.id === activeEnv)?.id ?? activeEnv;

  function switchEnv(env: string) {
    if (env === activeEnv) return;
    setActiveEnv(env);
    setActiveEnvId(env);
    // Clear all cached data so every active query re-fetches against the new env.
    queryClient.clear();
    router.refresh();
  }

  return (
    <>
      <Header aria-label="Granite.build" className={styles.headerActionIcons}>
        <SkipToContent />
        <HeaderMenuButton
          aria-label={isSideNavExpanded ? "Close menu" : "Open menu"}
          isActive={isSideNavExpanded}
          isCollapsible
          onClick={() => setIsSideNavExpanded((v) => !v)}
        />
        <HeaderName href="/" prefix="">
          Granite.build
          {environments.length > 1 && (
            <span className={styles.envBadge}>{activeEnvLabel}</span>
          )}
        </HeaderName>
        <HeaderGlobalBar>
          <HeaderGlobalAction
            aria-label={isPanelOpen ? "Close switcher" : "Open switcher"}
            aria-expanded={isPanelOpen}
            isActive={isPanelOpen}
            onClick={() => setIsPanelOpen((v) => !v)}
            tooltipAlignment="end"
          >
            <SwitcherIcon size={20} />
          </HeaderGlobalAction>
        </HeaderGlobalBar>
        <HeaderPanel
          expanded={isPanelOpen}
          onHeaderPanelFocus={() => setIsPanelOpen(false)}
        >
          <Switcher aria-label="Application switcher" expanded={isPanelOpen}>
            <p className={styles.sectionHeader}>Appearance</p>
            <SwitcherItem
              aria-label={
                theme === "g10"
                  ? "Switch to dark theme"
                  : "Switch to light theme"
              }
              onClick={() => {
                toggleTheme();
                setIsPanelOpen(false);
              }}
            >
              {theme === "g10" ? (
                <>
                  <Asleep
                    size={16}
                    style={{ marginRight: "0.5rem", verticalAlign: "middle" }}
                  />
                  Switch to dark theme
                </>
              ) : (
                <>
                  <Light
                    size={16}
                    style={{ marginRight: "0.5rem", verticalAlign: "middle" }}
                  />
                  Switch to light theme
                </>
              )}
            </SwitcherItem>
            {environments.length > 1 && (
              <>
                <SwitcherDivider />
                <p className={styles.sectionHeader}>Environment</p>
                {environments.map((env) => (
                  <SwitcherItem
                    key={env.id}
                    aria-label={env.label}
                    isSelected={env.id === activeEnv}
                    onClick={() => {
                      switchEnv(env.id);
                      setIsPanelOpen(false);
                    }}
                  >
                    {env.label}
                  </SwitcherItem>
                ))}
              </>
            )}
            {!isStandalone && (
              <>
                <SwitcherDivider />
                <p className={styles.sectionHeader}>Account</p>
                <SwitcherItem
                  aria-label="Sign out"
                  onClick={() => {
                    logout();
                    // Hard navigation prevents AuthGate's useEffect from racing
                    // with router.push and stripping the ?logged_out param.
                    window.location.href = "/login?logged_out=true";
                  }}
                >
                  Sign out{auth?.username ? ` (${auth.username})` : ""}
                </SwitcherItem>
              </>
            )}
          </Switcher>
        </HeaderPanel>
      </Header>
      <Theme theme={theme === "g10" ? "white" : "g100"}>
        <SideNav
          aria-label="Side navigation"
          isRail
          isPersistent
          expanded={isSideNavExpanded}
          onSideNavBlur={() => setIsSideNavExpanded(false)}
          className={styles.sideNav}
        >
          <SideNavItems>
            <SideNavLink
              as={Link}
              href="/dashboard"
              renderIcon={Dashboard}
              aria-current={pathname === "/dashboard" ? "page" : undefined}
            >
              Dashboard
            </SideNavLink>
            <SideNavLink
              as={Link}
              href="/builds"
              renderIcon={DeliveryParcel}
              aria-current={
                pathname === "/builds" || pathname.startsWith("/builds/")
                  ? "page"
                  : undefined
              }
            >
              Builds
            </SideNavLink>
            <SideNavLink
              as={Link}
              href="/plans"
              renderIcon={FlowModeler}
              aria-current={
                pathname === "/plans" || pathname.startsWith("/plans/")
                  ? "page"
                  : undefined
              }
            >
              Flight Plans
            </SideNavLink>
            <SideNavLink
              as={Link}
              href="/data-processing"
              renderIcon={DataVis_4}
              aria-current={
                pathname === "/data-processing" ? "page" : undefined
              }
            >
              Data Processing
            </SideNavLink>
            <SideNavLink
              as={Link}
              href="/artifacts"
              renderIcon={Product}
              aria-current={pathname === "/artifacts" ? "page" : undefined}
            >
              Artifacts
            </SideNavLink>
            <SideNavLink
              as={Link}
              href="/analytics"
              renderIcon={Analytics}
              aria-current={pathname === "/analytics" ? "page" : undefined}
            >
              Analytics
            </SideNavLink>
            <SideNavLink
              as={Link}
              href="/workloads"
              renderIcon={KubernetesPod}
              aria-current={pathname === "/workloads" ? "page" : undefined}
            >
              Workloads
            </SideNavLink>
          </SideNavItems>
        </SideNav>
      </Theme>
    </>
  );
}
