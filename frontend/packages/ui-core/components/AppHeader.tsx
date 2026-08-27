"use client";

import { useState, useEffect } from "react";
import { usePathname } from "next/navigation";
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
  SwitcherItem,
  Theme,
} from "@carbon/react";
import {
  Analytics,
  Product,
  Asleep,
  Dashboard,
  DeliveryParcel,
  Light,
  Pipelines,
  Switcher as SwitcherIcon,
} from "@carbon/icons-react";
import { useTheme } from "../hooks/useTheme";
import styles from "./AppHeader.module.scss";

export function AppHeader() {
  const { theme, toggleTheme } = useTheme();
  const pathname = usePathname();
  const [isSideNavExpanded, setIsSideNavExpanded] = useState(false);
  const [isPanelOpen, setIsPanelOpen] = useState(false);

  useEffect(() => {
    (document.activeElement as HTMLElement)?.blur();
  }, [pathname]);

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
        <HeaderName as={Link} href="/dashboard" prefix="">
          Granite.build
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
              href="/dashboard/builds"
              renderIcon={DeliveryParcel}
              aria-current={
                pathname === "/dashboard/builds" || pathname.startsWith("/dashboard/builds/")
                  ? "page"
                  : undefined
              }
            >
              Builds
            </SideNavLink>
            <SideNavLink
              as={Link}
              href="/dashboard/data-processing"
              renderIcon={Pipelines}
              aria-current={
                pathname === "/dashboard/data-processing" ? "page" : undefined
              }
            >
              Data Processing
            </SideNavLink>
            <SideNavLink
              as={Link}
              href="/dashboard/artifacts"
              renderIcon={Product}
              aria-current={
                pathname === "/dashboard/artifacts" || pathname.startsWith("/dashboard/artifacts/")
                  ? "page"
                  : undefined
              }
            >
              Artifacts
            </SideNavLink>
            <SideNavLink
              as={Link}
              href="/dashboard/analytics"
              renderIcon={Analytics}
              aria-current={pathname === "/dashboard/analytics" ? "page" : undefined}
            >
              Analytics
            </SideNavLink>
          </SideNavItems>
        </SideNav>
      </Theme>
    </>
  );
}
