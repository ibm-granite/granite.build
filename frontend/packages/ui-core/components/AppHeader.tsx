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
  Checkmark,
  Dashboard,
  DeliveryParcel,
  Light,
  Pipelines,
  ModelTuned,
  Screen,
  Switcher as SwitcherIcon,
} from "@carbon/icons-react";
import { useTheme } from "../hooks/useTheme";
import type { ThemePreference } from "../lib/themePreference";
import styles from "./AppHeader.module.scss";

// Our own ids rather than Carbon's internal class names, so the outside-click
// check below doesn't break when Carbon renames a class.
const SIDE_NAV_ID = "gb-side-nav";
const SIDE_NAV_TOGGLE_ID = "gb-side-nav-toggle";

const THEME_OPTIONS: { preference: ThemePreference; label: string; Icon: typeof Light }[] = [
  { preference: "system", label: "Use system theme", Icon: Screen },
  { preference: "g10", label: "Light", Icon: Light },
  { preference: "g100", label: "Dark", Icon: Asleep },
];

export function AppHeader() {
  const { theme, preference, setPreference } = useTheme();
  const pathname = usePathname();
  const [isSideNavExpanded, setIsSideNavExpanded] = useState(false);
  const [isPanelOpen, setIsPanelOpen] = useState(false);

  useEffect(() => {
    (document.activeElement as HTMLElement)?.blur();
    // Carbon re-expands the rail on any click inside it, a nav link included, so
    // a navigation would otherwise leave the menu sitting open over the new page.
    setIsSideNavExpanded(false);
  }, [pathname]);

  // Carbon's rail handles hover-out itself, but only for clicks and Escape that
  // land inside the nav. Neither covers a click on the page behind it: a plain
  // click on non-focusable content moves no focus, so onSideNavBlur never fires.
  useEffect(() => {
    if (!isSideNavExpanded) return;
    const onPointerDown = (event: PointerEvent) => {
      const target = event.target as Element | null;
      if (target?.closest(`#${SIDE_NAV_ID}`)) return;
      // The toggle's own onClick closes it; acting here too would double-fire.
      if (target?.closest(`#${SIDE_NAV_TOGGLE_ID}`)) return;
      setIsSideNavExpanded(false);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setIsSideNavExpanded(false);
    };
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [isSideNavExpanded]);

  return (
    <>
      <Header aria-label="Granite.build" className={styles.headerActionIcons}>
        <SkipToContent />
        <HeaderMenuButton
          id={SIDE_NAV_TOGGLE_ID}
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
            {THEME_OPTIONS.map(({ preference: option, label, Icon }) => (
              <SwitcherItem
                key={option}
                aria-label={label}
                aria-current={preference === option ? "true" : undefined}
                onClick={() => {
                  setPreference(option);
                  setIsPanelOpen(false);
                }}
              >
                <Icon
                  size={16}
                  style={{ marginRight: "0.5rem", verticalAlign: "middle" }}
                />
                {label}
                {preference === option && (
                  <Checkmark
                    size={16}
                    style={{ marginLeft: "0.5rem", verticalAlign: "middle" }}
                  />
                )}
              </SwitcherItem>
            ))}
          </Switcher>
        </HeaderPanel>
      </Header>
      <Theme theme={theme === "g10" ? "white" : "g100"}>
        <SideNav
          id={SIDE_NAV_ID}
          aria-label="Side navigation"
          isRail
          isPersistent
          expanded={isSideNavExpanded}
          // Carbon routes its rail listeners — hover in/out, focus/blur, Escape —
          // through onToggle. Leaving it unwired was why mouse-leave never closed
          // the nav: Carbon cleared its internal hover flag while the controlled
          // `expanded` above kept the nav pinned open. Both params are optional so
          // the handler satisfies Carbon's (event, value) signature and the native
          // <details> ToggleEventHandler that HTMLAttributes adds once `id` is set.
          onToggle={(_event?: unknown, value?: boolean) => {
            if (typeof value === "boolean") setIsSideNavExpanded(value);
          }}
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
            <SideNavLink
              as={Link}
              href="/dashboard/autotunex"
              renderIcon={ModelTuned}
              aria-current={
                pathname === "/dashboard/autotunex" || pathname.startsWith("/dashboard/autotunex/")
                  ? "page"
                  : undefined
              }
            >
              Model Customization
            </SideNavLink>
          </SideNavItems>
        </SideNav>
      </Theme>
    </>
  );
}
