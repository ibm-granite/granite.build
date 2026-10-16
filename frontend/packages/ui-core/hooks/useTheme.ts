'use client'

import { useState, useCallback, useEffect } from 'react'
import {
  DARK_QUERY,
  THEME_ATTR,
  THEME_PREF_ATTR,
  THEME_STORAGE_KEY,
  parseStoredPreference,
  resolveTheme,
} from '../lib/themePreference'
import type { Theme, ThemePreference } from '../lib/themePreference'

export type { Theme, ThemePreference }

function systemPrefersDark(): boolean {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return false
  return window.matchMedia(DARK_QUERY).matches
}

// Writes the resolved theme and the preference behind it onto <html>, then
// persists the preference. "system" is stored as the *absence* of the key, so a
// browser that has never chosen and a deliberate "follow the system" choice
// behave identically.
function applyPreference(preference: ThemePreference) {
  if (typeof window === 'undefined') return
  const root = document.documentElement
  const resolved = resolveTheme(preference, systemPrefersDark())
  if (resolved === 'g10') root.removeAttribute(THEME_ATTR)
  else root.setAttribute(THEME_ATTR, resolved)
  root.setAttribute(THEME_PREF_ATTR, preference)
  try {
    if (preference === 'system') localStorage.removeItem(THEME_STORAGE_KEY)
    else localStorage.setItem(THEME_STORAGE_KEY, preference)
  } catch {
    // Private-mode or blocked storage: the theme still applies for this session.
  }
}

function readTheme(): Theme {
  if (typeof window === 'undefined') return 'g10'
  return document.documentElement.getAttribute(THEME_ATTR) === 'g100' ? 'g100' : 'g10'
}

function readPreference(): ThemePreference {
  if (typeof window === 'undefined') return 'system'
  return parseStoredPreference(document.documentElement.getAttribute(THEME_PREF_ATTR))
}

export function useTheme() {
  // Always starts at the light default, matching what the server rendered — SSR
  // can't know the stored preference or the OS setting, and mutating the DOM
  // during render (instead of in an effect) causes a React hydration mismatch on
  // <html>. The inline script in app/layout.tsx already set the real attributes
  // before hydration; the effect below just reads them back afterward, which is
  // an ordinary post-hydration state update, not something hydration validates
  // against.
  const [theme, setThemeState] = useState<Theme>('g10')
  const [preference, setPreferenceState] = useState<ThemePreference>('system')

  useEffect(() => {
    const sync = () => {
      setThemeState(readTheme())
      setPreferenceState(readPreference())
    }
    sync()
    // Both attributes live on <html>, so every useTheme() consumer observes one
    // shared source of truth instead of holding its own copy of the preference.
    const observer = new MutationObserver(sync)
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: [THEME_ATTR, THEME_PREF_ATTR],
    })
    return () => observer.disconnect()
  }, [])

  // An OS light/dark switch while the app is open should move the theme, but
  // only for someone actually following the system. Read through the DOM rather
  // than closing over `preference`, so the listener never goes stale.
  useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return
    const query = window.matchMedia(DARK_QUERY)
    const onChange = () => {
      if (readPreference() === 'system') applyPreference('system')
    }
    query.addEventListener('change', onChange)
    return () => query.removeEventListener('change', onChange)
  }, [])

  const setPreference = useCallback((next: ThemePreference) => applyPreference(next), [])

  // Retained for callers that just want to flip light/dark; pins an explicit
  // choice, the same as picking one from the menu.
  const toggleTheme = useCallback(() => {
    applyPreference(readTheme() === 'g100' ? 'g10' : 'g100')
  }, [])

  return { theme, preference, setPreference, toggleTheme }
}

function readChartsTheme(): 'white' | 'g100' {
  if (typeof window === 'undefined') return 'white'
  return document.documentElement.getAttribute(THEME_ATTR) === 'g100' ? 'g100' : 'white'
}

export function useChartsTheme(): 'white' | 'g100' {
  // Same SSR-safe-default-then-sync-in-effect pattern as useTheme() above.
  const [theme, setTheme] = useState<'white' | 'g100'>('white')
  useEffect(() => {
    setTheme(readChartsTheme())
    const observer = new MutationObserver(() => setTheme(readChartsTheme()))
    observer.observe(document.documentElement, { attributes: true, attributeFilter: [THEME_ATTR] })
    return () => observer.disconnect()
  }, [])
  return theme
}
