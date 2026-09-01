'use client'

import { useState, useCallback, useEffect } from 'react'

export type Theme = 'g10' | 'g100'

const STORAGE_KEY = 'gb-ui-theme'

function applyTheme(theme: Theme) {
  if (typeof window === 'undefined') return
  if (theme === 'g10') {
    document.documentElement.removeAttribute('data-carbon-theme')
  } else {
    document.documentElement.setAttribute('data-carbon-theme', theme)
  }
  localStorage.setItem(STORAGE_KEY, theme)
}

function readTheme(): Theme {
  if (typeof window === 'undefined') return 'g10'
  return (document.documentElement.getAttribute('data-carbon-theme') as Theme) ?? 'g10'
}

export function useTheme() {
  // Always starts at 'g10', matching what the server rendered — SSR can't know
  // the stored preference, and mutating the DOM during render (instead of in an
  // effect) causes a React hydration mismatch on <html>. The inline script in
  // app/layout.tsx already set the real attribute before hydration; the effect
  // below just reads it back afterward, which is an ordinary post-hydration
  // state update, not something hydration validates against.
  const [theme, setThemeState] = useState<Theme>('g10')

  useEffect(() => {
    setThemeState(readTheme())
    const observer = new MutationObserver(() => setThemeState(readTheme()))
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-carbon-theme'] })
    return () => observer.disconnect()
  }, [])

  const toggleTheme = useCallback(() => {
    const next: Theme = readTheme() === 'g100' ? 'g10' : 'g100'
    applyTheme(next)
  }, [])

  return { theme, toggleTheme }
}

function readChartsTheme(): 'white' | 'g100' {
  if (typeof window === 'undefined') return 'white'
  return document.documentElement.getAttribute('data-carbon-theme') === 'g100'
    ? 'g100'
    : 'white'
}

export function useChartsTheme(): 'white' | 'g100' {
  // Same SSR-safe-default-then-sync-in-effect pattern as useTheme() above.
  const [theme, setTheme] = useState<'white' | 'g100'>('white')
  useEffect(() => {
    setTheme(readChartsTheme())
    const observer = new MutationObserver(() => setTheme(readChartsTheme()))
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-carbon-theme'] })
    return () => observer.disconnect()
  }, [])
  return theme
}
