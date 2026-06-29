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
  const [theme, setThemeState] = useState<Theme>(() => {
    if (typeof window === 'undefined') return 'g10'
    const stored = (localStorage.getItem(STORAGE_KEY) as Theme) ?? 'g10'
    applyTheme(stored)
    return stored
  })

  // Keep all useTheme instances in sync by observing the attribute —
  // the same pattern useChartsTheme uses.
  useEffect(() => {
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
  const [theme, setTheme] = useState<'white' | 'g100'>(readChartsTheme)
  useEffect(() => {
    const observer = new MutationObserver(() => setTheme(readChartsTheme()))
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-carbon-theme'] })
    return () => observer.disconnect()
  }, [])
  return theme
}
