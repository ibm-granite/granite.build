// Theme preference resolution, kept free of DOM access so it can be unit-tested.
//
// Preference is tri-state. "system" (the default, and what an absent stored
// value means) follows the OS via prefers-color-scheme; an explicit 'g10' or
// 'g100' pins the theme and keeps winning over the OS until the user chooses to
// follow the system again.

export type Theme = 'g10' | 'g100'
export type ThemePreference = 'system' | Theme

export const THEME_STORAGE_KEY = 'gb-ui-theme'
// Resolved theme, read by Carbon and by the charts hook. Absent means g10.
export const THEME_ATTR = 'data-carbon-theme'
// The preference behind it. Held on <html> as well so every useTheme() consumer
// observes one shared source of truth, the same way the resolved theme already is.
export const THEME_PREF_ATTR = 'data-theme-pref'
export const DARK_QUERY = '(prefers-color-scheme: dark)'

// Values written by the previous light/dark-only version parse unchanged, so an
// existing explicit choice survives the upgrade instead of resetting to system.
export function parseStoredPreference(raw: string | null): ThemePreference {
  if (raw === 'g10' || raw === 'g100' || raw === 'system') return raw
  return 'system'
}

export function resolveTheme(preference: ThemePreference, systemPrefersDark: boolean): Theme {
  if (preference === 'system') return systemPrefersDark ? 'g100' : 'g10'
  return preference
}
