/**
 * Tests for the theme preference resolution behind the system-theme support.
 *
 * The app previously stored only a resolved theme ('g10' | 'g100') under
 * `gb-ui-theme` and defaulted to light, ignoring the OS setting. Preference is
 * now tri-state: absent means "follow the system", and an explicit light/dark
 * choice pins the theme and must keep winning over the OS. Existing installs
 * already have 'g10' or 'g100' stored, so those values must keep parsing as
 * explicit choices rather than being discarded.
 *
 * Usage: node --test tests/theme-preference.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')

const { parseStoredPreference, resolveTheme } = require('../../../packages/ui-core/lib/themePreference.ts')

describe('parseStoredPreference', () => {
  it('treats a missing value as following the system', () => {
    assert.equal(parseStoredPreference(null), 'system')
  })

  it('keeps an explicit dark choice written by the previous version', () => {
    assert.equal(parseStoredPreference('g100'), 'g100')
  })

  it('keeps an explicit light choice written by the previous version', () => {
    assert.equal(parseStoredPreference('g10'), 'g10')
  })

  it('accepts an explicitly stored system preference', () => {
    assert.equal(parseStoredPreference('system'), 'system')
  })

  it('falls back to the system for an unrecognised value', () => {
    assert.equal(parseStoredPreference('midnight'), 'system')
  })
})

describe('resolveTheme', () => {
  it('follows a dark system setting when no explicit choice is stored', () => {
    assert.equal(resolveTheme('system', true), 'g100')
  })

  it('follows a light system setting when no explicit choice is stored', () => {
    assert.equal(resolveTheme('system', false), 'g10')
  })

  it('keeps an explicit light choice on a dark system', () => {
    assert.equal(resolveTheme('g10', true), 'g10')
  })

  it('keeps an explicit dark choice on a light system', () => {
    assert.equal(resolveTheme('g100', false), 'g100')
  })
})
