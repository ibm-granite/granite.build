/**
 * Parity test for the pre-hydration theme bootstrap in app/layout.tsx.
 *
 * That script must run before hydration so a dark-mode viewer never sees a white
 * flash, and an inline <head> script cannot import a module — so it duplicates
 * parseStoredPreference/resolveTheme as a string. This executes the real script
 * text out of layout.tsx against a stubbed DOM and asserts it agrees with
 * lib/themePreference.ts for every input, so the copy cannot drift unnoticed.
 *
 * Usage: node --test tests/theme-init-script.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')
const vm = require('node:vm')
const fs = require('node:fs')
const path = require('node:path')

const { parseStoredPreference, resolveTheme } = require('../../../packages/ui-core/lib/themePreference.ts')

const THEME_ATTR = 'data-carbon-theme'
const PREF_ATTR = 'data-theme-pref'

function extractScript() {
  const source = fs.readFileSync(path.join(__dirname, '../app/layout.tsx'), 'utf8')
  const match = source.match(/const THEME_INIT_SCRIPT = `([\s\S]*?)`\.trim\(\)/)
  assert.ok(match, 'THEME_INIT_SCRIPT template literal not found in app/layout.tsx')
  return match[1]
}

// Runs the real bootstrap text against a stubbed localStorage/matchMedia/document
// and returns the attributes it set on <html>.
function runScript(stored, systemDark) {
  const attributes = {}
  const sandbox = {
    localStorage: {
      getItem: (key) => (key === 'gb-ui-theme' ? stored : null),
      setItem: () => {},
      removeItem: () => {},
    },
    document: {
      documentElement: {
        setAttribute: (key, value) => {
          attributes[key] = value
        },
        removeAttribute: (key) => {
          delete attributes[key]
        },
      },
    },
  }
  sandbox.window = {
    matchMedia: (query) => ({ matches: query.includes('dark') ? systemDark : false }),
  }
  vm.createContext(sandbox)
  vm.runInContext(extractScript(), sandbox)
  return attributes
}

describe('theme bootstrap script', () => {
  it('applies dark on a dark system when nothing is stored', () => {
    const attrs = runScript(null, true)
    assert.equal(attrs[THEME_ATTR], 'g100')
    assert.equal(attrs[PREF_ATTR], 'system')
  })

  it('leaves the theme attribute unset on a light system when nothing is stored', () => {
    const attrs = runScript(null, false)
    assert.equal(attrs[THEME_ATTR], undefined)
    assert.equal(attrs[PREF_ATTR], 'system')
  })

  it('keeps an explicit light choice on a dark system', () => {
    const attrs = runScript('g10', true)
    assert.equal(attrs[THEME_ATTR], undefined)
    assert.equal(attrs[PREF_ATTR], 'g10')
  })

  it('keeps an explicit dark choice on a light system', () => {
    const attrs = runScript('g100', false)
    assert.equal(attrs[THEME_ATTR], 'g100')
    assert.equal(attrs[PREF_ATTR], 'g100')
  })

  it('survives a browser with no matchMedia', () => {
    const attributes = {}
    const sandbox = {
      localStorage: { getItem: () => null, setItem: () => {}, removeItem: () => {} },
      document: {
        documentElement: {
          setAttribute: (k, v) => {
            attributes[k] = v
          },
          removeAttribute: (k) => {
            delete attributes[k]
          },
        },
      },
      window: {},
    }
    vm.createContext(sandbox)
    vm.runInContext(extractScript(), sandbox)
    assert.equal(attributes[PREF_ATTR], 'system')
    assert.equal(attributes[THEME_ATTR], undefined)
  })

  // The point of this file: the inline copy must agree with the module for every
  // combination, since only the module is used after hydration.
  it('agrees with lib/themePreference.ts for every stored value and system setting', () => {
    for (const stored of [null, 'g10', 'g100', 'system', 'midnight', '']) {
      for (const systemDark of [true, false]) {
        const attrs = runScript(stored, systemDark)
        const preference = parseStoredPreference(stored)
        const expectedTheme = resolveTheme(preference, systemDark)
        assert.equal(
          attrs[PREF_ATTR],
          preference,
          `preference mismatch for stored=${JSON.stringify(stored)} systemDark=${systemDark}`
        )
        assert.equal(
          attrs[THEME_ATTR] ?? 'g10',
          expectedTheme,
          `theme mismatch for stored=${JSON.stringify(stored)} systemDark=${systemDark}`
        )
      }
    }
  })
})
