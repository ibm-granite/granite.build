/**
 * Tests for the HuggingFace model-card front-matter strip.
 *
 * `---` is both YAML's front-matter delimiter and Markdown's thematic break, and
 * HF model cards use the latter freely as a section rule. Toggling an
 * "inside front matter" flag on every `---` line therefore hid the body between a
 * card's rules: a typical README lost everything between its third and fourth
 * `---`, with no sign anything was missing.
 *
 * Usage: node --test tests/model-card.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')

const { stripFrontMatter } = require('../../../packages/ui-core/lib/autotunex/modelCard.ts')

describe('stripFrontMatter', () => {
  it('removes a leading YAML block', () => {
    const raw = ['---', 'license: apache-2.0', 'language:', '  - en', '---', '', '# Granite'].join('\n')
    assert.equal(stripFrontMatter(raw), '# Granite')
  })

  it('keeps body text separated by thematic breaks', () => {
    // The regression: everything between the 3rd and 4th `---` used to vanish.
    const raw = [
      '---',
      'license: apache-2.0',
      '---',
      '',
      '# Granite',
      'Intro paragraph.',
      '',
      '---',
      '',
      '## Training data',
      'This section used to disappear.',
      '',
      '---',
      '',
      '## Evaluation',
      'So did the rule above this one.',
    ].join('\n')

    const out = stripFrontMatter(raw)
    assert.ok(out.includes('## Training data'), 'section after the first body rule survives')
    assert.ok(out.includes('This section used to disappear.'))
    assert.ok(out.includes('## Evaluation'), 'section after the second body rule survives')
    assert.ok(!out.includes('license: apache-2.0'), 'front matter is still stripped')
    // The rules themselves are body content and stay.
    assert.equal(out.split('\n').filter((l) => l.trim() === '---').length, 2)
  })

  it('leaves a document with no front matter alone', () => {
    const raw = '# Granite\n\nNo front matter here.\n\n---\n\nStill here.'
    assert.equal(stripFrontMatter(raw), raw.trim())
  })

  it('returns the document whole when the opener never closes', () => {
    // Not front matter at all — discarding everything would blank the card.
    const raw = '---\n# Granite\nA card that opens a rule and never closes it.'
    assert.equal(stripFrontMatter(raw), raw.trim())
  })

  it('tolerates blank lines before the opener', () => {
    const raw = '\n\n---\nlicense: mit\n---\n\n# Body'
    assert.equal(stripFrontMatter(raw), '# Body')
  })

  it('handles an empty document and a front-matter-only document', () => {
    assert.equal(stripFrontMatter(''), '')
    assert.equal(stripFrontMatter('---\nlicense: mit\n---\n'), '')
  })
})
