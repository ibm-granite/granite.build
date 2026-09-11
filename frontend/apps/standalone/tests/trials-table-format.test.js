/**
 * Tests for the trials-table cell display formatting.
 *
 * These live in their own module rather than inside TrialsTable.tsx because the
 * toolbar search filters on the SAME strings the cells render — Carbon's default
 * filter matches raw cell values, so searching "5m 20" would miss a row whose
 * Total time cell shows "5m 20s" but whose underlying value is 320. One
 * formatter feeds both, and this file is what keeps it honest. (The frontend
 * test harness has no jsdom and cannot require a .tsx file, so a formatter
 * defined inside the component would be untestable.)
 *
 * Note on the created_at case: toLocaleString() output depends on the runner's
 * locale and timezone, so asserting a literal date string would fail in CI or
 * on another machine. The test instead asserts the value equals the same
 * toLocaleString() call and is NOT the raw ISO input — which is what actually
 * matters: that the branch ran at all.
 */
const { describe, it } = require('node:test')
const assert = require('node:assert/strict')

const { formatCell, formatTime } = require('../../../packages/ui-core/components/trialsTableFormat.ts')

describe('formatTime', () => {
  it('renders minutes and seconds past a minute', () => {
    assert.equal(formatTime(320), '5m 20s')
  })

  it('renders bare seconds under a minute', () => {
    assert.equal(formatTime(45), '45s')
  })

  it('floors fractional seconds', () => {
    assert.equal(formatTime(45.9), '45s')
  })

  it('treats zero and negatives as zero', () => {
    assert.equal(formatTime(0), '0 s')
    assert.equal(formatTime(-5), '0 s')
  })
})

describe('formatCell', () => {
  it('renders created_at through the locale formatter, not as raw ISO', () => {
    const iso = '2026-09-11T16:02:00.000Z'
    const out = formatCell('created_at', iso)
    // Three independent checks, none of which pin the runner's locale: the
    // branch produced a human date (contains the year), it is not the raw
    // input, and it agrees with the formatter the cells use.
    assert.match(out, /2026/)
    assert.notEqual(out, iso)
    assert.equal(out, new Date(iso).toLocaleString())
  })

  it('renders loss to four decimal places', () => {
    assert.equal(formatCell('loss', 0.42131), '0.4213')
  })

  it('renders total_time as a duration', () => {
    assert.equal(formatCell('total_time', 320), '5m 20s')
  })

  it('passes strings through untouched', () => {
    assert.equal(formatCell('status', 'completed'), 'completed')
    assert.equal(formatCell('id', 'trial-0002'), 'trial-0002')
  })

  it('renders a missing value as an em dash', () => {
    assert.equal(formatCell('loss', undefined), '—')
    assert.equal(formatCell('total_time', null), '—')
  })

  it('does not format a non-numeric loss or total_time', () => {
    // Guards the `typeof value === 'number'` conditions: a string slipping
    // through must not reach toFixed/formatTime and throw.
    assert.equal(formatCell('loss', 'n/a'), 'n/a')
    assert.equal(formatCell('total_time', 'n/a'), 'n/a')
  })

  it('sends an empty created_at through the date formatter, not the em-dash fallback', () => {
    // Guards the branch order. Hoisting the null/undefined check above the
    // created_at branch would return '—' here and silently change what the
    // Created on column shows for a trial with no timestamp.
    assert.notEqual(formatCell('created_at', undefined), '—')
    assert.notEqual(formatCell('created_at', null), '—')
    assert.equal(formatCell('created_at', undefined), new Date(undefined).toLocaleString())
    assert.equal(formatCell('created_at', null), new Date(null).toLocaleString())
  })
})
