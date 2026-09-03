/**
 * Regression tests for dataset record parsing and counting.
 *
 * Three related defects:
 *
 * 1. parseCsv() split the file on '\n' BEFORE quote-aware field parsing, so a
 *    quoted field containing a newline (common in instruction datasets, where
 *    an `output`/`response` column holds a multi-line answer) was sheared into
 *    several records and every continuation fragment was misaligned against the
 *    headers — silently corrupting the preview, the column metadata, the
 *    AI-mapping samples and the record count.
 *
 * 2. The >5MB counting path counted physical LINES (header row included),
 *    while the <=5MB path counted parseCsv() records. A 6MB CSV therefore
 *    reported N+1 where a 4MB one reported N.
 *
 * 3. For a pretty-printed .json array, line counting is not even the right
 *    unit: 1,000 objects over ~7,000 lines reported ~7,000 records, which is
 *    what the wizard displayed and what the train/validation split maths used.
 *
 * Usage: node --test tests/csv-record-count.test.js
 */

const { describe, it } = require('node:test')
const assert = require('node:assert/strict')

const {
  parseCsv,
  countRecordsInBlob,
  createCsvScanState,
  scanCsvChunk,
} = require('../../../packages/ui-core/lib/autotunex/file-parser.ts')

// A quoted value spanning three physical lines, plus a doubled ("") escape.
const MULTILINE_CSV = [
  'instruction,output',
  '"Summarise this","Line one',
  'Line two',
  'Line three"',
  '"Quote me","He said ""hello"" twice"',
].join('\n')

describe('parseCsv — newlines inside quoted fields', () => {
  it('keeps a multi-line quoted value in a single record', () => {
    const rows = parseCsv(MULTILINE_CSV)
    assert.equal(rows.length, 2, 'two data records, not five sheared fragments')
    assert.equal(rows[0].instruction, 'Summarise this')
    assert.equal(rows[0].output, 'Line one\nLine two\nLine three')
  })

  it('does not misalign the columns of the following record', () => {
    const rows = parseCsv(MULTILINE_CSV)
    assert.equal(rows[1].instruction, 'Quote me')
    assert.equal(rows[1].output, 'He said "hello" twice')
  })

  it('reports every header key on every row', () => {
    for (const row of parseCsv(MULTILINE_CSV)) {
      assert.deepEqual(Object.keys(row), ['instruction', 'output'])
    }
  })

  it('still honours maxLines, counted in records not lines', () => {
    assert.equal(parseCsv(MULTILINE_CSV, 1).length, 1)
  })

  it('handles CRLF line endings', () => {
    const rows = parseCsv('a,b\r\n1,2\r\n3,4')
    assert.equal(rows.length, 2)
    assert.deepEqual(rows[0], { a: '1', b: '2' })
    assert.deepEqual(rows[1], { a: '3', b: '4' })
  })

  it('returns [] for empty content', () => {
    assert.deepEqual(parseCsv(''), [])
  })
})

describe('scanCsvChunk — record boundaries across chunk boundaries', () => {
  // The streaming counter feeds arbitrary chunks, so a quoted newline (or a ""
  // escape) can straddle a chunk edge. Quote state must carry in the state obj.
  it('carries quote state across two chunks', () => {
    const state = createCsvScanState()
    const records = []
    scanCsvChunk('a,b\n"one', state, (r) => records.push(r))
    scanCsvChunk('\ntwo",x\n', state, (r) => records.push(r))
    assert.deepEqual(records, ['a,b', '"one\ntwo",x'])
  })

  it('treats a "" escape split across chunks as still-quoted', () => {
    const state = createCsvScanState()
    const records = []
    scanCsvChunk('h\n"a"', state, (r) => records.push(r))
    scanCsvChunk('"b\nc"\n', state, (r) => records.push(r))
    assert.deepEqual(records, ['h', '"a""b\nc"'])
  })
})

const blob = (text) => new Blob([text])

describe('countRecordsInBlob — CSV', () => {
  it('excludes the header row', async () => {
    assert.equal(await countRecordsInBlob(blob('a,b\n1,2\n3,4\n'), 'd.csv'), 2)
  })

  it('counts a multi-line quoted record once', async () => {
    assert.equal(await countRecordsInBlob(blob(MULTILINE_CSV), 'd.csv'), 2)
  })

  it('agrees with parseCsv().length — the two paths must not diverge', async () => {
    for (const text of [MULTILINE_CSV, 'a,b\n1,2\n3,4\n', 'only,headers\n']) {
      assert.equal(
        await countRecordsInBlob(blob(text), 'd.csv'),
        parseCsv(text).length,
        `count must match parse for: ${JSON.stringify(text.slice(0, 30))}`
      )
    }
  })

  it('reports 0, not -1, for an empty file', async () => {
    assert.equal(await countRecordsInBlob(blob(''), 'd.csv'), 0)
  })
})

describe('countRecordsInBlob — JSONL', () => {
  it('counts valid JSON objects and skips invalid lines', async () => {
    const text = '{"a":1}\n{"a":2}\nnot json\n\n{"a":3}\n'
    assert.equal(await countRecordsInBlob(blob(text), 'd.jsonl'), 3)
  })

  it('is unaffected by quote characters inside the JSON', async () => {
    // The CSV scanner must not be applied to JSONL: a quoted string would
    // otherwise toggle quote state and merge two records into one.
    const text = '{"t":"a \\"quoted\\" word"}\n{"t":"another"}\n'
    assert.equal(await countRecordsInBlob(blob(text), 'd.jsonl'), 2)
  })
})

describe('countRecordsInBlob — JSON array', () => {
  it('counts array entries, not lines, for a pretty-printed file', async () => {
    const rows = Array.from({ length: 25 }, (_, i) => ({ id: i, text: 'x' }))
    const pretty = JSON.stringify(rows, null, 2)
    assert.ok(pretty.split('\n').length > 100, 'fixture must be many lines per entry')
    assert.equal(await countRecordsInBlob(blob(pretty), 'd.json'), 25)
  })
})
