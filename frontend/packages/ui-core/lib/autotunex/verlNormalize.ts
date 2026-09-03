// verl RL rows carry three *structured* fields:
//   - `prompt`       : a chat-message list  [{ role, content }, ...]
//   - `reward_model` : an object            { style, ground_truth }
//   - `extra_info`   : an object            { split, index, ... }
//
// Datasets exported to JSONL/CSV very often store these as `json.dumps`'d
// STRINGS rather than native array/object values (Python's json.dumps is the
// tell — it emits `", "` / `": "` separators). The Reward Function step's
// test-case pre-fill is verl-strict: it reads `Array.isArray(row.prompt)` and
// `row.reward_model?.ground_truth`. Against string-encoded fields both checks
// fail — `Array.isArray("[...]")` is false and `"{...}".ground_truth` is
// undefined — so every test case comes out blank even though the dataset is
// genuinely verl-shaped.
//
// This module is the preprocessing pass that coerces those fields back to their
// declared verl types *before* the rows reach the (unchanged) step. It is:
//   - idempotent — native values pass through untouched;
//   - safe       — a value is only parsed when it is a string that *looks* like
//                  JSON (`{`/`[`), and an unparseable string is left as-is;
//   - a no-op for non-RL datasets — SFT/DPO/KTO rows don't carry these fields
//     (and their plain-text `prompt` doesn't start with `{`/`[`).

type Row = Record<string, any>

// Only these verl fields are structured; `data_source`/`ability` stay strings.
const VERL_JSON_FIELDS = ['prompt', 'reward_model', 'extra_info'] as const

function parseIfJsonString(value: unknown): unknown {
  if (typeof value !== 'string') return value
  const trimmed = value.trim()
  // Restrict to object/array literals so a plain-string prompt (SFT/DPO) or a
  // bare scalar is never reinterpreted.
  if (trimmed[0] !== '{' && trimmed[0] !== '[') return value
  try {
    return JSON.parse(trimmed)
  } catch {
    return value
  }
}

/** Parse any json.dumps'd verl fields in a single row. Returns the same
 *  reference when nothing changed so callers can cheaply detect no-ops. */
export function normalizeVerlRow(row: Row): Row {
  if (!row || typeof row !== 'object' || Array.isArray(row)) return row
  let next: Row | null = null
  for (const field of VERL_JSON_FIELDS) {
    if (!(field in row)) continue
    const parsed = parseIfJsonString(row[field])
    if (parsed !== row[field]) {
      if (!next) next = { ...row }
      next[field] = parsed
    }
  }
  return next ?? row
}

/** Row-wise {@link normalizeVerlRow} over a dataset preview. Returns the same
 *  array reference when no row changed. */
export function normalizeVerlRows(rows: Row[] | undefined | null): Row[] {
  if (!Array.isArray(rows) || rows.length === 0) return rows ?? []
  let changed = false
  const next = rows.map((r) => {
    const nr = normalizeVerlRow(r)
    if (nr !== r) changed = true
    return nr
  })
  return changed ? next : rows
}
