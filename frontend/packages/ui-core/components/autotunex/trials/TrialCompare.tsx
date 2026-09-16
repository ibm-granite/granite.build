'use client'

import { useId, useState } from 'react'
import type { CSSProperties } from 'react'
import {
  Button,
  IconButton,
  StructuredListWrapper,
  StructuredListHead,
  StructuredListBody,
  StructuredListRow,
  StructuredListCell,
  Tag,
} from '@carbon/react'
import { ChevronDown, ChevronRight, Close } from '@carbon/icons-react'
import type { Trial } from '../../../types'
import { getOddOnesOut, groupCompareKeys, labelForCompareKey } from './trialCompareGrouping'
import { primaryMetric } from './trialsRadar'

// ── Reference-parity helpers (ported from AutoTuneX Compare.svelte / Utils) ────

// training_config keys stripped before flattening (paths / run-specific noise).
const STRIP_TRAINING_KEYS = [
  'output_dir',
  'train_file',
  'test_file',
  'validation_file',
  'resource_name',
]

function formatTime(seconds: number): string {
  if (seconds <= 0) return '0 s'
  const total = Math.floor(seconds)
  const mins = Math.floor(total / 60)
  const secs = total % 60
  return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`
}

function flattenObject(obj: Record<string, any>, parentKey = '', sep = '.'): Record<string, any> {
  const flat: Record<string, any> = {}
  for (const key in obj) {
    if (!Object.prototype.hasOwnProperty.call(obj, key)) continue
    const newKey = parentKey ? parentKey + sep + key : key
    const value = obj[key]
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      Object.assign(flat, flattenObject(value, newKey, sep))
    } else {
      flat[newKey] = value
    }
  }
  return flat
}

// Build one comparison row per trial: id + flattened (stripped) config + rounded metrics.
function toCompareRow(trial: Trial): Record<string, any> {
  const rawConfig = (trial.config ?? {}) as Record<string, any>
  const config = structuredClone(rawConfig)
  if (config.training_config && typeof config.training_config === 'object') {
    for (const k of STRIP_TRAINING_KEYS) delete config.training_config[k]
  }
  const flatConfig = flattenObject(config)

  const metrics: Record<string, any> = { ...(trial.metrics ?? {}) }
  if (typeof metrics.loss === 'number') metrics.loss = +metrics.loss.toFixed(5)
  if (typeof metrics.train_loss === 'number') metrics.train_loss = +metrics.train_loss.toFixed(5)
  if (typeof metrics.total_time === 'number') metrics.total_time = formatTime(+metrics.total_time)

  return { id: trial.id, ...flatConfig, ...metrics }
}

// ── Cell rendering ─────────────────────────────────────────────────────────────

function CompareValue({
  value,
  isOdd,
}: {
  value: unknown
  isOdd: boolean
}) {
  if (Array.isArray(value)) {
    return (
      <>
        {value.map((el, i) => {
          if (typeof el === 'string') return <Tag key={i}>{el}</Tag>
          if (el && typeof el === 'object' && 'experiment_name' in el) return <Tag key={i}>{el.experiment_name}</Tag>
          if (el && typeof el === 'object' && 'name' in el) return <Tag key={i}>{el.name}</Tag>
          return <Tag key={i}>No name attribute</Tag>
        })}
      </>
    )
  }
  const display = value === null || value === undefined ? '' : String(value)
  return isOdd ? <strong>{display}</strong> : <>{display}</>
}

// The "loss" a trial is judged on — its primary metric, falling back to a literal
// `loss`. Delegates to the shared accessor so this really does match the Loss column
// in the trials table, which it previously only claimed to.
function lossOf(trial: Trial): number | null {
  return primaryMetric(trial)?.value ?? null
}

// ── Section headings ───────────────────────────────────────────────────────────

const HEADING_TITLE: CSSProperties = {
  fontSize: '0.75rem',
  fontWeight: 600,
  letterSpacing: '0.32px',
  textTransform: 'uppercase',
  color: 'var(--cds-text-primary)',
  whiteSpace: 'nowrap',
}
const HEADING_SUBTITLE: CSSProperties = {
  fontSize: '0.75rem',
  fontWeight: 400,
  letterSpacing: '0.32px',
  color: 'var(--cds-text-secondary)',
  textTransform: 'none',
}

// Full-width band, so heading text lays out across the whole row instead of
// contracting. layer-02 is #f4f4f4 in the app's light theme (g10) and #393939 in
// g100 — a literal #f4f4f4 would leave near-invisible text once the theme is
// switched, since text-primary is #f4f4f4 there too.
const HEADING_BAND: CSSProperties = {
  background: 'var(--cds-layer-02)',
  width: '100%',
  marginTop: '1rem',
}
const HEADING_BAND_PADDING = '0.75rem 1rem'

// Carbon's structured list already rules each section off with a subtle border on
// its first and last row, so headings add spacing rather than another border.
function SectionHeading({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <div
      style={{
        ...HEADING_BAND,
        display: 'flex',
        alignItems: 'baseline',
        gap: '0.75rem',
        padding: HEADING_BAND_PADDING,
      }}
    >
      <span style={HEADING_TITLE}>{title}</span>
      <span style={HEADING_SUBTITLE}>{subtitle}</span>
    </div>
  )
}

function plural(count: number, noun: string): string {
  return `${count} ${noun}${count === 1 ? '' : 's'}`
}

// "1 hyperparameter is" / "3 hyperparameters are" — count, noun and copula all
// agreeing. The section subtitles are present tense, so the verb has to be
// inflected by the same count `plural` inflects the noun by. It lives in one
// helper because getting it wrong stays invisible until a section holds exactly
// one key, and Differences holds one as soon as two trials differ in a single
// hyperparameter — routine for a narrow search space.
function pluralIs(count: number, noun: string): string {
  return `${plural(count, noun)} ${count === 1 ? 'is' : 'are'}`
}

interface Props {
  trials: Trial[]
  /** Drop one trial from the comparison. */
  onRemove: (trialId: string) => void
  /**
   * The job's winning trial, or undefined when no trial reported a usable metric.
   *
   * It is decided over every trial in the job, which is why it arrives as a prop
   * instead of being read off `trials`: this component only ever sees the
   * compared subset, and the lower loss of two arbitrary trials is not the
   * winner. So a comparison that leaves the winner out carries no tag at all
   * rather than crowning whichever of the two happened to score better. It is
   * also the same id that colours the winning run in the charts, so the tag and
   * that colour cannot disagree.
   */
  bestTrialId?: string
}

export function TrialCompare({ trials, onRemove, bestTrialId }: Props) {
  // Shared hyperparameters are collapsed by default so the parameters that
  // actually varied occupy the visible page.
  const [sharedExpanded, setSharedExpanded] = useState(false)
  const sharedListId = useId()

  // Sort by loss ascending — lowest loss first; trials without a loss sink to the end.
  const sortedTrials = [...trials].sort((a, b) => {
    const la = lossOf(a)
    const lb = lossOf(b)
    if (la === null && lb === null) return 0
    if (la === null) return 1
    if (lb === null) return -1
    return la - lb
  })

  const rows = sortedTrials.map(toCompareRow)
  if (rows.length === 0) return null

  // The trials' own primary-metric names, which are dynamic — without these the
  // Results section only recognises the fixed RESULT_KEYS list.
  const metricKeys = [...new Set(sortedTrials.map((t) => t.metric).filter(Boolean))] as string[]
  const { resultKeys, differingKeys, sameKeys } = groupCompareKeys(rows, metricKeys)
  const oddOnes = getOddOnesOut(rows, differingKeys)
  // The bold "differs from most trials" convention only appears when some key has
  // a minority value — never in a two-trial comparison, where every differing key
  // splits evenly. Only explain it when it is actually on screen.
  const showLegend = Object.keys(oddOnes).length > 0

  // A comparison needs two columns to be a comparison, so removal stops at two.
  //
  // The disabled buttons say nothing about why, by request. If that is ever
  // revisited, do not reach for a tooltip on them: Carbon sets
  // `pointer-events: none` on a disabled icon-only button (_button.scss, the
  // #13815 workaround) and binds the tooltip's mouseenter to the button element
  // itself, so a disabled IconButton's tooltip never opens — nor can a disabled
  // button be focused into, so a keyboard reader would get nothing either. Any
  // explanation has to be always-rendered text somewhere else, or the buttons
  // have to be hidden at the floor rather than disabled.
  const atRemovalFloor = rows.length <= 2

  // Fixed label column + data columns split evenly across the available width.
  // Carbon's default auto table layout leaves large gaps when cell content is
  // short (numbers, short strings), so we pin the layout explicitly instead.
  // Every list below repeats the same widths — they are separate tables, so each
  // one needs them to stay column-aligned with its neighbours.
  const labelColWidth = '13rem'
  const dataColMinWidth = '10rem'
  const dataColWidth = `calc((100% - ${labelColWidth}) / ${rows.length})`
  const listStyle: CSSProperties = { tableLayout: 'fixed', width: '100%' }
  const labelCellStyle: CSSProperties = { width: labelColWidth, maxWidth: 'none', paddingRight: '2rem' }
  const dataCellStyle: CSSProperties = {
    width: dataColWidth,
    minWidth: dataColMinWidth,
    maxWidth: 'none',
  }

  const renderList = (keys: string[], ariaLabel: string, id?: string) => (
    <StructuredListWrapper isCondensed isFlush id={id} aria-label={ariaLabel} style={listStyle}>
      <StructuredListBody>
        {keys.map((key) => (
          <StructuredListRow key={key}>
            <StructuredListCell style={labelCellStyle}>
              <strong>{labelForCompareKey(key)}</strong>
            </StructuredListCell>
            {rows.map((row) => (
              <StructuredListCell key={row.id} style={dataCellStyle}>
                <CompareValue value={row[key]} isOdd={!!oddOnes[key]?.has(String(row[key]))} />
              </StructuredListCell>
            ))}
          </StructuredListRow>
        ))}
      </StructuredListBody>
    </StructuredListWrapper>
  )

  return (
    <div>
      {/* Column headers live in their own list so the section headings below can
          span the full width — Carbon's structured list is a CSS table, where
          colSpan has no effect on its div cells. */}
      <StructuredListWrapper isCondensed isFlush aria-label="Compared trials" style={listStyle}>
        <StructuredListHead>
          <StructuredListRow head>
            <StructuredListCell head style={labelCellStyle} />
            {rows.map((row) => (
              <StructuredListCell key={row.id} head style={dataCellStyle}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                  <span>{row.id}</span>
                  <IconButton
                    kind="ghost"
                    size="sm"
                    label={`Remove ${row.id} from comparison`}
                    disabled={atRemovalFloor}
                    onClick={() => onRemove(String(row.id))}
                  >
                    <Close />
                  </IconButton>
                </div>
                {/* Own block below the id, so a long tag never squeezes the id's
                    column or wraps beside it. Carbon's .cds--tag carries its own
                    margin; the wrapper owns the spacing instead. */}
                {row.id === bestTrialId && (
                  <div style={{ marginTop: '0.25rem' }}>
                    <Tag type="gray" size="sm" style={{ margin: 0 }}>
                      Winning trial
                    </Tag>
                  </div>
                )}
              </StructuredListCell>
            ))}
          </StructuredListRow>
        </StructuredListHead>
      </StructuredListWrapper>

      {resultKeys.length > 0 && (
        <>
          <SectionHeading title="Results" subtitle="Trial performance metrics" />
          {renderList(resultKeys, 'Trial results')}
        </>
      )}

      {differingKeys.length > 0 && (
        <>
          <SectionHeading
            title="Differences"
            subtitle={`${pluralIs(differingKeys.length, 'hyperparameter')} varied across the selected trials`}
          />
          {renderList(differingKeys, 'Differing hyperparameters')}
        </>
      )}

      {sameKeys.length > 0 && (
        <>
          {/* The band sits on the wrapper and the button fills it, so Carbon's
              translucent ghost-hover still reads over the grey. Carbon caps
              .cds--btn at max-inline-size: 20rem, which is what made this
              heading's text wrap — both cap properties are cleared below. */}
          <div style={HEADING_BAND}>
            <Button
              kind="ghost"
              size="sm"
              renderIcon={sharedExpanded ? ChevronDown : ChevronRight}
              onClick={() => setSharedExpanded((expanded) => !expanded)}
              aria-expanded={sharedExpanded}
              // The list is unmounted while collapsed, so only point at it when it exists.
              aria-controls={sharedExpanded ? sharedListId : undefined}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                width: '100%',
                maxWidth: 'none',
                maxInlineSize: 'none',
                padding: HEADING_BAND_PADDING,
                textAlign: 'left',
              }}
            >
              <span style={{ display: 'flex', alignItems: 'baseline', gap: '0.75rem' }}>
                <span style={HEADING_TITLE}>Similarities</span>
                <span style={HEADING_SUBTITLE}>
                  {`${pluralIs(sameKeys.length, 'hyperparameter')} identical across the selected trials`}
                </span>
              </span>
            </Button>
          </div>
          {sharedExpanded && renderList(sameKeys, 'Shared hyperparameters', sharedListId)}
        </>
      )}

      {showLegend && (
        <p style={{ ...HEADING_SUBTITLE, padding: '1rem 0 0' }}>
          Bold = value that differs from most trials.
        </p>
      )}
    </div>
  )
}
