'use client'

import { useMemo, useState } from 'react'
import { ContentSwitcher, Switch } from '@carbon/react'
import { useChartsTheme } from '../../../hooks/useTheme'
import { buildParallelCoords } from './trialsParallelCoords'
import type { ParallelMetric } from './trialsParallelCoords'
import { hyperparamColumnLabel } from './trialHyperparams'
import styles from './TrialSearchSpace.module.scss'
import type { Trial } from '../../../types'

// Named for its heading rather than for the chart type, which also keeps it clear
// of `trialsParallelCoords.ts` beside it: on a case-insensitive filesystem a
// PascalCase component and a camelCase module of the same word resolve to the same
// path, and `./TrialsParallelCoords` silently imported the pure module. The same
// reason trialMetrics.ts sits beside TrialMetricsCharts.tsx rather than a case-twin.

// Carbon blue, worst -> best. Two steps interpolated rather than a hand-picked
// ramp, which keeps lightness monotonic by construction; both ends were checked
// against their own surface at >= 3:1.
//
// Dark mode is its own pair, not a flip of the light one: on g100 the "best" end
// has to be the BRIGHT one, because a near-black navy is what disappears against a
// dark surface — the reverse of which end vanishes on white.
const METRIC_RAMP: Record<'white' | 'g100', [string, string]> = {
  white: ['#4589ff', '#001141'],
  g100: ['#4589ff', '#d0e2ff'],
}

// viewBox units. The SVG scales to its container, so these are a fixed drawing
// grid rather than pixels — `plot`'s min-width in the stylesheet is what actually
// keeps the axis names from colliding.
//
// The side margins are deliberately tiny. Nothing is drawn outside the outermost
// axes: their labels are anchored `start` and `end` so they read INWARD, and
// `nudgeFor` shifts them only 4 units out, so 10 units clears the widest thing that
// can overhang (a 4-unit nudge, a 2.5-unit vertex circle, a 1-unit half-stroke).
// Larger margins were pure dead space at the left and right of the row — the same
// waste that made the radar chart this replaced leave most of the row empty.
const VIEW_W = 1040
const VIEW_H = 392
const PLOT_LEFT = 10
const PLOT_RIGHT = 1030
const PLOT_TOP = 98
const PLOT_BOTTOM = 356

function mix(from: string, to: string, t: number): string {
  const parse = (hex: string) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16))
  const a = parse(from)
  const b = parse(to)
  return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(',')})`
}

interface Props {
  /** Trials to draw — the reader's ticked selection. */
  trials: Trial[]
  /** Trials that set the axis scales, normally every plottable trial in the job. */
  boundsFrom: Trial[]
  /** From `hyperparamColumns`, so the plot and the table's columns cannot disagree. */
  hyperparamKeys: string[]
  metric: ParallelMetric
  /** The app's run -> colour map, for the "Trial" colour mode. */
  colorScale: Record<string, string>
  bestTrialId?: string
}

/**
 * The trials parallel-coordinates plot: one vertical axis per searched
 * hyperparameter, then total time and the metric, with one polyline per trial.
 *
 * This replaced the radar chart. A radar is this same construction in polar
 * coordinates, and unrolling it drops the two things that made it hard to read: the
 * square aspect ratio, which left most of a full-width row empty, and the enclosed
 * area, which reads as meaning when the axes carry different units. It also holds
 * more than a handful of axes and can carry a categorical one such as
 * `lr_scheduler_type`, which a radar cannot plot at all.
 *
 * Colour defaults to the run, so a line here is the colour of that trial's row
 * checkbox and of its curves in the charts below — one trial, one colour, everywhere
 * on the page. Two limits of that scale surface here, both inherited rather than
 * introduced: `METRIC_PALETTE`'s slot 8 (#520408) is too dark to sit in a
 * categorical set, and above `EMPHASIS_THRESHOLD` runs `trialColorScale` returns its
 * emphasis form, which paints every run but the best in one de-emphasis grey — so in
 * a job of more than ten trials several selected lines can share a colour. The metric
 * mode is the way out of both: one hue light-to-dark, legible at any count.
 */
export function TrialSearchSpace({
  trials,
  boundsFrom,
  hyperparamKeys,
  metric,
  colorScale,
  bestTrialId,
}: Props) {
  const theme = useChartsTheme()
  const [colorByMetric, setColorByMetric] = useState(false)
  const [hiddenIds, setHiddenIds] = useState<Record<string, boolean>>({})
  const [hoveredId, setHoveredId] = useState<string | null>(null)

  const { axes, lines } = useMemo(
    () => buildParallelCoords(trials, hyperparamKeys, metric, boundsFrom),
    [trials, hyperparamKeys, metric, boundsFrom]
  )

  // Two axes is the floor: a single-axis parallel-coordinates plot is a dot plot
  // with extra chrome, and with none there is nothing to draw.
  if (axes.length < 2) return null

  const step = (PLOT_RIGHT - PLOT_LEFT) / (axes.length - 1)
  const axisX = (index: number) => PLOT_LEFT + index * step
  const axisY = (position: number) => PLOT_BOTTOM - position * (PLOT_BOTTOM - PLOT_TOP)
  const firstOutcome = axes.findIndex((a) => a.isOutcome)

  const colorFor = (line: (typeof lines)[number]) => {
    if (!colorByMetric) return colorScale[line.id] ?? 'var(--cds-text-primary)'
    const [worst, best] = METRIC_RAMP[theme]
    return mix(worst, best, line.goodness ?? 0)
  }

  // A null position breaks the path rather than joining across it: the trial did
  // not report that key, and a straight line through the gap would assert a value.
  const pathFor = (positions: (number | null)[]) => {
    let d = ''
    let penDown = false
    positions.forEach((position, index) => {
      if (position === null) {
        penDown = false
        return
      }
      d += `${penDown ? 'L' : 'M'}${axisX(index)},${axisY(position)} `
      penDown = true
    })
    return d.trim()
  }

  const visibleLines = lines.filter((l) => !hiddenIds[l.id])
  const hovered = hoveredId && !hiddenIds[hoveredId] ? lines.find((l) => l.id === hoveredId) : undefined

  const anchorFor = (index: number) =>
    index === 0 ? 'start' : index === axes.length - 1 ? 'end' : 'middle'
  // Nudge only the outermost labels, so they sit inside the viewBox rather than
  // hanging off the drawing's edge.
  const nudgeFor = (index: number) => (index === 0 ? -4 : index === axes.length - 1 ? 4 : 0)

  return (
    <div>
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          alignItems: 'baseline',
          justifyContent: 'space-between',
          gap: '0.75rem',
          marginBottom: '0.5rem',
        }}
      >
        <div>
          <h5 style={{ margin: 0 }}>Trial search space</h5>
          <p style={{ margin: '0.125rem 0 0', fontSize: '0.75rem', color: 'var(--cds-text-secondary)' }}>
            Both outcome axes put the better value at the top, so a line that stays high did well.
          </p>
        </div>
        {/* Carbon's own segmented control, so this reads as part of the page rather
            than as two hand-styled buttons. */}
        <ContentSwitcher
          size="sm"
          selectedIndex={colorByMetric ? 1 : 0}
          onChange={({ index }) => setColorByMetric(index === 1)}
          style={{ maxWidth: '16rem' }}
        >
          <Switch name="trial" text="Trial" />
          <Switch name="metric" text={hyperparamColumnLabel(metric.name)} />
        </ContentSwitcher>
      </div>

      <div className={styles.scroller}>
        <svg
          className={styles.plot}
          viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
          role="img"
          aria-label={`Parallel coordinates plot of ${lines.length} trials across ${axes.length} axes.`}
        >
          <text className={styles.groupLabel} x={PLOT_LEFT} y={26}>
            SEARCHED
          </text>
          {firstOutcome !== -1 && (
            <>
              <text className={styles.groupLabel} x={PLOT_RIGHT} y={26} textAnchor="end">
                OUTCOME ↑ BETTER
              </text>
              <line
                className={styles.divider}
                x1={axisX(firstOutcome) - step / 2}
                y1={16}
                x2={axisX(firstOutcome) - step / 2}
                y2={PLOT_BOTTOM + 24}
              />
            </>
          )}

          {axes.map((axis, index) => (
            <g key={axis.key}>
              <line
                className={styles.axisLine}
                x1={axisX(index)}
                y1={PLOT_TOP}
                x2={axisX(index)}
                y2={PLOT_BOTTOM}
              />
              <text
                className={styles.axisName}
                x={axisX(index) + nudgeFor(index)}
                y={54}
                textAnchor={anchorFor(index)}
              >
                {hyperparamColumnLabel(axis.key)}
              </text>
              <text
                className={styles.axisTick}
                x={axisX(index) + nudgeFor(index)}
                y={PLOT_TOP - 11}
                textAnchor={anchorFor(index)}
              >
                {axis.topLabel}
              </text>
              <text
                className={styles.axisTick}
                x={axisX(index) + nudgeFor(index)}
                y={PLOT_BOTTOM + 18}
                textAnchor={anchorFor(index)}
              >
                {axis.bottomLabel}
              </text>
            </g>
          ))}

          {visibleLines.map((line) => {
            const faded = hovered !== undefined && hovered.id !== line.id
            return (
              <g key={line.id} opacity={faded ? 0.12 : 1}>
                <path
                  className={styles.line}
                  d={pathFor(line.positions)}
                  stroke={colorFor(line)}
                  strokeWidth={hovered?.id === line.id ? 3 : 2}
                  pointerEvents="none"
                />
                {line.positions.map((position, index) =>
                  position === null ? null : (
                    <circle
                      key={axes[index].key}
                      cx={axisX(index)}
                      cy={axisY(position)}
                      r={2.5}
                      fill="var(--cds-layer)"
                      stroke={colorFor(line)}
                      strokeWidth={2}
                      pointerEvents="none"
                    />
                  )
                )}
              </g>
            )
          })}

          {/* Hit targets after the marks, so the topmost line under the pointer wins. */}
          {visibleLines.map((line) => (
            <path
              key={`hit-${line.id}`}
              className={styles.hit}
              d={pathFor(line.positions)}
              onMouseEnter={() => setHoveredId(line.id)}
              onMouseLeave={() => setHoveredId(null)}
            />
          ))}

          {/* Values for the hovered trial only. A number on every vertex of every
              line would be unreadable; the axis ticks and this carry the rest. */}
          {hovered?.positions.map((position, index) =>
            position === null ? null : (
              <text
                key={`value-${axes[index].key}`}
                className={styles.valueLabel}
                x={axisX(index) + nudgeFor(index)}
                y={axisY(position) + (axisY(position) > PLOT_TOP + 26 ? -9 : 17)}
                textAnchor={anchorFor(index)}
                fill={colorFor(hovered)}
              >
                {hovered.values[index]}
              </text>
            )
          )}
        </svg>
      </div>

      {colorByMetric && (
        <div className={styles.ramp}>
          <span>{hyperparamColumnLabel(metric.name)}</span>
          <span>{axes[axes.length - 1].bottomLabel} worst</span>
          <span
            className={styles.rampBar}
            style={{
              background: `linear-gradient(to right, ${METRIC_RAMP[theme][0]}, ${METRIC_RAMP[theme][1]})`,
            }}
          />
          <span>{axes[axes.length - 1].topLabel} best</span>
        </div>
      )}

      <div className={styles.legend}>
        {lines.map((line) => (
          <button
            key={line.id}
            type="button"
            className={styles.chip}
            aria-pressed={!hiddenIds[line.id]}
            onClick={() => setHiddenIds((hidden) => ({ ...hidden, [line.id]: !hidden[line.id] }))}
            onMouseEnter={() => setHoveredId(line.id)}
            onMouseLeave={() => setHoveredId(null)}
            onFocus={() => setHoveredId(line.id)}
            onBlur={() => setHoveredId(null)}
          >
            <span className={styles.swatch} style={{ background: colorFor(line) }} />
            {line.id}
            {line.id === bestTrialId && <span className={styles.chipBest}>best</span>}
          </button>
        ))}
      </div>
    </div>
  )
}
