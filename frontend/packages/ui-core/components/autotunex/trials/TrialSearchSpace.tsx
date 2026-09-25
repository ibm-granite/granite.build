'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
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

// Drawing units, one per CSS pixel. The width is measured from the container and
// the viewBox matches it, so the SVG never scales: text renders at the size the
// stylesheet gives it, and the plot is exactly VIEW_H tall on any screen. A fixed
// viewBox stretched to the container made both grow with the window — on a wide
// screen the plot ran past 500px tall with axis names larger than the table's text.
//
// MIN_W is where nine axis names stop colliding; narrower, the container scrolls
// rather than the labels overlapping. It is also the width drawn before the first
// measurement.
//
// The side margins are deliberately tiny. Nothing is drawn outside the outermost
// axes: their labels are anchored `start` and `end` so they read INWARD, and
// `nudgeFor` shifts them only 4 units out, so 10 units clears the widest thing that
// can overhang (a 4-unit nudge, a 2.5-unit vertex circle, a 1-unit half-stroke).
// Larger margins were pure dead space at the left and right of the row — the same
// waste that made the radar chart this replaced leave most of the row empty.
const MIN_W = 832
const VIEW_H = 272
const PLOT_LEFT = 10
const SIDE_MARGIN = 10
const GROUP_LABEL_Y = 14
const AXIS_NAME_Y = 36
const PLOT_TOP = 62
const PLOT_BOTTOM = 242

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
 * Colour is the run's, so a line here is the colour of that trial's row checkbox
 * and of its curves in the charts below — one trial, one colour, everywhere on the
 * page. The lines drawn are the ticked ones, which the table keeps in distinct hues
 * even past `EMPHASIS_THRESHOLD` trials, where the palette wraps — see
 * `colorClash`. One limit of that scale surfaces here, inherited rather than
 * introduced: `METRIC_PALETTE`'s slot 8 (#520408) is too dark to sit in a
 * categorical set.
 *
 * There was a second colour mode, a light-to-dark ramp on the metric, behind a
 * Trial / <metric> switcher. It was removed so the plot always matches the rest of
 * the page; `ParallelLine.goodness` still carries the value it was drawn from.
 */
export function TrialSearchSpace({
  trials,
  boundsFrom,
  hyperparamKeys,
  metric,
  colorScale,
  bestTrialId,
}: Props) {
  const scrollerRef = useRef<HTMLDivElement>(null)
  const [viewW, setViewW] = useState(MIN_W)
  const [hiddenIds, setHiddenIds] = useState<Record<string, boolean>>({})
  const [hoveredId, setHoveredId] = useState<string | null>(null)

  const { axes, lines } = useMemo(
    () => buildParallelCoords(trials, hyperparamKeys, metric, boundsFrom),
    [trials, hyperparamKeys, metric, boundsFrom]
  )

  // Before the early return, which would otherwise change the hook count. The
  // effect re-runs when the plot first mounts, since the ref is empty until then.
  const canDraw = axes.length >= 2
  useEffect(() => {
    const el = scrollerRef.current
    if (!el) return
    const observer = new ResizeObserver(([entry]) => {
      setViewW(Math.max(MIN_W, Math.floor(entry.contentRect.width)))
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [canDraw])

  // Two axes is the floor: a single-axis parallel-coordinates plot is a dot plot
  // with extra chrome, and with none there is nothing to draw.
  if (!canDraw) return null

  const PLOT_RIGHT = viewW - SIDE_MARGIN
  const step = (PLOT_RIGHT - PLOT_LEFT) / (axes.length - 1)
  const axisX = (index: number) => PLOT_LEFT + index * step
  const axisY = (position: number) => PLOT_BOTTOM - position * (PLOT_BOTTOM - PLOT_TOP)
  const firstOutcome = axes.findIndex((a) => a.isOutcome)

  const colorFor = (line: (typeof lines)[number]) => colorScale[line.id] ?? 'var(--cds-text-primary)'

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
      <div style={{ marginBottom: '0.5rem' }}>
        <h5 style={{ margin: 0 }}>Trial search space</h5>
        <p style={{ margin: '0.125rem 0 0', fontSize: '0.75rem', color: 'var(--cds-text-secondary)' }}>
          Hyperparameter with differences amongst trials are shown.
        </p>
      </div>

      <div className={styles.scroller} ref={scrollerRef}>
        <svg
          className={styles.plot}
          width={viewW}
          height={VIEW_H}
          viewBox={`0 0 ${viewW} ${VIEW_H}`}
          role="img"
          aria-label={`Parallel coordinates plot of ${lines.length} trials across ${axes.length} axes.`}
        >
          <text className={styles.groupLabel} x={PLOT_LEFT} y={GROUP_LABEL_Y}>
            Hyperparameters
          </text>
          {firstOutcome !== -1 && (
            <>
              <text className={styles.groupLabel} x={PLOT_RIGHT} y={GROUP_LABEL_Y} textAnchor="end">
                Results ↑ Better
              </text>
              <line
                className={styles.divider}
                x1={axisX(firstOutcome) - step / 2}
                y1={4}
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
                y={AXIS_NAME_Y}
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
