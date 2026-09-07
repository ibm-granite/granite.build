import { ScaleTypes } from '@carbon/charts'
import type { LineChartOptions } from '@carbon/charts'
import type { ChartsTheme } from './trialMetrics'

export interface MetricAxisSpec {
  xTitle: string
  yTitle: string
  theme: ChartsTheme
  /** Run id → colour. Keyed by run so hiding a series never repaints the rest. */
  colorScale: Record<string, string>
  height: string
  /** Log y — required for learning rate, whose values span several decades. */
  logY?: boolean
  /** Draw a marker per point. For eval series, which have too few points to read as a line. */
  points?: boolean
}

/**
 * Shared Carbon line-chart options for the metric charts.
 *
 * Three settings here are load-bearing rather than cosmetic:
 *
 *  - `includeZero: false` on the left axis — see the comment at the call site.
 *
 *  - `alwaysShowRulerTooltip` gives the crosshair readout listing every run at
 *    the hovered x. Without it Carbon shows a single-point tooltip, so comparing
 *    runs means landing the pointer on each line in turn.
 *  - `scaleType: LOG` on the left axis for learning rate. Schedules span from
 *    ~1e-9 up to ~5e-6, and on a linear axis the entire warmup flattens onto the
 *    floor — the chart would show nothing but the peak.
 *
 * No zoom bar: Carbon's implementation assumes a time x-axis and calls
 * `.getTime()` on each mapped x value, so enabling it against these numeric step
 * and epoch axes throws `l[s].getTime is not a function` from `setZoomBarData`
 * before the chart ever mounts. Zooming would need the x values to be Dates,
 * which they are not.
 *
 * Colours arrive as literal hex because Carbon resolves them at render time and
 * cannot read CSS custom properties; `useChartsTheme()` picks the set, so a
 * theme change re-renders with the right one.
 */
export function metricChartOptions(spec: MetricAxisSpec): LineChartOptions {
  return {
    axes: {
      bottom: {
        title: spec.xTitle,
        mapsTo: 'key',
        scaleType: ScaleTypes.LINEAR,
      },
      left: {
        title: spec.yTitle,
        mapsTo: 'value',
        scaleType: spec.logY ? ScaleTypes.LOG : ScaleTypes.LINEAR,
        // Carbon anchors a linear axis at zero by default, which is wrong for
        // every measure here. Loss sits around 15 and the runs differ by well
        // under one unit, so a zero-based axis squeezes every curve into a flat
        // sliver at the top of the plot and hides the only thing worth seeing.
        includeZero: false,
      },
    },
    color: { scale: spec.colorScale },
    points: { enabled: Boolean(spec.points), radius: 3 },
    // `showTotal: false` because Carbon's ruler tooltip sums the series by
    // default, and the sum of several runs' losses is not a quantity — it just
    // grows with how many runs happen to be visible.
    tooltip: { alwaysShowRulerTooltip: true, showTotal: false },
    theme: spec.theme,
    height: spec.height,
    toolbar: { enabled: false },
  }
}
