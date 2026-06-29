'use client'

import { LineChart } from '@carbon/charts-react'
import { ScaleTypes } from '@carbon/charts'
import type { BuildStatusChartPoint } from '@/types'
import { useChartsTheme } from '@/hooks/useTheme'

interface Props {
  data: BuildStatusChartPoint[]
  showTestRuns?: boolean
}

const STATUSES = ['running', 'success', 'failed', 'pending', 'submitted', 'suspended'] as const

export function BuildStatusChart({ data, showTestRuns = true }: Props) {
  const theme = useChartsTheme()

  if (!data || data.length === 0) {
    return <p style={{ color: '#525252', padding: '1rem' }}>No data available for this period.</p>
  }

  const chartData = data.flatMap((point) =>
    STATUSES.flatMap((status) => {
      const label = status.charAt(0).toUpperCase() + status.slice(1)
      const rows: { group: string; date: string; value: number }[] = [
        { group: label, date: point.date, value: point[status] ?? 0 },
      ]
      if (showTestRuns) {
        rows.push({
          group: `${label} (Test)`,
          date: point.date,
          value: point[`${status}_test`] ?? 0,
        })
      }
      return rows
    }),
  )

  const options = {
    title: '',
    axes: {
      bottom: { title: 'Date', mapsTo: 'date', scaleType: ScaleTypes.TIME },
      left:   { title: 'Builds', mapsTo: 'value', scaleType: ScaleTypes.LINEAR },
    },
    curve: 'curveMonotoneX',
    height: '350px',
    theme,
  }

  return <LineChart data={chartData} options={options} />
}
