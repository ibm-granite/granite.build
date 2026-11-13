// Copyright Granite.Build Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

'use client'

import { LineChart } from '@carbon/charts-react'
import { ScaleTypes } from '@carbon/charts'
import type { BuildStatusChartPoint } from '../types'
import { useChartsTheme } from '../hooks/useTheme'

interface Props {
  data: BuildStatusChartPoint[]
  showTestRuns?: boolean
}

// Mirrors gbserver's Status enum (src/gbserver/types/status.py).
const STATUS_LABELS = {
  running: 'Running',
  success: 'Succeeded',
  failed: 'Failed',
  invalid: 'Invalid',
  pending: 'Pending',
  submitted: 'Submitted',
  retry_pending: 'Retrying',
  cancel_requested: 'Cancelling',
  cancelled: 'Cancelled',
} as const

const STATUSES = Object.keys(STATUS_LABELS) as (keyof typeof STATUS_LABELS)[]

export function BuildStatusChart({ data, showTestRuns = true }: Props) {
  const theme = useChartsTheme()

  if (!data || data.length === 0) {
    return <p style={{ color: '#525252', padding: '1rem' }}>No data available for this period.</p>
  }

  const chartData = data.flatMap((point) =>
    STATUSES.flatMap((status) => {
      const label = STATUS_LABELS[status]
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
