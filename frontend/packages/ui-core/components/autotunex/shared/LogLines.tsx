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

import { InlineLoading, InlineNotification, ProgressBar } from '@carbon/react'
import type { LogEntry } from '../../../types'
import styles from './LogLines.module.scss'

interface Props {
  logs: LogEntry[]
  isLoading: boolean
  isLoadingMore: boolean
  /** A fetch failed. Without this an errored request renders as the affirmative
   *  "No logs available", which claims the job produced no output. */
  isError?: boolean
  onScroll: (e: React.UIEvent<HTMLDivElement>) => void
  /** Optional cap for the scroll container height (px). Overrides the SCSS 70vh default. */
  maxHeight?: number
}

export function LogLines({ logs, isLoading, isLoadingMore, isError, onScroll, maxHeight }: Props) {
  if (isLoading) {
    return <ProgressBar size="small" label="Loading" helperText="Loading logs..." />
  }

  const style = maxHeight ? { maxHeight } : undefined

  if (logs.length === 0) {
    return isError ? (
      <InlineNotification
        kind="error"
        title="Couldn't load logs"
        subtitle="The request failed. You may not have access to this job, or the server is unreachable."
        lowContrast
        hideCloseButton
      />
    ) : (
      <div className={styles.logViewer} style={style}><div className={styles.logLine}>No logs available</div></div>
    )
  }

  return (
    <div className={styles.logViewer} style={style} onScroll={onScroll}>
      {logs.map((log) => (
        <div className={styles.logLine} key={log.id}>
          {new Date(log.timestamp).toLocaleString()} {log.level} -- {log.filename} -- {log.message}
        </div>
      ))}
      {isLoadingMore && (
        <div className={styles.logLine}>
          <InlineLoading description="Loading more…" />
        </div>
      )}
      {isError && !isLoadingMore && (
        <div className={styles.logLine}>-- couldn't load older lines --</div>
      )}
    </div>
  )
}
