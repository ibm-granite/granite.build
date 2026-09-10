'use client'

import { InlineLoading, InlineNotification, ProgressBar } from '@carbon/react'
import type { LogEntry } from '../types'
import styles from './TuningLogViewer.module.scss'

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
