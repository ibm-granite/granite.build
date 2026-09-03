'use client'

import { InlineLoading, ProgressBar } from '@carbon/react'
import type { LogEntry } from '../types'
import styles from './TuningLogViewer.module.scss'

interface Props {
  logs: LogEntry[]
  isLoading: boolean
  isLoadingMore: boolean
  onScroll: (e: React.UIEvent<HTMLDivElement>) => void
  /** Optional cap for the scroll container height (px). Overrides the SCSS 70vh default. */
  maxHeight?: number
}

export function LogLines({ logs, isLoading, isLoadingMore, onScroll, maxHeight }: Props) {
  if (isLoading) {
    return <ProgressBar size="small" label="Loading" helperText="Loading logs..." />
  }

  const style = maxHeight ? { maxHeight } : undefined

  if (logs.length === 0) {
    return <div className={styles.logViewer} style={style}><div className={styles.logLine}>No logs available</div></div>
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
    </div>
  )
}
