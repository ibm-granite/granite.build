'use client'

import * as React from 'react'
import { IconButton } from '@carbon/react'
import { Close } from '@carbon/icons-react'
import styles from './NodeDetailsDrawer.module.scss'

interface Props {
  /** Which node the drawer describes; null closes it. Changing it re-points. */
  openFor: string | null
  onClose: () => void
  /** Focused when the drawer opens, and where focus returns from. */
  returnFocusTo?: React.RefObject<HTMLElement | null>
  title: string
  subtitle?: React.ReactNode
  /** Rendered under the heading, above the scrolling body — status, badges. */
  meta?: React.ReactNode
  children: React.ReactNode
}

/**
 * Side panel for lineage node details.
 *
 * Extracted from the build lineage tab so the artifact tab and the internal
 * gb-ui deployment share one implementation instead of three copies.
 *
 * **A drawer, not a modal, on purpose**: no overlay, so the graph behind stays
 * visible and clickable, and picking another node just re-points this panel. That
 * is why the focus handling below is entry-and-restore rather than a focus trap —
 * trapping would fight the interaction the design is built around.
 */
export function NodeDetailsDrawer({
  openFor,
  onClose,
  returnFocusTo,
  title,
  subtitle,
  meta,
  children,
}: Props) {
  const drawerRef = React.useRef<HTMLDivElement | null>(null)
  const closeButtonRef = React.useRef<HTMLButtonElement | null>(null)
  const internalReturnRef = React.useRef<HTMLElement | null>(null)
  const wasOpenRef = React.useRef(false)

  React.useEffect(() => {
    if (!openFor) return
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      // Non-modal: the graph behind stays interactive, so only swallow Escape
      // when focus is actually inside. Otherwise a user mid-interaction with the
      // graph would have the drawer yanked shut under them.
      if (drawerRef.current?.contains(document.activeElement)) onClose()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [openFor, onClose])

  React.useEffect(() => {
    const wasOpen = wasOpenRef.current
    wasOpenRef.current = Boolean(openFor)

    if (openFor) {
      // Capture only when opening from *closed*. Switching directly from node A
      // to node B must not recapture — B's trigger is not where focus should
      // return to — and must not restore, because nothing has closed yet.
      if (!wasOpen) {
        internalReturnRef.current = document.activeElement as HTMLElement | null
      }
      closeButtonRef.current?.focus()
      return
    }

    if (!wasOpen) return

    // The trigger is usually a graph node inside an SVG that re-renders on the
    // status poll, so by close time it may be detached — and focus() on a
    // detached node is a silent no-op that drops focus to <body>. Restore only
    // while it is still connected, else fall back to the graph container so a
    // keyboard user lands somewhere sensible rather than the top of the document.
    const returnTo = internalReturnRef.current
    if (returnTo?.isConnected) {
      returnTo.focus?.()
    } else {
      returnFocusTo?.current?.focus?.()
    }
    internalReturnRef.current = null
  }, [openFor, returnFocusTo])

  if (!openFor) return null

  return (
    <div
      ref={drawerRef}
      className={styles.drawer}
      role="dialog"
      aria-label={`Details — ${title}`}
    >
      <div className={styles.header}>
        <div className={styles.identity}>
          <h4 className={styles.heading}>{title}</h4>
          {subtitle && <div className={styles.subtitle}>{subtitle}</div>}
          {meta}
        </div>
        <IconButton ref={closeButtonRef} kind="ghost" label="Close" align="bottom" onClick={onClose}>
          <Close />
        </IconButton>
      </div>
      <div className={styles.body}>{children}</div>
    </div>
  )
}
