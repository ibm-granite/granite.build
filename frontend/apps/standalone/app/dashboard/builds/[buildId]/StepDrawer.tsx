'use client'

import * as React from 'react'
import { IconButton } from '@carbon/react'
import { Close } from '@carbon/icons-react'
import type { Build, BuildTargetRun } from '@granite-build/ui-core/types'
import { BuildStatusBadge } from '@granite-build/ui-core/components/BuildStatusBadge'
import StepDetailsPanel, { stepDrawerSummary } from './StepDetailsPanel'
import styles from './LineagePanel.module.scss'

interface Props {
  targetName: string
  target: BuildTargetRun | undefined
  build: Build | undefined
  onClose: () => void
  drawerRef?: React.Ref<HTMLDivElement>
  closeButtonRef?: React.Ref<HTMLButtonElement>
}

// A target's step details, shared by the build and artifact lineage panels. A
// drawer, not a modal: no overlay, so the graph behind stays visible and
// clickable and picking another target just re-points the drawer.
export default function StepDrawer({ targetName, target, build, onClose, drawerRef, closeButtonRef }: Props) {
  const { status, subtitle, summary } = stepDrawerSummary(target, build)
  return (
    <div
      ref={drawerRef}
      className={styles.stepSidePanel}
      role="dialog"
      aria-label={`Step details — ${targetName}`}
    >
      <div className={styles.stepSidePanelHeader}>
        <div className={styles.stepSidePanelIdentity}>
          <h4 className={styles.stepSidePanelHeading}>{targetName}</h4>
          <div className={styles.stepSidePanelSubtitle}>{subtitle}</div>
          {status && (
            <div className={styles.stepSidePanelStatus}>
              <BuildStatusBadge status={status} />
            </div>
          )}
          {summary && (
            <div className={styles.stepSidePanelSummary}>{summary}</div>
          )}
        </div>
        <IconButton ref={closeButtonRef} kind="ghost" label="Close" align="bottom" onClick={onClose}>
          <Close />
        </IconButton>
      </div>
      <div className={styles.stepSidePanelBody}>
        <StepDetailsPanel
          targetName={targetName}
          target={target}
          sourceUri={build?.source_uri}
          buildId={build?.uuid}
        />
      </div>
    </div>
  )
}
