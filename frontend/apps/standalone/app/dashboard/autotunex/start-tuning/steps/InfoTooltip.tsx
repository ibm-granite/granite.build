'use client'

import { Tooltip } from '@carbon/react'
import { Information } from '@carbon/icons-react'
import styles from './InfoTooltip.module.scss'

/**
 * The little (i) beside a field label.
 *
 * Moved out of Step1DatasetUpload so the HuggingFace form can render the same
 * affordance on its own mapping rows: the column descriptions come from the
 * backend's dataset-types response and are the only thing that explains what an
 * optional target like `documents` is for.
 */
export function InfoTooltip({ label }: { label: string }) {
  return (
    // Both settings exist to keep the bubble inside its tile. Carbon centres the
    // tooltip on its trigger and caps it at 288px, so a trigger ~115px into this
    // 380px settings column put 144px of bubble past the tile's left edge and over
    // the side nav. `bottom-start` anchors the bubble's left edge to the trigger so
    // it grows rightward instead, and the narrower cap keeps that growth inside the
    // column. Not `autoAlign`: it is flagged experimental, and it measures against
    // the viewport, where the bubble was never actually clipped.
    <Tooltip label={label} align="bottom-start" className={styles.tooltip}>
      <button type="button" className={styles.tooltipTrigger} aria-label={label}>
        <Information size={16} />
      </button>
    </Tooltip>
  )
}
