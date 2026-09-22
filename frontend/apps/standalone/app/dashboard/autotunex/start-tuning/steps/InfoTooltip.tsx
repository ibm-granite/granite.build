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
    <Tooltip label={label}>
      <button type="button" className={styles.tooltipTrigger} aria-label={label}>
        <Information size={16} />
      </button>
    </Tooltip>
  )
}
