'use client'

import * as React from 'react'
import { useRouter } from 'next/navigation'
import { Link, Modal } from '@carbon/react'
import { Launch } from '@carbon/icons-react'
import { useRoutes } from '../../config/routes'
import styles from './OpenArtifactModal.module.scss'

// Where a lineage click can take an artifact: its registry page, its
// HuggingFace page, or both. Nothing opens without this dialog, so a mis-click
// on a node never navigates.
export interface OpenArtifactTarget {
  name: string
  uri?: string
  // Registry UUID; absent when the artifact is not registered, or is the page's own.
  uuid?: string
  artifactType?: string
  hfUrl?: string | null
}

// Whether there is anywhere to offer: a dialog with no destination is not shown.
export function canOpenArtifact(target: OpenArtifactTarget): boolean {
  return Boolean(target.uuid || target.hfUrl)
}

export default function OpenArtifactModal({
  target,
  onClose,
}: {
  target: OpenArtifactTarget | null
  onClose: () => void
}) {
  const router = useRouter()
  const routes = useRoutes()
  const uuid = target?.uuid
  const hfUrl = target?.hfUrl
  const openHf = () => {
    if (hfUrl) window.open(hfUrl, '_blank', 'noopener,noreferrer')
    onClose()
  }
  // Carbon's footer: primary is the registry page when there is one; with
  // HuggingFace too, both secondaries (Cancel, then HuggingFace) sit beside it.
  // HuggingFace alone takes the primary, and Cancel the secondary.
  const both = Boolean(uuid && hfUrl)
  return (
    <Modal
      open={target !== null}
      onRequestClose={onClose}
      modalHeading="Open artifact"
      primaryButtonText={uuid ? 'View artifact page' : 'Open on HuggingFace'}
      onRequestSubmit={() => {
        if (uuid) {
          router.push(routes.artifactHref(uuid))
          onClose()
        } else openHf()
      }}
      {...(both
        ? { secondaryButtons: [{ buttonText: 'Cancel', onClick: onClose }, { buttonText: 'Open on HuggingFace', onClick: openHf }] }
        : { secondaryButtonText: 'Cancel', onSecondarySubmit: onClose })}
      // Three buttons do not fit a small footer.
      size={both ? 'md' : 'sm'}
    >
      {target && (
        // What the dialog is about to open, so a mis-click is recognisable.
        <dl className={styles.summary}>
          <dt>Name</dt>
          <dd>{target.name}</dd>
          {uuid && (
            <>
              <dt>ID</dt>
              <dd><code>{uuid}</code></dd>
            </>
          )}
          {target.artifactType && (
            <>
              <dt>Type</dt>
              <dd>{target.artifactType.toLowerCase()}</dd>
            </>
          )}
          {target.uri && (
            <>
              <dt>URI</dt>
              <dd>
                {hfUrl ? (
                  <Link href={hfUrl} target="_blank" rel="noopener noreferrer" renderIcon={Launch}>{target.uri}</Link>
                ) : (
                  <code>{target.uri}</code>
                )}
              </dd>
            </>
          )}
          {!uuid && (
            <>
              <dt>Registry</dt>
              <dd>Not in the artifact registry</dd>
            </>
          )}
        </dl>
      )}
    </Modal>
  )
}
