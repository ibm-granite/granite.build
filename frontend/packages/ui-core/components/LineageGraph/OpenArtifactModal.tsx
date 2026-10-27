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
  // A URI the artifact page resolves itself, when the registry was not asked
  // here; that page says so if nothing is registered under it.
  pageUri?: string
  artifactType?: string
  hfUrl?: string | null
  // Any other web page the URI names, for an http(s) URI not on HuggingFace.
  webUrl?: string | null
}

// Whether there is anywhere to offer: a dialog with no destination is not shown.
export function canOpenArtifact(target: OpenArtifactTarget): boolean {
  return Boolean(target.uuid || target.pageUri || target.hfUrl || target.webUrl)
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
  const pageId = uuid ?? target?.pageUri
  const hfUrl = target?.hfUrl
  const externalUrl = hfUrl ?? target?.webUrl
  const openHf = () => {
    if (externalUrl) window.open(externalUrl, '_blank', 'noopener,noreferrer')
    onClose()
  }
  // Carbon's footer: primary is the registry page when there is one; with
  // HuggingFace too, both secondaries (Cancel, then HuggingFace) sit beside it.
  // HuggingFace alone takes the primary, and Cancel the secondary.
  const both = Boolean(pageId && externalUrl)
  const webLabel = hfUrl ? 'Open on HuggingFace' : 'Open link'
  return (
    <Modal
      open={target !== null}
      onRequestClose={onClose}
      modalHeading="Open artifact"
      primaryButtonText={pageId ? 'View artifact page' : webLabel}
      onRequestSubmit={() => {
        if (pageId) {
          router.push(routes.artifactHref(pageId))
          onClose()
        } else openHf()
      }}
      {...(both
        ? { secondaryButtons: [{ buttonText: 'Cancel', onClick: onClose }, { buttonText: webLabel, onClick: openHf }] }
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
                {externalUrl ? (
                  <Link href={externalUrl} target="_blank" rel="noopener noreferrer" renderIcon={Launch}>{target.uri}</Link>
                ) : (
                  <code>{target.uri}</code>
                )}
              </dd>
            </>
          )}
        </dl>
      )}
    </Modal>
  )
}
