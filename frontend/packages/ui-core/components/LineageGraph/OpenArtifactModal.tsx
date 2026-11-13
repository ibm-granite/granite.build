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

import * as React from 'react'
import { useRouter } from 'next/navigation'
import { Link, Modal } from '@carbon/react'
import { Launch } from '@carbon/icons-react'
import { useRoutes } from '../../config/routes'
import { getHuggingFaceUrl } from './diagramUtilities'
import styles from './OpenArtifactModal.module.scss'

// What a lineage click offers to open. Nothing opens without this dialog, so a
// mis-click on a node never navigates.
export interface OpenArtifactTarget {
  name: string
  uri?: string
  // Registry UUID, when the node already names one.
  uuid?: string
  artifactType?: string
}

// The page an hf:// or http(s):// URI names, for the URI's link; none for other schemes.
function webUrl(uri: string | undefined): string | null {
  if (!uri) return null
  if (uri.startsWith('hf://')) return getHuggingFaceUrl(uri)
  return /^https?:\/\//.test(uri) ? getHuggingFaceUrl(uri) ?? uri : null
}

// One dialog for every scheme. The registry is not asked until "View artifact
// page" is clicked: the artifact page resolves a URI itself, and says so when
// nothing is registered under it.
export default function OpenArtifactModal({
  target,
  onClose,
}: {
  target: OpenArtifactTarget | null
  onClose: () => void
}) {
  const router = useRouter()
  const routes = useRoutes()
  const pageId = target?.uuid ?? target?.uri
  const link = webUrl(target?.uri)
  return (
    <Modal
      open={target !== null}
      onRequestClose={onClose}
      modalHeading="Open artifact"
      primaryButtonText="View artifact page"
      primaryButtonDisabled={!pageId}
      secondaryButtonText="Cancel"
      onSecondarySubmit={onClose}
      onRequestSubmit={() => {
        if (pageId) router.push(routes.artifactHref(pageId))
        onClose()
      }}
      size="sm"
    >
      {target && (
        // What the dialog is about to open, so a mis-click is recognisable.
        <dl className={styles.summary}>
          <dt>Name</dt>
          <dd>{target.name}</dd>
          {target.uuid && (
            <>
              <dt>ID</dt>
              <dd><code>{target.uuid}</code></dd>
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
                {link ? (
                  <Link href={link} target="_blank" rel="noopener noreferrer" renderIcon={Launch}>{target.uri}</Link>
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
