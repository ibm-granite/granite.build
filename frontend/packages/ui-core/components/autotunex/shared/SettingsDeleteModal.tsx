'use client'

import { Modal, InlineNotification } from '@carbon/react'

interface Props {
  open: boolean
  count: number
  itemLabel: 'dataset' | 'configuration'
  isDeleting: boolean
  errorMessage?: string
  onClose: () => void
  onConfirm: () => void
}

/**
 * Generic danger delete-confirm modal for the Settings tables. `errorMessage`
 * is rendered inline (e.g. a 409 "in use by a running job") so the modal can
 * stay open and explain why a delete failed rather than closing silently.
 */
export function SettingsDeleteModal({
  open,
  count,
  itemLabel,
  isDeleting,
  errorMessage,
  onClose,
  onConfirm,
}: Props) {
  const plural = `${itemLabel}s`
  const heading = count > 1 ? `Delete ${count} ${plural}` : `Delete ${itemLabel}`
  return (
    <Modal
      open={open}
      danger
      modalHeading={heading}
      primaryButtonText="Delete"
      secondaryButtonText="Cancel"
      primaryButtonDisabled={isDeleting}
      onRequestClose={onClose}
      onRequestSubmit={onConfirm}
    >
      <p>This is a permanent action and cannot be undone.</p>
      {errorMessage && (
        <InlineNotification
          kind="error"
          title="Delete failed"
          subtitle={errorMessage}
          hideCloseButton
          lowContrast
          style={{ marginTop: '1rem' }}
        />
      )}
    </Modal>
  )
}
