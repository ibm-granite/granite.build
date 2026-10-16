'use client'

import { Modal, InlineNotification } from '@carbon/react'

interface Props {
  open: boolean
  count: number
  onClose: () => void
  onConfirm: () => void
  isDeleting: boolean
  /** Rendered inline so the modal can stay open and explain a failed delete. */
  errorMessage?: string
}

export function TuningDeleteModal({ open, count, onClose, onConfirm, isDeleting, errorMessage }: Props) {
  return (
    <Modal
      open={open}
      danger
      modalHeading={count > 1 ? `Delete ${count} tunings` : 'Delete tuning'}
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
