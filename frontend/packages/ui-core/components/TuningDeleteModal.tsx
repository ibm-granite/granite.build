'use client'

import { Modal } from '@carbon/react'

interface Props {
  open: boolean
  count: number
  onClose: () => void
  onConfirm: () => void
  isDeleting: boolean
}

export function TuningDeleteModal({ open, count, onClose, onConfirm, isDeleting }: Props) {
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
    </Modal>
  )
}
