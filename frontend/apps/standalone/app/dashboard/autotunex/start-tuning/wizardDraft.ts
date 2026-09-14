import type { WizardDraft } from '@granite-build/ui-core/types'

const DRAFT_KEY = 'autotunex_wizard_draft'
const DRAFT_MAX_AGE_MS = 24 * 60 * 60 * 1000 // 24 hours

export function saveDraft(draft: WizardDraft): void {
  try {
    localStorage.setItem(DRAFT_KEY, JSON.stringify(draft))
  } catch {
    // Silently ignore storage errors
  }
}

export function loadDraft(): WizardDraft | null {
  try {
    const raw = localStorage.getItem(DRAFT_KEY)
    if (!raw) return null
    const draft: WizardDraft = JSON.parse(raw)
    const age = Date.now() - new Date(draft.savedAt).getTime()
    if (age >= DRAFT_MAX_AGE_MS || !draft.selectedGoal) {
      localStorage.removeItem(DRAFT_KEY)
      return null
    }
    return draft
  } catch {
    return null
  }
}

export interface ResolvedDraft {
  draft: WizardDraft
  /** What the restore could not carry over, phrased for the user. */
  notes: string[]
}

/**
 * Reconciles a loaded draft with what can actually be restored.
 *
 * Two things a draft cannot carry:
 *
 *  - An uploaded File, which is not serializable. A draft that reached Step 2 by
 *    uploading a file therefore has no dataset to launch against; only one that
 *    picked an EXISTING dataset can treat Step 1 as done.
 *  - Any guarantee that the dataset or configuration it names still exists. A draft
 *    lives up to 24 hours, so either may have been deleted since it was written.
 *
 * Anything unrestorable rewinds the wizard to the step that owns it, rather than
 * leaving a step marked complete with nothing behind it — which would let the user
 * walk to Review and launch against a dead id.
 */
export function resolveDraft(
  draft: WizardDraft,
  exists: { dataset: boolean; config: boolean }
): ResolvedDraft {
  const notes: string[] = []
  let currentStep = draft.currentStep
  let completedSteps = [...draft.completedSteps]
  let existingDatasetId = draft.existingDatasetId
  let selectedConfigId = draft.selectedConfigId

  function rewindTo(step: number) {
    currentStep = Math.min(currentStep, step)
    completedSteps = completedSteps.map((done, i) => (i >= step ? false : done))
  }

  if (!selectedConfigId) {
    // Either nothing was chosen, or it was an unsaved "__pending__" config, which
    // saveDraft stores as null because it has no id to come back to.
    if (draft.completedSteps[2]) {
      notes.push('A configuration you had not saved yet cannot be restored — choose one again.')
    }
    rewindTo(2)
  } else if (!exists.config) {
    notes.push('The configuration this draft used no longer exists — choose another.')
    selectedConfigId = null
    rewindTo(2)
  }

  if (!existingDatasetId) {
    if (draft.completedSteps[1]) {
      notes.push('An uploaded file cannot be saved in a draft — re-select or re-upload your dataset.')
    }
    rewindTo(1)
  } else if (!exists.dataset) {
    notes.push('The dataset this draft used no longer exists — choose another.')
    existingDatasetId = null
    rewindTo(1)
  }

  return {
    draft: { ...draft, currentStep, completedSteps, existingDatasetId, selectedConfigId },
    notes,
  }
}

export function clearDraft(): void {
  try {
    localStorage.removeItem(DRAFT_KEY)
  } catch {
    // Silently ignore storage errors
  }
}
