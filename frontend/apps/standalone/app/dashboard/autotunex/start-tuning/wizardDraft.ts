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

export function clearDraft(): void {
  try {
    localStorage.removeItem(DRAFT_KEY)
  } catch {
    // Silently ignore storage errors
  }
}
