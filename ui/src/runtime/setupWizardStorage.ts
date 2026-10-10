export const WIZARD_STATE_STORAGE_KEY = 'homesec.setup.wizard'

export function clearPersistedSetupWizardState(): void {
  if (typeof window !== 'undefined') {
    window.localStorage.removeItem(WIZARD_STATE_STORAGE_KEY)
  }
}
