export interface WizardStepDef {
  id: string
  title: string
  subtitle?: string
  skippable: boolean
}

export interface WizardState {
  currentStep: number
  stepData: Record<string, unknown>
  completedSteps: Set<string>
  skippedSteps: Set<string>
}
