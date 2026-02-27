import type { AlertPolicyFormState, RiskLevel } from '../notifiers/types'

interface AlertPolicyFormProps {
  value: AlertPolicyFormState
  onChange: (value: AlertPolicyFormState) => void
}

const RISK_LEVELS: readonly RiskLevel[] = ['low', 'medium', 'high', 'critical'] as const

export function AlertPolicyForm({ value, onChange }: AlertPolicyFormProps) {
  return (
    <section className="inline-form">
      <h3 className="backend-picker__title">Alert policy baseline</h3>
      <p className="subtle">Select risk levels that should trigger notifications.</p>
      <div className="alert-policy__grid">
        {RISK_LEVELS.map((riskLevel) => {
          const checked = value.selectedRiskLevels.includes(riskLevel)
          return (
            <label key={riskLevel} className="field-label camera-checkbox-field">
              <input
                type="checkbox"
                checked={checked}
                onChange={(event) => {
                  const nextValues = event.target.checked
                    ? [...value.selectedRiskLevels, riskLevel]
                    : value.selectedRiskLevels.filter((item) => item !== riskLevel)
                  onChange({
                    selectedRiskLevels: nextValues,
                  })
                }}
              />
              {riskLevel}
            </label>
          )
        })}
      </div>
    </section>
  )
}
