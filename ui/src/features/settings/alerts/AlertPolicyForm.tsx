import {
  ALERT_POLICY_RISK_LEVELS,
  type AlertPolicyFormState,
} from '../notifiers/types'

interface AlertPolicyFormProps {
  value: AlertPolicyFormState
  onChange: (value: AlertPolicyFormState) => void
}

export function AlertPolicyForm({ value, onChange }: AlertPolicyFormProps) {
  return (
    <section className="inline-form">
      <h3 className="backend-picker__title">Alert policy baseline</h3>
      <p className="subtle">Choose the minimum risk level that should trigger notifications.</p>
      <label className="field-label" htmlFor="setup-notifier-min-risk-level">
        Minimum risk level
        <select
          id="setup-notifier-min-risk-level"
          className="input"
          value={value.minRiskLevel}
          onChange={(event) => {
            const nextValue = event.target.value
            if (
              nextValue === 'low'
              || nextValue === 'medium'
              || nextValue === 'high'
              || nextValue === 'critical'
            ) {
              onChange({
                minRiskLevel: nextValue,
              })
            }
          }}
        >
          {ALERT_POLICY_RISK_LEVELS.map((riskLevel) => (
            <option key={riskLevel} value={riskLevel}>
              {riskLevel}
            </option>
          ))}
        </select>
      </label>
    </section>
  )
}
