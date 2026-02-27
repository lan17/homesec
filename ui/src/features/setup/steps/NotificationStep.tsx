import { useMemo, useState } from 'react'

import type { TestConnectionResponse } from '../../../api/generated/types'
import { Button } from '../../../components/ui/Button'
import { AlertPolicyForm } from '../../settings/alerts/AlertPolicyForm'
import { NOTIFIER_BACKENDS } from '../../settings/notifiers/backends'
import { NotifierConfigForm } from '../../settings/notifiers/NotifierConfigForm'
import {
  buildNotificationStepData,
  buildNotifierTestRequest,
  notificationFormStateFromStepData,
  type NotificationStepData,
  type NotifierBackend,
  type NotifierFormState,
  validateNotificationState,
} from '../../settings/notifiers/types'
import { TestConnectionButton } from '../../shared/TestConnectionButton'

interface NotificationStepProps {
  initialData: NotificationStepData | null
  onComplete: () => void
  onUpdateData: (data: NotificationStepData) => void
  onSkip: () => void
}

type NotifierResultByBackend = Partial<Record<NotifierBackend, TestConnectionResponse | null>>

const DEFAULT_NOTIFIER_CONFIGS = {
  mqtt: NOTIFIER_BACKENDS.mqtt.defaultConfig,
  sendgrid_email: NOTIFIER_BACKENDS.sendgrid_email.defaultConfig,
}

function enabledNotifierBackends(value: NotifierFormState): NotifierBackend[] {
  const backends: NotifierBackend[] = []
  if (value.mqtt.enabled) {
    backends.push('mqtt')
  }
  if (value.sendgrid_email.enabled) {
    backends.push('sendgrid_email')
  }
  return backends
}

export function NotificationStep({
  initialData,
  onComplete,
  onUpdateData,
  onSkip,
}: NotificationStepProps) {
  const initialState = useMemo(
    () =>
      notificationFormStateFromStepData(initialData, DEFAULT_NOTIFIER_CONFIGS),
    [initialData],
  )
  const [notifiers, setNotifiers] = useState(initialState.notifiers)
  const [alertPolicy, setAlertPolicy] = useState(initialState.alertPolicy)
  const [validationError, setValidationError] = useState<string | null>(null)
  const [resultsByBackend, setResultsByBackend] = useState<NotifierResultByBackend>({
    mqtt: null,
    sendgrid_email: null,
  })

  function handleSaveAndContinue(): void {
    const maybeError = validateNotificationState(notifiers, alertPolicy, {
      mqtt: NOTIFIER_BACKENDS.mqtt.validate,
      sendgrid_email: NOTIFIER_BACKENDS.sendgrid_email.validate,
    })
    if (maybeError) {
      setValidationError(maybeError)
      return
    }
    setValidationError(null)
    onUpdateData(buildNotificationStepData(notifiers, alertPolicy))
    onComplete()
  }

  const enabledBackends = enabledNotifierBackends(notifiers)

  return (
    <section className="wizard-step-card">
      <NotifierConfigForm
        value={notifiers}
        onChange={(nextValue) => {
          setNotifiers(nextValue)
          setValidationError(null)
          setResultsByBackend((current) => {
            const nextState = { ...current }
            if (!nextValue.mqtt.enabled) {
              nextState.mqtt = null
            }
            if (!nextValue.sendgrid_email.enabled) {
              nextState.sendgrid_email = null
            }
            return nextState
          })
        }}
      />

      {enabledBackends.length > 0 ? (
        <section className="inline-form">
          <h3 className="backend-picker__title">Test notifications</h3>
          {enabledBackends.map((backendId) => {
            const backend = NOTIFIER_BACKENDS[backendId]
            return (
              <section key={backendId} className="notifier-config__test-card">
                <p className="field-label">{backend.label}</p>
                <TestConnectionButton
                  request={buildNotifierTestRequest(backendId, notifiers[backendId].config)}
                  result={resultsByBackend[backendId] ?? null}
                  onResult={(result) => {
                    setResultsByBackend((current) => ({
                      ...current,
                      [backendId]: result,
                    }))
                  }}
                  idleLabel={`Test ${backend.label}`}
                  retryLabel={`Retry ${backend.label} test`}
                  pendingLabel={`Testing ${backend.label}...`}
                />
              </section>
            )
          })}
        </section>
      ) : null}

      <AlertPolicyForm
        value={alertPolicy}
        onChange={(nextValue) => {
          setAlertPolicy(nextValue)
          setValidationError(null)
        }}
      />

      {validationError ? <p className="error-text">{validationError}</p> : null}

      <div className="inline-form__actions">
        <Button variant="ghost" onClick={onSkip}>
          Skip notification step
        </Button>
        <Button onClick={handleSaveAndContinue}>Save and continue</Button>
      </div>
      {enabledBackends.length === 0 ? (
        <p className="subtle">
          No notifier selected. You can still continue and add one later.
        </p>
      ) : null}
      {enabledBackends.length > 0 && alertPolicy.selectedRiskLevels.length === 0 ? (
        <p className="subtle">
          Choose at least one risk level to decide when enabled notifiers should trigger.
        </p>
      ) : null}
    </section>
  )
}
