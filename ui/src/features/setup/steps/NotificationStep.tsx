import { useState } from 'react'

import type { TestConnectionResponse } from '../../../api/generated/types'
import { Button } from '../../../components/ui/Button'
import { AlertPolicyForm } from '../../settings/alerts/AlertPolicyForm'
import { NOTIFIER_BACKENDS } from '../../settings/notifiers/backends'
import { NotifierConfigForm } from '../../settings/notifiers/NotifierConfigForm'
import {
  buildNotificationStepData,
  buildNotifierTestRequest,
  enabledNotifierBackends,
  NOTIFIER_BACKEND_IDS,
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

function emptyNotifierResults(): NotifierResultByBackend {
  const entries = NOTIFIER_BACKEND_IDS.map((backendId) => [backendId, null] as const)
  return Object.fromEntries(entries) as NotifierResultByBackend
}

function clearStaleNotifierResults(
  currentNotifiers: NotifierFormState,
  nextNotifiers: NotifierFormState,
  currentResults: NotifierResultByBackend,
): NotifierResultByBackend {
  const nextResults = { ...currentResults }
  for (const backendId of NOTIFIER_BACKEND_IDS) {
    const previous = currentNotifiers[backendId]
    const next = nextNotifiers[backendId]
    const enabledChanged = previous.enabled !== next.enabled
    const configChanged = previous.config !== next.config
    if (!next.enabled || enabledChanged || configChanged) {
      nextResults[backendId] = null
    }
  }
  return nextResults
}

export function NotificationStep({
  initialData,
  onComplete,
  onUpdateData,
  onSkip,
}: NotificationStepProps) {
  const [initialState] = useState(() =>
    notificationFormStateFromStepData(initialData, DEFAULT_NOTIFIER_CONFIGS),
  )
  const [notifiers, setNotifiers] = useState(initialState.notifiers)
  const [alertPolicy, setAlertPolicy] = useState(initialState.alertPolicy)
  const [validationError, setValidationError] = useState<string | null>(null)
  const [resultsByBackend, setResultsByBackend] = useState<NotifierResultByBackend>(
    emptyNotifierResults,
  )

  function handleSaveAndContinue(): void {
    const maybeError = validateNotificationState(notifiers, {
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
          setResultsByBackend((currentResults) =>
            clearStaleNotifierResults(notifiers, nextValue, currentResults),
          )
          setNotifiers(nextValue)
          setValidationError(null)
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
    </section>
  )
}
