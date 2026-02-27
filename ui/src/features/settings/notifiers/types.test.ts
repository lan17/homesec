import { describe, expect, it } from 'vitest'

import { NOTIFIER_BACKENDS } from './backends'
import {
  type AlertPolicyFormState,
  buildNotificationStepData,
  defaultAlertPolicyFormState,
  defaultNotifierFormState,
  validateNotificationState,
} from './types'

describe('notifier types helpers', () => {
  it('builds alert-policy threshold from selected risk levels', () => {
    // Given: Notification form with mqtt enabled and medium+high risk selected
    const notifiers = defaultNotifierFormState({
      mqtt: NOTIFIER_BACKENDS.mqtt.defaultConfig,
      sendgrid_email: NOTIFIER_BACKENDS.sendgrid_email.defaultConfig,
    })
    notifiers.mqtt.enabled = true
    const alertPolicy: AlertPolicyFormState = {
      selectedRiskLevels: ['medium', 'high', 'critical'],
    }

    // When: Building step payload for finalize
    const payload = buildNotificationStepData(notifiers, alertPolicy)

    // Then: Alert policy min risk level maps to lowest selected severity
    expect(payload.alert_policy.config.min_risk_level).toBe('medium')
    expect(payload.notifiers).toHaveLength(1)
  })

  it('validates selected notifier config and risk-level selection', () => {
    // Given: Notification state with enabled mqtt but empty host
    const notifiers = defaultNotifierFormState({
      mqtt: { ...NOTIFIER_BACKENDS.mqtt.defaultConfig, host: '' },
      sendgrid_email: NOTIFIER_BACKENDS.sendgrid_email.defaultConfig,
    })
    notifiers.mqtt.enabled = true
    const alertPolicy = defaultAlertPolicyFormState()

    // When: Validating notifier and alert-policy state
    const error = validateNotificationState(notifiers, alertPolicy, {
      mqtt: NOTIFIER_BACKENDS.mqtt.validate,
      sendgrid_email: NOTIFIER_BACKENDS.sendgrid_email.validate,
    })

    // Then: Validation surfaces notifier backend config error
    expect(error).toBe('MQTT host is required.')
  })
})
