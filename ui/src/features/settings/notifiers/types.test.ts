import { describe, expect, it } from 'vitest'

import { NOTIFIER_BACKENDS } from './backends'
import {
  type AlertPolicyFormState,
  buildAlertPolicyConfigEntry,
  buildNotificationStepData,
  defaultNotifierFormState,
  validateNotificationState,
} from './types'

describe('notifier types helpers', () => {
  it('builds alert-policy threshold from selected minimum risk level', () => {
    // Given: Notification form with mqtt enabled and minimum risk set to medium
    const notifiers = defaultNotifierFormState({
      mqtt: NOTIFIER_BACKENDS.mqtt.defaultConfig,
      sendgrid_email: NOTIFIER_BACKENDS.sendgrid_email.defaultConfig,
    })
    notifiers.mqtt.enabled = true
    const alertPolicy: AlertPolicyFormState = {
      minRiskLevel: 'medium',
    }

    // When: Building step payload for finalize
    const payload = buildNotificationStepData(notifiers, alertPolicy)

    // Then: Alert policy min risk level is preserved in payload
    expect(payload.alert_policy.config.min_risk_level).toBe('medium')
    expect(payload.notifiers).toHaveLength(1)
  })

  it('validates selected notifier config', () => {
    // Given: Notification state with enabled mqtt but empty host
    const notifiers = defaultNotifierFormState({
      mqtt: { ...NOTIFIER_BACKENDS.mqtt.defaultConfig, host: '' },
      sendgrid_email: NOTIFIER_BACKENDS.sendgrid_email.defaultConfig,
    })
    notifiers.mqtt.enabled = true
    // When: Validating notifier state
    const error = validateNotificationState(notifiers, {
      mqtt: NOTIFIER_BACKENDS.mqtt.validate,
      sendgrid_email: NOTIFIER_BACKENDS.sendgrid_email.validate,
    })

    // Then: Validation surfaces notifier backend config error
    expect(error).toBe('MQTT host is required.')
  })

  it('does not require extra alert-policy validation for enabled notifiers', () => {
    // Given: Notification state with valid notifier config and default risk threshold
    const notifiers = defaultNotifierFormState({
      mqtt: NOTIFIER_BACKENDS.mqtt.defaultConfig,
      sendgrid_email: NOTIFIER_BACKENDS.sendgrid_email.defaultConfig,
    })

    // When: Enabling mqtt notifier and re-validating
    notifiers.mqtt.enabled = true
    const enabledNotifierError = validateNotificationState(notifiers, {
      mqtt: NOTIFIER_BACKENDS.mqtt.validate,
      sendgrid_email: NOTIFIER_BACKENDS.sendgrid_email.validate,
    })

    // Then: Validation passes because threshold is represented as a single required value
    expect(enabledNotifierError).toBeNull()
  })

  it('builds canonical alert-policy config entry', () => {
    // Given: Minimum risk threshold chosen by operator
    const minRiskLevel = 'critical'

    // When: Building API payload entry for alert policy
    const payload = buildAlertPolicyConfigEntry(minRiskLevel)

    // Then: Payload keeps canonical backend/enabled fields and selected threshold
    expect(payload).toEqual({
      backend: 'default',
      enabled: true,
      config: {
        min_risk_level: 'critical',
      },
    })
  })
})
