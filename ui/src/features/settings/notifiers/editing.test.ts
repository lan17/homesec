import { describe, expect, it } from 'vitest'

import { buildNotificationPatch, readNotificationSettings } from './editing'

function savedConfig() {
  return {
    notifiers: [
      { backend: 'mqtt', enabled: true, config: { host: 'one', qos: 2, auth: { username_env: 'USER_ONE', password_env: 'PASSWORD_ONE' } } },
      { backend: 'mqtt', enabled: true, config: { host: 'two', retain: true } },
      { backend: 'custom', enabled: false, config: { custom_setting: 'preserve' } },
    ],
    alert_policy: { backend: 'default', enabled: false, config: { min_risk_level: 'medium', notify_on_motion: true, overrides: { front: { min_risk_level: 'critical' } } } },
  }
}

describe('existing notification configuration editing', () => {
  it('preserves duplicate notifier instances and disabled custom integrations', () => {
    // Given: Duplicate MQTT instances and a disabled custom notifier
    const original = readNotificationSettings(savedConfig())
    // When: The second MQTT destination is edited
    const edited = { ...original, notifiers: original.notifiers.map((entry) => entry.index === 1 ? { ...entry, config: { ...entry.config, host: 'new-two' } } : entry) }
    // Then: The index patch identifies only that instance and preserves every other field
    expect(original.notifiers.map((entry) => entry.backend)).toEqual(['mqtt', 'mqtt', 'custom'])
    expect(buildNotificationPatch(original, edited)).toEqual({ notifiers: [{ index: 1, config: { host: 'new-two' } }] })
    expect(original.notifiers[2]).toMatchObject({ enabled: false, config: { custom_setting: 'preserve' } })
  })

  it('edits an auth reference without resending other credentials or defaults', () => {
    // Given: MQTT has advanced settings and a configured password reference
    const original = readNotificationSettings(savedConfig())
    const entry = original.notifiers[0]
    // When: Only the username environment reference changes
    const edited = { ...original, notifiers: [{ ...entry, config: { ...entry.config, auth: { username_env: 'NEW_USER', password_env: 'PASSWORD_ONE' } } }, ...original.notifiers.slice(1)] }
    // Then: The recursive patch keeps password references, QoS, and omitted server defaults intact
    expect(buildNotificationPatch(original, edited)).toEqual({ notifiers: [{ index: 0, config: { auth: { username_env: 'NEW_USER' } } }] })
  })

  it('changes only the default minimum risk while retaining disabled state and overrides', () => {
    // Given: A disabled policy has per-camera overrides and other rules
    const original = readNotificationSettings(savedConfig())
    // When: Its baseline threshold changes
    const edited = { ...original, alertPolicy: { ...original.alertPolicy, minRiskLevel: 'high' as const } }
    // Then: Only the guided config field is patched
    expect(buildNotificationPatch(original, edited)).toEqual({ alert_policy: { config: { min_risk_level: 'high' } } })
    expect(original.alertPolicy.config.overrides).toEqual({ front: { min_risk_level: 'critical' } })
    expect(original.alertPolicy.enabled).toBe(false)
  })

  it('disables one notifier without clearing its configuration', () => {
    // Given: An enabled notifier with stored advanced configuration
    const original = readNotificationSettings(savedConfig())
    // When: That instance is disabled
    const edited = { ...original, notifiers: original.notifiers.map((entry) => entry.index === 0 ? { ...entry, enabled: false } : entry) }
    // Then: The patch only changes enabled state
    expect(buildNotificationPatch(original, edited)).toEqual({ notifiers: [{ index: 0, enabled: false }] })
  })

  it('rejects malformed list payloads and edits to custom integrations', () => {
    // Given: A custom notifier and an untrusted payload with an invalid enabled value
    const original = readNotificationSettings(savedConfig())
    // When/Then: Invalid payloads and unsupported writes fail explicitly
    expect(() => readNotificationSettings({ ...savedConfig(), notifiers: [{ backend: 'mqtt', enabled: 'yes', config: {} }] })).toThrow('must be a boolean')
    expect(() => readNotificationSettings({ ...savedConfig(), notifiers: [{ backend: 'mqtt', enabled: true, config: { host: 'one', port: '1883' } }] })).toThrow('must be a number')
    const edited = { ...original, notifiers: original.notifiers.map((entry) => entry.index === 2 ? { ...entry, enabled: true } : entry) }
    expect(() => buildNotificationPatch(original, edited)).toThrow('read-only')
  })

  it('validates changed connection fields and environment variable references', () => {
    // Given: Valid saved MQTT settings
    const original = readNotificationSettings(savedConfig())
    // When/Then: A fractional port or literal credential is rejected before saving
    const change = (config: Record<string, unknown>) => ({ ...original, notifiers: [{ ...original.notifiers[0], config: { ...original.notifiers[0].config, ...config } }, ...original.notifiers.slice(1)] })
    expect(() => buildNotificationPatch(original, change({ port: 1883.5 }))).toThrow('whole number')
    expect(() => buildNotificationPatch(original, change({ auth: { password_env: 'literal secret' } }))).toThrow('environment variable')
  })

  it('allows optional MQTT references to clear without rewriting an unchanged external value', () => {
    // Given: A saved external username reference uses an operator-specific name
    const config = savedConfig()
    config.notifiers[0].config.auth!.username_env = 'EXTERNAL-USER'
    const original = readNotificationSettings(config)
    const entry = original.notifiers[0]

    // When: Only the optional password reference is cleared
    const edited = { ...original, notifiers: [{ ...entry, config: { ...entry.config,
      auth: { username_env: 'EXTERNAL-USER', password_env: '' },
    } }, ...original.notifiers.slice(1)] }

    // Then: Clearing stays supported while unchanged references are preserved
    expect(buildNotificationPatch(original, edited)).toEqual({ notifiers: [{ index: 0,
      config: { auth: { password_env: '' } },
    }] })
  })

})
