import type { ConfigPatch } from '../../../api/client'
import {
  diffConfig,
  expectConfigBoolean,
  expectConfigObject,
  expectConfigString,
  expectConfigStringList,
  isConfigObject,
  isEnvReference,
} from '../configEditing'
import { NOTIFIER_BACKENDS } from './backends'
import type { NotifierBackend, RiskLevel } from './types'

export interface ConfiguredNotifier {
  index: number
  backend: string
  enabled: boolean
  config: Record<string, unknown>
}

export interface NotificationSettingsState {
  notifiers: ConfiguredNotifier[]
  alertPolicy: {
    backend: string
    enabled: boolean
    config: Record<string, unknown>
    minRiskLevel: RiskLevel | null
  }
}

export function isEditableNotifier(backend: string): backend is NotifierBackend {
  return backend === 'mqtt' || backend === 'sendgrid_email'
}

function notifierConfig(backend: string, config: Record<string, unknown>): Record<string, unknown> {
  if (backend === 'mqtt') {
    const port = config.port === undefined ? 1883 : config.port
    if (typeof port !== 'number' || !Number.isFinite(port)) {
      throw new Error('MQTT port must be a number.')
    }
    if (config.auth !== undefined && config.auth !== null) {
      const auth = expectConfigObject(config.auth, 'MQTT auth')
      for (const field of ['username_env', 'password_env']) {
        if (auth[field] !== undefined && auth[field] !== null) {
          expectConfigString(auth[field], `MQTT auth.${field}`)
        }
      }
    }
    return {
      ...config,
      host: config.host === undefined ? '' : expectConfigString(config.host, 'MQTT host'),
      port,
      topic_template: config.topic_template === undefined
        ? 'homecam/alerts/{camera_name}'
        : expectConfigString(config.topic_template, 'MQTT topic template'),
    }
  }
  if (backend === 'sendgrid_email') {
    return {
      ...config,
      api_key_env: config.api_key_env === undefined
        ? 'SENDGRID_API_KEY'
        : expectConfigString(config.api_key_env, 'SendGrid API key reference'),
      from_email: config.from_email === undefined ? '' : expectConfigString(config.from_email, 'SendGrid sender'),
      to_emails: config.to_emails === undefined ? [] : expectConfigStringList(config.to_emails, 'SendGrid recipients'),
    }
  }
  return { ...config }
}

export function readNotificationSettings(config: Record<string, unknown>): NotificationSettingsState {
  if (!Array.isArray(config.notifiers)) {
    throw new Error('notifiers must be a list.')
  }
  const notifiers = config.notifiers.map((raw, index): ConfiguredNotifier => {
    const entry = expectConfigObject(raw, `notifiers[${index}]`)
    const backend = expectConfigString(entry.backend, `notifiers[${index}].backend`)
    return {
      index,
      backend,
      enabled: expectConfigBoolean(entry.enabled, `notifiers[${index}].enabled`),
      config: notifierConfig(backend, expectConfigObject(entry.config, `notifiers[${index}].config`)),
    }
  })
  const policy = expectConfigObject(config.alert_policy, 'alert_policy')
  const backend = expectConfigString(policy.backend, 'alert_policy.backend')
  const policyConfig = expectConfigObject(policy.config, 'alert_policy.config')
  let minRiskLevel: RiskLevel | null = null
  if (backend === 'default') {
    const risk = policyConfig.min_risk_level === undefined ? 'medium' : policyConfig.min_risk_level
    if (risk !== 'low' && risk !== 'medium' && risk !== 'high' && risk !== 'critical') {
      throw new Error('alert_policy.config.min_risk_level is invalid.')
    }
    minRiskLevel = risk
  }
  return {
    notifiers,
    alertPolicy: {
      backend,
      enabled: expectConfigBoolean(policy.enabled, 'alert_policy.enabled'),
      config: { ...policyConfig },
      minRiskLevel,
    },
  }
}

function validateNotifier(entry: ConfiguredNotifier): void {
  if (!isEditableNotifier(entry.backend)) {
    return
  }
  const error = NOTIFIER_BACKENDS[entry.backend].validate(entry.config)
  if (error) {
    throw new Error(`Notifier ${entry.index + 1}: ${error}`)
  }
  if (entry.backend === 'mqtt') {
    if (!Number.isInteger(entry.config.port)) {
      throw new Error('MQTT port must be a whole number.')
    }
    const auth = entry.config.auth
    if (isConfigObject(auth)) {
      for (const field of ['username_env', 'password_env']) {
        const value = auth[field]
        if (value !== undefined && value !== null && value !== '' && (typeof value !== 'string' || !isEnvReference(value))) {
          throw new Error('MQTT credentials must reference environment variable names.')
        }
      }
    }
  } else if (!isEnvReference(String(entry.config.api_key_env))) {
    throw new Error('SendGrid credentials must reference an environment variable name.')
  }
}

export function buildNotificationPatch(
  original: NotificationSettingsState,
  edited: NotificationSettingsState,
): Pick<ConfigPatch, 'notifiers' | 'alert_policy'> {
  if (original.notifiers.length !== edited.notifiers.length) {
    throw new Error('Notifier instances cannot be added or removed here.')
  }
  const notifiers: NonNullable<ConfigPatch['notifiers']> = []
  for (const [index, next] of edited.notifiers.entries()) {
    const previous = original.notifiers[index]
    if (next.index !== previous.index || next.backend !== previous.backend) {
      throw new Error('Notifier order and backends cannot be changed here.')
    }
    const configPatch = diffConfig(previous.config, next.config)
    const configChanged = Object.keys(configPatch).length > 0
    const enabledChanged = previous.enabled !== next.enabled
    if (!configChanged && !enabledChanged) {
      continue
    }
    if (!isEditableNotifier(next.backend)) {
      throw new Error(`The ${next.backend} notifier is read-only.`)
    }
    if (next.enabled) {
      validateNotifier(next)
    }
    notifiers.push({
      index,
      ...(enabledChanged ? { enabled: next.enabled } : {}),
      ...(configChanged ? { config: configPatch } : {}),
    })
  }
  const patch: Pick<ConfigPatch, 'notifiers' | 'alert_policy'> = {}
  if (notifiers.length > 0) {
    patch.notifiers = notifiers
  }
  if (original.alertPolicy.minRiskLevel !== edited.alertPolicy.minRiskLevel) {
    if (original.alertPolicy.backend !== 'default' || edited.alertPolicy.minRiskLevel === null) {
      throw new Error('This alert policy is read-only.')
    }
    patch.alert_policy = { config: { min_risk_level: edited.alertPolicy.minRiskLevel } }
  }
  return patch
}
