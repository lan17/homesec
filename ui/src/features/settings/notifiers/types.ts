import type { JSX } from 'react'

import type { TestConnectionRequest } from '../../../api/generated/types'

export type NotifierBackend = 'mqtt' | 'sendgrid_email'
export type RiskLevel = 'low' | 'medium' | 'high' | 'critical'

export interface NotifierEntryState {
  enabled: boolean
  config: Record<string, unknown>
}

export interface NotifierFormState {
  mqtt: NotifierEntryState
  sendgrid_email: NotifierEntryState
}

export interface AlertPolicyFormState {
  selectedRiskLevels: RiskLevel[]
}

export interface NotificationStepData {
  notifiers: Array<{
    backend: string
    enabled: boolean
    config: Record<string, unknown>
  }>
  alert_policy: {
    backend: string
    enabled: boolean
    config: Record<string, unknown>
  }
}

export interface NotifierBackendFormProps {
  config: Record<string, unknown>
  onChange: (config: Record<string, unknown>) => void
}

export interface NotifierBackendDef {
  id: NotifierBackend
  label: string
  description: string
  defaultConfig: Record<string, unknown>
  validate: (config: Record<string, unknown>) => string | null
  component: (props: NotifierBackendFormProps) => JSX.Element
}

const RISK_LEVEL_ORDER: readonly RiskLevel[] = ['low', 'medium', 'high', 'critical'] as const

function normalizeRiskLevels(values: readonly string[]): RiskLevel[] {
  const normalized: RiskLevel[] = []
  for (const candidate of RISK_LEVEL_ORDER) {
    if (values.includes(candidate)) {
      normalized.push(candidate)
    }
  }
  return normalized
}

function lowestRiskLevel(values: readonly RiskLevel[]): RiskLevel {
  const normalized = normalizeRiskLevels(values)
  return normalized[0] ?? 'high'
}

export function defaultNotifierFormState(
  defaults: Record<NotifierBackend, Record<string, unknown>>,
): NotifierFormState {
  return {
    mqtt: {
      enabled: false,
      config: defaults.mqtt,
    },
    sendgrid_email: {
      enabled: false,
      config: defaults.sendgrid_email,
    },
  }
}

export function defaultAlertPolicyFormState(): AlertPolicyFormState {
  return {
    selectedRiskLevels: ['high', 'critical'],
  }
}

function normalizeNotifierEntry(
  entry: { enabled: boolean; config: Record<string, unknown> } | undefined,
  fallback: Record<string, unknown>,
): NotifierEntryState {
  return {
    enabled: Boolean(entry?.enabled),
    config: entry?.config ?? fallback,
  }
}

export function notificationFormStateFromStepData(
  value: NotificationStepData | null | undefined,
  defaults: Record<NotifierBackend, Record<string, unknown>>,
): { notifiers: NotifierFormState; alertPolicy: AlertPolicyFormState } {
  const byBackend = new Map<string, { enabled: boolean; config: Record<string, unknown> }>()
  for (const entry of value?.notifiers ?? []) {
    if (typeof entry.backend !== 'string' || typeof entry.enabled !== 'boolean') {
      continue
    }
    if (!entry.config || typeof entry.config !== 'object' || Array.isArray(entry.config)) {
      continue
    }
    byBackend.set(entry.backend, {
      enabled: entry.enabled,
      config: entry.config,
    })
  }

  const risk = value?.alert_policy?.config?.min_risk_level
  const selectedRiskLevels: RiskLevel[] =
    risk === 'low'
      ? ['low', 'medium', 'high', 'critical']
      : risk === 'medium'
        ? ['medium', 'high', 'critical']
        : risk === 'high'
          ? ['high', 'critical']
          : risk === 'critical'
            ? ['critical']
            : defaultAlertPolicyFormState().selectedRiskLevels

  return {
    notifiers: {
      mqtt: normalizeNotifierEntry(byBackend.get('mqtt'), defaults.mqtt),
      sendgrid_email: normalizeNotifierEntry(
        byBackend.get('sendgrid_email'),
        defaults.sendgrid_email,
      ),
    },
    alertPolicy: {
      selectedRiskLevels,
    },
  }
}

export function validateNotificationState(
  value: NotifierFormState,
  alertPolicy: AlertPolicyFormState,
  validators: Record<NotifierBackend, (config: Record<string, unknown>) => string | null>,
): string | null {
  const enabledBackends: NotifierBackend[] = []
  if (value.mqtt.enabled) {
    enabledBackends.push('mqtt')
  }
  if (value.sendgrid_email.enabled) {
    enabledBackends.push('sendgrid_email')
  }

  for (const backend of enabledBackends) {
    const entry = value[backend]
    const error = validators[backend](entry.config)
    if (error) {
      return error
    }
  }

  if (normalizeRiskLevels(alertPolicy.selectedRiskLevels).length === 0) {
    return 'Select at least one risk level for alert policy.'
  }

  return null
}

export function buildNotificationStepData(
  value: NotifierFormState,
  alertPolicy: AlertPolicyFormState,
): NotificationStepData {
  const notifiers: NotificationStepData['notifiers'] = []
  if (value.mqtt.enabled) {
    notifiers.push({
      backend: 'mqtt',
      enabled: true,
      config: value.mqtt.config,
    })
  }
  if (value.sendgrid_email.enabled) {
    notifiers.push({
      backend: 'sendgrid_email',
      enabled: true,
      config: value.sendgrid_email.config,
    })
  }

  return {
    notifiers,
    alert_policy: {
      backend: 'default',
      enabled: true,
      config: {
        min_risk_level: lowestRiskLevel(alertPolicy.selectedRiskLevels),
      },
    },
  }
}

export function buildNotifierTestRequest(
  backend: NotifierBackend,
  config: Record<string, unknown>,
): TestConnectionRequest {
  return {
    type: 'notifier',
    backend,
    config,
  }
}
