import type { JSX } from 'react'

import type { TestConnectionRequest } from '../../../api/generated/types'

export type NotifierBackend = 'mqtt' | 'sendgrid_email'
export type RiskLevel = 'low' | 'medium' | 'high' | 'critical'

export const NOTIFIER_BACKEND_IDS: readonly NotifierBackend[] = [
  'mqtt',
  'sendgrid_email',
] as const

export const ALERT_POLICY_RISK_LEVELS: readonly RiskLevel[] = [
  'low',
  'medium',
  'high',
  'critical',
] as const

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

function normalizeRiskLevels(values: readonly string[]): RiskLevel[] {
  const normalized: RiskLevel[] = []
  for (const candidate of ALERT_POLICY_RISK_LEVELS) {
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
  const enabledBackends = enabledNotifierBackends(value)

  for (const backend of enabledBackends) {
    const entry = value[backend]
    const error = validators[backend](entry.config)
    if (error) {
      return error
    }
  }

  if (
    enabledBackends.length > 0
    && normalizeRiskLevels(alertPolicy.selectedRiskLevels).length === 0
  ) {
    return 'Select at least one risk level for alert policy.'
  }

  return null
}

export function enabledNotifierBackends(value: NotifierFormState): NotifierBackend[] {
  const enabled: NotifierBackend[] = []
  for (const backendId of NOTIFIER_BACKEND_IDS) {
    if (value[backendId].enabled) {
      enabled.push(backendId)
    }
  }
  return enabled
}

export function buildNotificationStepData(
  value: NotifierFormState,
  alertPolicy: AlertPolicyFormState,
): NotificationStepData {
  const notifiers: NotificationStepData['notifiers'] = enabledNotifierBackends(value).map(
    (backendId) => ({
      backend: backendId,
      enabled: true,
      config: value[backendId].config,
    }),
  )

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
