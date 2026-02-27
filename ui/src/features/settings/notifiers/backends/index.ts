import type { NotifierBackend, NotifierBackendDef } from '../types'
import { MqttForm } from './MqttForm'
import { SendgridForm } from './SendgridForm'

const MQTT_BACKEND: NotifierBackendDef = {
  id: 'mqtt',
  label: 'MQTT',
  description: 'Publish alerts to an MQTT broker.',
  defaultConfig: {
    host: 'localhost',
    port: 1883,
    topic_template: 'homecam/alerts/{camera_name}',
  },
  validate: (config) => {
    const host = config.host
    if (typeof host !== 'string' || host.trim().length === 0) {
      return 'MQTT host is required.'
    }
    const port = config.port
    if (typeof port !== 'number' || !Number.isFinite(port) || port < 1 || port > 65535) {
      return 'MQTT port must be between 1 and 65535.'
    }
    const topicTemplate = config.topic_template
    if (typeof topicTemplate !== 'string' || topicTemplate.trim().length === 0) {
      return 'MQTT topic template is required.'
    }
    return null
  },
  component: MqttForm,
}

const SENDGRID_BACKEND: NotifierBackendDef = {
  id: 'sendgrid_email',
  label: 'SendGrid Email',
  description: 'Send alerts via SendGrid email delivery.',
  defaultConfig: {
    api_key_env: 'SENDGRID_API_KEY',
    from_email: 'homesec@localhost',
    to_emails: ['ops@localhost'],
  },
  validate: (config) => {
    const fromEmail = config.from_email
    if (typeof fromEmail !== 'string' || fromEmail.trim().length === 0) {
      return 'SendGrid from email is required.'
    }

    const toEmails = config.to_emails
    if (!Array.isArray(toEmails) || toEmails.length === 0) {
      return 'At least one SendGrid recipient email is required.'
    }
    if (!toEmails.every((item) => typeof item === 'string' && item.trim().length > 0)) {
      return 'SendGrid recipient emails must be non-empty.'
    }

    const apiKeyEnv = config.api_key_env
    if (typeof apiKeyEnv !== 'string' || apiKeyEnv.trim().length === 0) {
      return 'SendGrid API key env var is required.'
    }
    return null
  },
  component: SendgridForm,
}

export const NOTIFIER_BACKEND_ORDER: readonly NotifierBackend[] = [
  'mqtt',
  'sendgrid_email',
] as const

export const NOTIFIER_BACKENDS: Record<NotifierBackend, NotifierBackendDef> = {
  mqtt: MQTT_BACKEND,
  sendgrid_email: SENDGRID_BACKEND,
}
