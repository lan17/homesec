import type { CameraCreate } from '../../api/generated/types'
import type { DetectionStepData } from '../settings/detection/types'
import type { NotificationStepData } from '../settings/notifiers/types'
import type { StorageFormState } from '../settings/storage/types'

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function readNestedRecord(value: unknown, key: string): Record<string, unknown> | null {
  if (!isRecord(value)) {
    return null
  }
  const nested = value[key]
  if (!isRecord(nested)) {
    return null
  }
  return nested
}

export function parseCameraStepDraft(value: unknown): CameraCreate | null {
  const camera = readNestedRecord(value, 'camera')
  if (!camera) {
    return null
  }
  return camera as CameraCreate
}

export function parseStorageStepDraft(value: unknown): StorageFormState | null {
  const storage = readNestedRecord(value, 'storage')
  if (!storage) {
    return null
  }

  const backend = storage.backend
  const config = storage.config
  if ((backend === 'local' || backend === 'dropbox') && isRecord(config)) {
    return {
      backend,
      config,
    }
  }
  return null
}

export function parseDetectionStepDraft(value: unknown): DetectionStepData | null {
  const detection = readNestedRecord(value, 'detection')
  if (!detection) {
    return null
  }

  const filter = detection.filter
  if (!isRecord(filter) || typeof filter.backend !== 'string' || !isRecord(filter.config)) {
    return null
  }
  const parsedFilter: DetectionStepData['filter'] = {
    backend: filter.backend,
    config: filter.config,
  }

  const vlm = detection.vlm
  if (vlm !== null) {
    if (
      !isRecord(vlm)
      || typeof vlm.backend !== 'string'
      || !isRecord(vlm.config)
      || !Array.isArray(vlm.trigger_classes)
      || !vlm.trigger_classes.every((item) => typeof item === 'string')
      || (vlm.run_mode !== 'trigger_only' && vlm.run_mode !== 'always' && vlm.run_mode !== 'never')
    ) {
      return null
    }

    return {
      filter: parsedFilter,
      vlm: {
        backend: vlm.backend,
        config: vlm.config,
        run_mode: vlm.run_mode,
        trigger_classes: [...vlm.trigger_classes],
      },
    }
  }

  return {
    filter: parsedFilter,
    vlm: null,
  }
}

export function parseNotificationStepDraft(value: unknown): NotificationStepData | null {
  const notifications = readNestedRecord(value, 'notifications')
  if (!notifications) {
    return null
  }

  const notifiers = notifications.notifiers
  if (!Array.isArray(notifiers)) {
    return null
  }
  const parsedNotifiers: NotificationStepData['notifiers'] = []
  for (const entry of notifiers) {
    if (!isRecord(entry)) {
      return null
    }
    if (
      typeof entry.backend !== 'string'
      || typeof entry.enabled !== 'boolean'
      || !isRecord(entry.config)
    ) {
      return null
    }
    parsedNotifiers.push({
      backend: entry.backend,
      enabled: entry.enabled,
      config: entry.config,
    })
  }

  const alertPolicy = notifications.alert_policy
  if (
    !isRecord(alertPolicy)
    || typeof alertPolicy.backend !== 'string'
    || typeof alertPolicy.enabled !== 'boolean'
    || !isRecord(alertPolicy.config)
  ) {
    return null
  }

  return {
    notifiers: parsedNotifiers,
    alert_policy: {
      backend: alertPolicy.backend,
      enabled: alertPolicy.enabled,
      config: alertPolicy.config,
    },
  }
}
