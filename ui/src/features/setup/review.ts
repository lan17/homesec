import type { CameraCreate, FinalizeRequest } from '../../api/generated/types'
import type { DetectionStepData } from '../settings/detection/types'
import type { NotificationStepData } from '../settings/notifiers/types'
import type { StorageFormState } from '../settings/storage/types'

export type ReviewSectionStepId = 'camera' | 'storage' | 'detection' | 'notifications'
export type ReviewSectionStatus = 'configured' | 'skipped' | 'defaults'

export interface ReviewSummaryItem {
  label: string
  value: string
}

export interface ReviewSectionSummary {
  stepId: ReviewSectionStepId
  title: string
  status: ReviewSectionStatus
  items: ReviewSummaryItem[]
  emptyMessage: string
}

export interface ReviewWizardDrafts {
  camera: CameraCreate | null
  storage: StorageFormState | null
  detection: DetectionStepData | null
  notifications: NotificationStepData | null
}

function sectionStatus({
  hasData,
  stepId,
  skippedSteps,
}: {
  hasData: boolean
  stepId: ReviewSectionStepId
  skippedSteps: ReadonlySet<string>
}): ReviewSectionStatus {
  if (hasData) {
    return 'configured'
  }
  if (skippedSteps.has(stepId)) {
    return 'skipped'
  }
  return 'defaults'
}

function valueOrFallback(value: unknown, fallback: string): string {
  if (typeof value !== 'string') {
    return fallback
  }
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : fallback
}

function describeStorageRoot(config: Record<string, unknown>): string {
  const root = config.root
  if (typeof root === 'string' && root.trim().length > 0) {
    return root
  }
  const uploads = config.uploads_dir
  if (typeof uploads === 'string' && uploads.trim().length > 0) {
    return uploads
  }
  return 'n/a'
}

function summarizeCamera(camera: CameraCreate): ReviewSummaryItem[] {
  return [
    { label: 'Name', value: camera.name },
    { label: 'Backend', value: camera.source_backend },
    { label: 'Enabled', value: camera.enabled ? 'Yes' : 'No' },
  ]
}

function summarizeStorage(storage: StorageFormState): ReviewSummaryItem[] {
  return [
    { label: 'Backend', value: storage.backend },
    { label: 'Root', value: describeStorageRoot(storage.config) },
  ]
}

function summarizeDetection(detection: DetectionStepData): ReviewSummaryItem[] {
  const classes = detection.filter.config.classes
  const classCount = Array.isArray(classes) ? classes.length : 0
  return [
    { label: 'Filter backend', value: detection.filter.backend },
    { label: 'Filter classes', value: classCount > 0 ? String(classCount) : '0' },
    {
      label: 'Analyzer',
      value: detection.vlm ? `${detection.vlm.backend} (${detection.vlm.run_mode})` : 'Disabled',
    },
  ]
}

function summarizeNotifications(notifications: NotificationStepData): ReviewSummaryItem[] {
  const enabledBackends = notifications.notifiers
    .filter((entry) => entry.enabled)
    .map((entry) => entry.backend)
  const minRisk = valueOrFallback(
    notifications.alert_policy.config.min_risk_level,
    'high',
  )

  return [
    {
      label: 'Enabled notifiers',
      value: enabledBackends.length > 0 ? enabledBackends.join(', ') : 'None',
    },
    { label: 'Min risk level', value: minRisk },
  ]
}

function toFinalizeCamera(camera: CameraCreate): NonNullable<FinalizeRequest['cameras']>[number] {
  return {
    name: camera.name,
    enabled: camera.enabled,
    source: {
      backend: camera.source_backend,
      config: camera.source_config,
    },
  }
}

export function buildFinalizeRequestFromDrafts(drafts: ReviewWizardDrafts): FinalizeRequest {
  const payload: FinalizeRequest = { validate_only: false }

  if (drafts.camera) {
    payload.cameras = [toFinalizeCamera(drafts.camera)]
  }
  if (drafts.storage) {
    payload.storage = {
      backend: drafts.storage.backend,
      config: drafts.storage.config,
    }
  }
  if (drafts.detection) {
    payload.filter = drafts.detection.filter
    payload.vlm = drafts.detection.vlm
  }
  if (drafts.notifications) {
    payload.notifiers = drafts.notifications.notifiers
    payload.alert_policy = drafts.notifications.alert_policy
  }

  return payload
}

export function buildReviewSectionSummaries(
  drafts: ReviewWizardDrafts,
  skippedSteps: ReadonlySet<string>,
): ReviewSectionSummary[] {
  return [
    {
      stepId: 'camera',
      title: 'Camera',
      status: sectionStatus({
        hasData: drafts.camera !== null,
        stepId: 'camera',
        skippedSteps,
      }),
      items: drafts.camera ? summarizeCamera(drafts.camera) : [],
      emptyMessage: 'Not configured. Existing config or defaults will be used.',
    },
    {
      stepId: 'storage',
      title: 'Storage',
      status: sectionStatus({
        hasData: drafts.storage !== null,
        stepId: 'storage',
        skippedSteps,
      }),
      items: drafts.storage ? summarizeStorage(drafts.storage) : [],
      emptyMessage: 'Not configured. Existing config or defaults will be used.',
    },
    {
      stepId: 'detection',
      title: 'Detection',
      status: sectionStatus({
        hasData: drafts.detection !== null,
        stepId: 'detection',
        skippedSteps,
      }),
      items: drafts.detection ? summarizeDetection(drafts.detection) : [],
      emptyMessage: 'Not configured. Existing config or defaults will be used.',
    },
    {
      stepId: 'notifications',
      title: 'Notifications',
      status: sectionStatus({
        hasData: drafts.notifications !== null,
        stepId: 'notifications',
        skippedSteps,
      }),
      items: drafts.notifications ? summarizeNotifications(drafts.notifications) : [],
      emptyMessage: 'Not configured. Existing config or defaults will be used.',
    },
  ]
}
