import type { ClipResponse } from '../../api/generated/types'
import { describeUnknownError } from '../shared/errorPresentation'

export function formatTimestamp(value: string): string {
  const date = new Date(value)
  if (Number.isNaN(date.valueOf())) {
    return value
  }
  return date.toLocaleString()
}

export function describeClipError(error: unknown): string {
  return describeUnknownError(error)
}

export function renderDetectedObjects(clip: ClipResponse): string {
  const detectedObjects = clip.detected_objects ?? []
  if (detectedObjects.length === 0) {
    return 'None'
  }
  return detectedObjects.join(', ')
}

function isHttpUrl(value: string | null | undefined): value is string {
  if (!value) {
    return false
  }
  return /^https?:\/\//i.test(value)
}

export function resolveClipViewLink(clip: ClipResponse): string | null {
  if (isHttpUrl(clip.view_url)) {
    return clip.view_url
  }
  return null
}

export function resolveClipExternalLink(clip: ClipResponse): string | null {
  const viewLink = resolveClipViewLink(clip)
  if (viewLink) {
    return viewLink
  }
  if (isHttpUrl(clip.storage_uri)) {
    return clip.storage_uri
  }
  return null
}
