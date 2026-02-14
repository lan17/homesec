import { isAPIError } from '../../api/client'
import type { ClipResponse } from '../../api/generated/types'

export function formatTimestamp(value: string): string {
  const date = new Date(value)
  if (Number.isNaN(date.valueOf())) {
    return value
  }
  return date.toLocaleString()
}

export function describeClipError(error: unknown): string {
  if (isAPIError(error)) {
    return error.errorCode ? `${error.message} (${error.errorCode})` : error.message
  }
  if (error instanceof Error) {
    return error.message
  }
  return 'Unknown error'
}

export function renderDetectedObjects(clip: ClipResponse): string {
  const detectedObjects = clip.detected_objects ?? []
  if (detectedObjects.length === 0) {
    return 'None'
  }
  return detectedObjects.join(', ')
}
