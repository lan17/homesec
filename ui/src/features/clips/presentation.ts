import type { ClipResponse } from '../../api/generated/types'
import { describeUnknownError } from '../shared/errorPresentation'

export function formatTimestamp(value: string): string {
  const date = new Date(value)
  if (Number.isNaN(date.valueOf())) {
    return value
  }
  return date.toLocaleString()
}

function clipDate(value: string): Date | null {
  const date = new Date(value)
  return Number.isNaN(date.valueOf()) ? null : date
}

function localDateKey(date: Date): string {
  const year = date.getFullYear()
  const month = `${date.getMonth() + 1}`.padStart(2, '0')
  const day = `${date.getDate()}`.padStart(2, '0')
  return `${year}-${month}-${day}`
}

function startOfLocalDay(date: Date): Date {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate())
}

export function formatEventTime(value: string): string {
  const date = clipDate(value)
  if (!date) {
    return value
  }
  return date.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' })
}

export function formatEventDateGroup(value: string, now: Date = new Date()): string {
  const date = clipDate(value)
  if (!date) {
    return 'Unknown date'
  }

  const dayDiff = Math.round(
    (startOfLocalDay(now).getTime() - startOfLocalDay(date).getTime()) / 86_400_000,
  )
  if (dayDiff === 0) {
    return 'Today'
  }
  if (dayDiff === 1) {
    return 'Yesterday'
  }
  return date.toLocaleDateString(undefined, {
    weekday: 'long',
    month: 'short',
    day: 'numeric',
    year: now.getFullYear() === date.getFullYear() ? undefined : 'numeric',
  })
}

export function groupClipsByDate(clips: readonly ClipResponse[]): Array<{
  key: string
  label: string
  clips: ClipResponse[]
}> {
  const groups = new Map<string, { key: string; label: string; clips: ClipResponse[] }>()

  for (const clip of clips) {
    const date = clipDate(clip.created_at)
    const key = date ? localDateKey(date) : 'unknown-date'
    const label = formatEventDateGroup(clip.created_at)
    const group = groups.get(key)
    if (group) {
      group.clips.push(clip)
    } else {
      groups.set(key, { key, label, clips: [clip] })
    }
  }

  return [...groups.values()]
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

export function formatDetectedObjects(clip: ClipResponse): string {
  const detectedObjects = clip.detected_objects ?? []
  if (detectedObjects.length === 0) {
    return 'No objects detected'
  }
  return detectedObjects.join(', ')
}

export function formatActivityType(activityType: string | null | undefined): string {
  const normalized = activityType?.trim()
  if (!normalized) {
    return 'Activity unavailable'
  }
  return normalized
    .split(/[_\s-]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')
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
