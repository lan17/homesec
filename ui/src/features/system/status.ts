import type { APIError } from '../../api/client'

import { describeAPIError as describeSharedAPIError } from '../shared/errorPresentation'

export type HealthTone = 'healthy' | 'degraded' | 'unhealthy' | 'unknown'

export function healthTone(status: string): HealthTone {
  if (status === 'healthy' || status === 'degraded' || status === 'unhealthy') {
    return status
  }
  return 'unknown'
}

export function formatHealthStatusLabel(status: string | null | undefined): string {
  switch (status) {
    case 'healthy':
      return 'Healthy'
    case 'degraded':
      return 'Degraded'
    case 'unhealthy':
      return 'Unhealthy'
    default:
      return 'Status unavailable'
  }
}

export function formatSystemValue(value: string | null | undefined): string {
  const normalized = value?.trim()
  if (!normalized) {
    return 'Status unavailable'
  }
  return normalized
    .split(/[_\s-]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')
}

export function formatUptime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) {
    return 'Status unavailable'
  }
  if (seconds < 60) {
    return `${Math.round(seconds)} sec`
  }
  const minutes = Math.round(seconds / 60)
  if (minutes < 60) {
    return `${minutes} min`
  }
  const hours = Math.round(minutes / 60)
  return `${hours} hr`
}

export function formatLastUpdated(dataUpdatedAt: number): string {
  if (dataUpdatedAt <= 0) {
    return 'Not yet updated'
  }

  return new Date(dataUpdatedAt).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

export function describeAPIError(error: APIError): string {
  return describeSharedAPIError(error)
}
