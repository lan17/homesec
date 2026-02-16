import type { APIError } from '../../api/client'

export type HealthTone = 'healthy' | 'degraded' | 'unhealthy' | 'unknown'

export function healthTone(status: string): HealthTone {
  if (status === 'healthy' || status === 'degraded' || status === 'unhealthy') {
    return status
  }
  return 'unknown'
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
  if (error.errorCode) {
    return `${error.message} (${error.errorCode})`
  }
  return error.message
}
