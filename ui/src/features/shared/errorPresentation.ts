import { isAPIError } from '../../api/client'
import type { APIError } from '../../api/client'

export function describeAPIError(error: APIError): string {
  if (error.errorCode) {
    return `${error.message} (${error.errorCode})`
  }
  return error.message
}

export function describeUnknownError(error: unknown): string {
  if (isAPIError(error)) {
    return describeAPIError(error)
  }
  if (error instanceof Error) {
    return error.message
  }
  return 'Unknown error'
}
