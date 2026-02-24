type JsonObject = Record<string, unknown>

export interface APIErrorEnvelope {
  detail: string
  errorCode: string | null
}

export class APIError extends Error {
  readonly status: number
  readonly payload: unknown
  readonly errorCode: string | null

  constructor(message: string, status: number, payload: unknown, errorCode: string | null) {
    super(message)
    this.name = 'APIError'
    this.status = status
    this.payload = payload
    this.errorCode = errorCode
  }
}

function isJsonObject(value: unknown): value is JsonObject {
  return typeof value === 'object' && value !== null
}

export function extractAPIErrorEnvelope(payload: unknown): APIErrorEnvelope {
  if (isJsonObject(payload)) {
    const detail = payload.detail
    const message = payload.message
    const errorCode = payload.error_code

    if (typeof detail === 'string' && detail.trim().length > 0) {
      return {
        detail,
        errorCode: typeof errorCode === 'string' && errorCode.length > 0 ? errorCode : null,
      }
    }

    if (typeof message === 'string' && message.trim().length > 0) {
      return {
        detail: message,
        errorCode: typeof errorCode === 'string' && errorCode.length > 0 ? errorCode : null,
      }
    }
  }

  return {
    detail: 'API request failed',
    errorCode: null,
  }
}

export function isAPIError(error: unknown): error is APIError {
  return error instanceof APIError
}

export function isUnauthorizedAPIError(error: unknown): error is APIError {
  return error instanceof APIError && error.status === 401
}
