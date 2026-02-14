import type { ApiRequestOptions, GeneratedHomeSecClient } from './generated/client'
import type { HealthResponse } from './generated/types'

const API_KEY_STORAGE_KEY = 'homesec.apiKey'
const DEFAULT_API_BASE_URL = ''

type JsonObject = Record<string, unknown>

export interface HealthSnapshot extends HealthResponse {
  httpStatus: number
}

interface RequestJsonOptions extends ApiRequestOptions {
  allowStatuses?: number[]
}

export class APIError extends Error {
  readonly status: number
  readonly payload: unknown

  constructor(message: string, status: number, payload: unknown) {
    super(message)
    this.name = 'APIError'
    this.status = status
    this.payload = payload
  }
}

function isJsonObject(value: unknown): value is JsonObject {
  return typeof value === 'object' && value !== null
}

function parseHealthResponse(payload: unknown): HealthResponse {
  if (!isJsonObject(payload)) {
    throw new Error('Health response is not a JSON object')
  }

  const { status, pipeline, postgres, cameras_online: camerasOnline } = payload
  if (
    typeof status !== 'string'
    || typeof pipeline !== 'string'
    || typeof postgres !== 'string'
    || typeof camerasOnline !== 'number'
  ) {
    throw new Error('Health response shape is invalid')
  }

  return {
    status,
    pipeline,
    postgres,
    cameras_online: camerasOnline,
  }
}

function resolveApiKey(explicitApiKey: string | null | undefined): string | null {
  if (explicitApiKey !== undefined) {
    return explicitApiKey
  }

  if (typeof window === 'undefined') {
    return null
  }

  return window.sessionStorage.getItem(API_KEY_STORAGE_KEY)
}

function joinUrl(baseUrl: string, path: string): string {
  if (!baseUrl) {
    return path
  }

  const normalizedBase = baseUrl.replace(/\/+$/, '')
  return `${normalizedBase}${path}`
}

function buildHeaders(apiKey: string | null): HeadersInit {
  const headers: Record<string, string> = {
    Accept: 'application/json',
  }

  if (apiKey && apiKey.trim().length > 0) {
    headers.Authorization = `Bearer ${apiKey}`
  }
  return headers
}

export class HomeSecApiClient implements GeneratedHomeSecClient {
  private readonly baseUrl: string

  constructor(baseUrl = DEFAULT_API_BASE_URL) {
    this.baseUrl = baseUrl
  }

  async getHealth(options: ApiRequestOptions = {}): Promise<HealthSnapshot> {
    const { status, payload } = await this.requestJson('/api/v1/health', {
      ...options,
      allowStatuses: [503],
    })

    let parsed: HealthResponse
    try {
      parsed = parseHealthResponse(payload)
    } catch {
      throw new APIError('Invalid health response payload', status, payload)
    }

    return {
      ...parsed,
      httpStatus: status,
    }
  }

  private async requestJson(
    path: string,
    { signal, apiKey, allowStatuses = [] }: RequestJsonOptions,
  ): Promise<{ status: number; payload: unknown }> {
    const response = await fetch(joinUrl(this.baseUrl, path), {
      method: 'GET',
      headers: buildHeaders(resolveApiKey(apiKey)),
      signal,
    })

    const payload = await this.parseResponsePayload(response)
    if (!response.ok && !allowStatuses.includes(response.status)) {
      const message = this.extractErrorMessage(payload)
      throw new APIError(message, response.status, payload)
    }

    return {
      status: response.status,
      payload,
    }
  }

  private async parseResponsePayload(response: Response): Promise<unknown> {
    const contentType = response.headers.get('content-type')?.toLowerCase() ?? ''
    if (contentType.includes('application/json')) {
      return response.json()
    }

    return response.text()
  }

  private extractErrorMessage(payload: unknown): string {
    if (isJsonObject(payload)) {
      const message = payload.message
      if (typeof message === 'string' && message.trim().length > 0) {
        return message
      }

      const detail = payload.detail
      if (typeof detail === 'string' && detail.trim().length > 0) {
        return detail
      }
    }

    return 'API request failed'
  }
}

export const apiClient = new HomeSecApiClient(import.meta.env.VITE_API_BASE_URL ?? DEFAULT_API_BASE_URL)

export function saveApiKey(apiKey: string): void {
  if (typeof window === 'undefined') {
    return
  }

  window.sessionStorage.setItem(API_KEY_STORAGE_KEY, apiKey)
}

export function clearApiKey(): void {
  if (typeof window === 'undefined') {
    return
  }

  window.sessionStorage.removeItem(API_KEY_STORAGE_KEY)
}
