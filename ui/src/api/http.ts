import type { ApiRequestOptions } from './generated/client'

import { resolveApiKey } from './apiKeyStorage'
import { APIError, extractAPIErrorEnvelope } from './errors'

type QueryValue = string | number | boolean | null | undefined

export interface RequestJsonOptions extends ApiRequestOptions {
  allowStatuses?: number[]
  query?: Record<string, QueryValue>
  method?: 'DELETE' | 'GET' | 'POST' | 'PUT'
  body?: unknown
}

export interface JsonResponse {
  status: number
  payload: unknown
}

function joinUrl(baseUrl: string, path: string): string {
  if (!baseUrl) {
    return path
  }

  const normalizedBase = baseUrl.replace(/\/+$/, '')
  return `${normalizedBase}${path}`
}

function withQueryString(path: string, query: Record<string, QueryValue> | undefined): string {
  if (!query) {
    return path
  }

  const params = new URLSearchParams()
  for (const [key, value] of Object.entries(query)) {
    if (value === null || value === undefined || value === '') {
      continue
    }
    params.set(key, String(value))
  }

  const queryString = params.toString()
  if (!queryString) {
    return path
  }
  return `${path}?${queryString}`
}

function buildHeaders(apiKey: string | null, hasJsonBody: boolean): HeadersInit {
  const headers: Record<string, string> = {
    Accept: 'application/json',
  }

  if (hasJsonBody) {
    headers['content-type'] = 'application/json'
  }

  if (apiKey && apiKey.trim().length > 0) {
    headers.Authorization = `Bearer ${apiKey}`
  }
  return headers
}

async function parseResponsePayload(response: Response): Promise<unknown> {
  const contentType = response.headers.get('content-type')?.toLowerCase() ?? ''
  if (contentType.includes('application/json')) {
    return response.json()
  }

  return response.text()
}

export class JsonHttpClient {
  private readonly baseUrl: string

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl
  }

  resolvePath(path: string): string {
    return joinUrl(this.baseUrl, path)
  }

  async requestJson(
    path: string,
    { signal, apiKey, allowStatuses = [], query, method = 'GET', body }: RequestJsonOptions,
  ): Promise<JsonResponse> {
    const hasJsonBody = body !== undefined
    const response = await fetch(joinUrl(this.baseUrl, withQueryString(path, query)), {
      method,
      headers: buildHeaders(resolveApiKey(apiKey), hasJsonBody),
      signal,
      body: hasJsonBody ? JSON.stringify(body) : undefined,
    })

    const payload = await parseResponsePayload(response)
    if (!response.ok && !allowStatuses.includes(response.status)) {
      const errorEnvelope = extractAPIErrorEnvelope(payload)
      throw new APIError(errorEnvelope.detail, response.status, payload, errorEnvelope.errorCode)
    }

    return {
      status: response.status,
      payload,
    }
  }
}
