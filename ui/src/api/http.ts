import type { ApiRequestOptions } from './generated/client'

import { APIError, extractAPIErrorEnvelope } from './errors'
import {
  browserAuthTokenProvider,
  resolveAuthToken,
  type AuthTokenProvider,
} from './tokenProvider'
import type { ClientServerBaseUrlProvider } from './serverBaseUrlProvider'

type QueryValue = string | number | boolean | null | undefined

export interface RequestJsonOptions extends ApiRequestOptions {
  allowStatuses?: number[]
  query?: Record<string, QueryValue>
  method?: 'DELETE' | 'GET' | 'PATCH' | 'POST' | 'PUT'
  body?: unknown
}

export interface JsonResponse {
  status: number
  payload: unknown
}

export interface JsonHttpClientOptions {
  authTokenProvider?: AuthTokenProvider
  serverBaseUrlProvider?: ClientServerBaseUrlProvider
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
  private readonly authTokenProvider: AuthTokenProvider
  private readonly serverBaseUrlProvider: ClientServerBaseUrlProvider | undefined

  constructor(baseUrl: string, options: JsonHttpClientOptions = {}) {
    this.baseUrl = baseUrl
    this.authTokenProvider = options.authTokenProvider ?? browserAuthTokenProvider
    this.serverBaseUrlProvider = options.serverBaseUrlProvider
  }

  private resolveBaseUrlSync(): string {
    return this.serverBaseUrlProvider?.getBaseUrlSync() ?? this.baseUrl
  }

  private async resolveBaseUrl(): Promise<string> {
    if (!this.serverBaseUrlProvider) {
      return this.baseUrl
    }

    return (await this.serverBaseUrlProvider.getBaseUrl()) ?? this.baseUrl
  }

  resolvePath(path: string): string {
    return joinUrl(this.resolveBaseUrlSync(), path)
  }

  async requestJson(
    path: string,
    { signal, apiKey, allowStatuses = [], query, method = 'GET', body }: RequestJsonOptions,
  ): Promise<JsonResponse> {
    const hasJsonBody = body !== undefined
    const resolvedApiKey = await resolveAuthToken(apiKey, this.authTokenProvider)
    const resolvedBaseUrl = await this.resolveBaseUrl()
    const response = await fetch(joinUrl(resolvedBaseUrl, withQueryString(path, query)), {
      method,
      headers: buildHeaders(resolvedApiKey, hasJsonBody),
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
