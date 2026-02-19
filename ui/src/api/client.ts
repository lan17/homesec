import type {
  ApiRequestOptions,
  CameraMutationOptions,
  GeneratedHomeSecClient,
} from './generated/client'
import type {
  CameraCreate,
  CameraListResponse,
  CameraResponse,
  CameraUpdate,
  ClipListResponse,
  ClipResponse,
  ConfigChangeResponse,
  DiagnosticsResponse,
  HealthResponse,
  ListClipsQuery,
  RuntimeReloadResponse,
  RuntimeStatusResponse,
  StatsResponse,
} from './generated/types'

import { JsonHttpClient } from './http'
import type { ApiSnapshot, ClipMediaTokenResponsePayload } from './parsing'
import {
  parseCameraListResponse,
  parseCameraResponse,
  parseClipListResponse,
  parseClipMediaTokenResponse,
  parseClipResponse,
  parseConfigChangeResponse,
  parseDiagnosticsResponse,
  parseHealthResponse,
  parseRuntimeReloadResponse,
  parseRuntimeStatusResponse,
  parseStatsResponse,
  withHttpStatus,
} from './parsing'
import { APIError } from './errors'

const DEFAULT_API_BASE_URL = ''

export type HealthSnapshot = ApiSnapshot<HealthResponse>
export type StatsSnapshot = ApiSnapshot<StatsResponse>
export type DiagnosticsSnapshot = ApiSnapshot<DiagnosticsResponse>
export type ClipListSnapshot = ApiSnapshot<ClipListResponse>
export type ClipSnapshot = ApiSnapshot<ClipResponse>
export type ConfigChangeSnapshot = ApiSnapshot<ConfigChangeResponse>
export type RuntimeReloadSnapshot = ApiSnapshot<RuntimeReloadResponse>
export type RuntimeStatusSnapshot = ApiSnapshot<RuntimeStatusResponse>
export type ClipMediaTokenSnapshot = ApiSnapshot<ClipMediaTokenResponsePayload>

export class HomeSecApiClient implements GeneratedHomeSecClient {
  private readonly httpClient: JsonHttpClient

  constructor(baseUrl = DEFAULT_API_BASE_URL) {
    this.httpClient = new JsonHttpClient(baseUrl)
  }

  async getCameras(options: ApiRequestOptions = {}): Promise<CameraListResponse> {
    const { status, payload } = await this.httpClient.requestJson('/api/v1/cameras', options)

    try {
      return parseCameraListResponse(payload)
    } catch {
      throw new APIError('Invalid cameras response payload', status, payload, null)
    }
  }

  async getCamera(name: string, options: ApiRequestOptions = {}): Promise<CameraResponse> {
    const { status, payload } = await this.httpClient.requestJson(
      `/api/v1/cameras/${encodeURIComponent(name)}`,
      options,
    )

    try {
      return parseCameraResponse(payload)
    } catch {
      throw new APIError('Invalid camera response payload', status, payload, null)
    }
  }

  async createCamera(
    payload: CameraCreate,
    options: CameraMutationOptions = {},
  ): Promise<ConfigChangeSnapshot> {
    const { applyChanges = false, ...requestOptions } = options
    const response = await this.httpClient.requestJson('/api/v1/cameras', {
      ...requestOptions,
      query: applyChanges ? { apply_changes: true } : undefined,
      method: 'POST',
      body: payload,
    })

    try {
      return withHttpStatus(parseConfigChangeResponse(response.payload), response.status)
    } catch {
      throw new APIError(
        'Invalid create-camera response payload',
        response.status,
        response.payload,
        null,
      )
    }
  }

  async updateCamera(
    name: string,
    payload: CameraUpdate,
    options: CameraMutationOptions = {},
  ): Promise<ConfigChangeSnapshot> {
    const { applyChanges = false, ...requestOptions } = options
    const response = await this.httpClient.requestJson(`/api/v1/cameras/${encodeURIComponent(name)}`, {
      ...requestOptions,
      query: applyChanges ? { apply_changes: true } : undefined,
      method: 'PATCH',
      body: payload,
    })

    try {
      return withHttpStatus(parseConfigChangeResponse(response.payload), response.status)
    } catch {
      throw new APIError(
        'Invalid update-camera response payload',
        response.status,
        response.payload,
        null,
      )
    }
  }

  async deleteCamera(
    name: string,
    options: CameraMutationOptions = {},
  ): Promise<ConfigChangeSnapshot> {
    const { applyChanges = false, ...requestOptions } = options
    const response = await this.httpClient.requestJson(`/api/v1/cameras/${encodeURIComponent(name)}`, {
      ...requestOptions,
      query: applyChanges ? { apply_changes: true } : undefined,
      method: 'DELETE',
    })

    try {
      return withHttpStatus(parseConfigChangeResponse(response.payload), response.status)
    } catch {
      throw new APIError(
        'Invalid delete-camera response payload',
        response.status,
        response.payload,
        null,
      )
    }
  }

  async getHealth(options: ApiRequestOptions = {}): Promise<HealthSnapshot> {
    const { status, payload } = await this.httpClient.requestJson('/api/v1/health', {
      ...options,
      allowStatuses: [503],
    })

    try {
      return withHttpStatus(parseHealthResponse(payload), status)
    } catch {
      throw new APIError('Invalid health response payload', status, payload, null)
    }
  }

  async getStats(options: ApiRequestOptions = {}): Promise<StatsSnapshot> {
    const { status, payload } = await this.httpClient.requestJson('/api/v1/stats', options)

    try {
      return withHttpStatus(parseStatsResponse(payload), status)
    } catch {
      throw new APIError('Invalid stats response payload', status, payload, null)
    }
  }

  async getDiagnostics(options: ApiRequestOptions = {}): Promise<DiagnosticsSnapshot> {
    const { status, payload } = await this.httpClient.requestJson('/api/v1/diagnostics', options)

    try {
      return withHttpStatus(parseDiagnosticsResponse(payload), status)
    } catch {
      throw new APIError('Invalid diagnostics response payload', status, payload, null)
    }
  }

  async reloadRuntime(options: ApiRequestOptions = {}): Promise<RuntimeReloadSnapshot> {
    const response = await this.httpClient.requestJson('/api/v1/runtime/reload', {
      ...options,
      method: 'POST',
    })

    try {
      return withHttpStatus(parseRuntimeReloadResponse(response.payload), response.status)
    } catch {
      throw new APIError(
        'Invalid runtime reload response payload',
        response.status,
        response.payload,
        null,
      )
    }
  }

  async getRuntimeStatus(options: ApiRequestOptions = {}): Promise<RuntimeStatusSnapshot> {
    const response = await this.httpClient.requestJson('/api/v1/runtime/status', options)

    try {
      return withHttpStatus(parseRuntimeStatusResponse(response.payload), response.status)
    } catch {
      throw new APIError(
        'Invalid runtime status response payload',
        response.status,
        response.payload,
        null,
      )
    }
  }

  async getClips(
    query: ListClipsQuery | undefined = undefined,
    options: ApiRequestOptions = {},
  ): Promise<ClipListSnapshot> {
    const { status, payload } = await this.httpClient.requestJson('/api/v1/clips', {
      ...options,
      query: query ?? undefined,
    })

    try {
      return withHttpStatus(parseClipListResponse(payload), status)
    } catch {
      throw new APIError('Invalid clips list response payload', status, payload, null)
    }
  }

  async getClip(clipId: string, options: ApiRequestOptions = {}): Promise<ClipSnapshot> {
    const { status, payload } = await this.httpClient.requestJson(
      `/api/v1/clips/${encodeURIComponent(clipId)}`,
      options,
    )

    try {
      return withHttpStatus(parseClipResponse(payload), status)
    } catch {
      throw new APIError('Invalid clip response payload', status, payload, null)
    }
  }

  async createClipMediaToken(
    clipId: string,
    options: ApiRequestOptions = {},
  ): Promise<ClipMediaTokenSnapshot> {
    const { status, payload } = await this.httpClient.requestJson(
      `/api/v1/clips/${encodeURIComponent(clipId)}/media-token`,
      {
        ...options,
        method: 'POST',
      },
    )

    try {
      return withHttpStatus(parseClipMediaTokenResponse(payload), status)
    } catch {
      throw new APIError('Invalid clip media token response payload', status, payload, null)
    }
  }

  resolvePath(path: string): string {
    return this.httpClient.resolvePath(path)
  }
}

export const apiClient = new HomeSecApiClient(import.meta.env.VITE_API_BASE_URL ?? DEFAULT_API_BASE_URL)

export { APIError, isAPIError, isUnauthorizedAPIError } from './errors'
export { clearApiKey, getStoredApiKey, hasStoredApiKey, saveApiKey } from './apiKeyStorage'
