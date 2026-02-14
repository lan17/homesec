import type { ApiRequestOptions, GeneratedHomeSecClient } from './generated/client'
import type {
  CameraListResponse,
  CameraResponse,
  ClipListResponse,
  ClipResponse,
  DiagnosticsResponse,
  HealthResponse,
  ListClipsQuery,
  StatsResponse,
} from './generated/types'

const API_KEY_STORAGE_KEY = 'homesec.apiKey'
const DEFAULT_API_BASE_URL = ''

type JsonObject = Record<string, unknown>
type QueryValue = string | number | boolean | null | undefined

export interface HealthSnapshot extends HealthResponse {
  httpStatus: number
}

export interface StatsSnapshot extends StatsResponse {
  httpStatus: number
}

export interface DiagnosticsSnapshot extends DiagnosticsResponse {
  httpStatus: number
}

export interface ClipListSnapshot extends ClipListResponse {
  httpStatus: number
}

export interface ClipSnapshot extends ClipResponse {
  httpStatus: number
}

interface RequestJsonOptions extends ApiRequestOptions {
  allowStatuses?: number[]
  query?: Record<string, QueryValue>
}

interface APIErrorEnvelope {
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

function expectString(value: unknown, fieldName: string): string {
  if (typeof value !== 'string') {
    throw new Error(`${fieldName} must be a string`)
  }
  return value
}

function expectNumber(value: unknown, fieldName: string): number {
  if (typeof value !== 'number') {
    throw new Error(`${fieldName} must be a number`)
  }
  return value
}

function expectBoolean(value: unknown, fieldName: string): boolean {
  if (typeof value !== 'boolean') {
    throw new Error(`${fieldName} must be a boolean`)
  }
  return value
}

function expectNullableString(value: unknown, fieldName: string): string | null {
  if (value === null || value === undefined) {
    return null
  }
  return expectString(value, fieldName)
}

function expectNullableNumber(value: unknown, fieldName: string): number | null {
  if (value === null || value === undefined) {
    return null
  }
  return expectNumber(value, fieldName)
}

function expectStringArray(value: unknown, fieldName: string): string[] {
  if (!Array.isArray(value)) {
    throw new Error(`${fieldName} must be an array`)
  }
  const parsed: string[] = []
  for (let index = 0; index < value.length; index += 1) {
    const element = value[index]
    if (typeof element !== 'string') {
      throw new Error(`${fieldName}[${index}] must be a string`)
    }
    parsed.push(element)
  }
  return parsed
}

function parseCameraResponse(payload: unknown): CameraResponse {
  if (!isJsonObject(payload)) {
    throw new Error('Camera response is not a JSON object')
  }
  const sourceConfig = payload.source_config
  if (!isJsonObject(sourceConfig)) {
    throw new Error('source_config must be an object')
  }

  return {
    name: expectString(payload.name, 'name'),
    enabled: expectBoolean(payload.enabled, 'enabled'),
    source_backend: expectString(payload.source_backend, 'source_backend'),
    healthy: expectBoolean(payload.healthy, 'healthy'),
    last_heartbeat: expectNullableNumber(payload.last_heartbeat, 'last_heartbeat'),
    source_config: sourceConfig,
  }
}

function parseCameraListResponse(payload: unknown): CameraListResponse {
  if (!Array.isArray(payload)) {
    throw new Error('Camera list response must be an array')
  }

  return payload.map((camera, index) => {
    try {
      return parseCameraResponse(camera)
    } catch (error) {
      throw new Error(`cameras[${index}] invalid: ${(error as Error).message}`)
    }
  })
}

function parseHealthResponse(payload: unknown): HealthResponse {
  if (!isJsonObject(payload)) {
    throw new Error('Health response is not a JSON object')
  }

  return {
    status: expectString(payload.status, 'status'),
    pipeline: expectString(payload.pipeline, 'pipeline'),
    postgres: expectString(payload.postgres, 'postgres'),
    cameras_online: expectNumber(payload.cameras_online, 'cameras_online'),
  }
}

function parseStatsResponse(payload: unknown): StatsResponse {
  if (!isJsonObject(payload)) {
    throw new Error('Stats response is not a JSON object')
  }

  return {
    clips_today: expectNumber(payload.clips_today, 'clips_today'),
    alerts_today: expectNumber(payload.alerts_today, 'alerts_today'),
    cameras_total: expectNumber(payload.cameras_total, 'cameras_total'),
    cameras_online: expectNumber(payload.cameras_online, 'cameras_online'),
    uptime_seconds: expectNumber(payload.uptime_seconds, 'uptime_seconds'),
  }
}

function parseComponentStatus(payload: unknown, fieldName: string): DiagnosticsResponse['postgres'] {
  if (!isJsonObject(payload)) {
    throw new Error(`${fieldName} must be an object`)
  }

  return {
    status: expectString(payload.status, `${fieldName}.status`),
    error: expectNullableString(payload.error, `${fieldName}.error`),
    latency_ms: expectNullableNumber(payload.latency_ms, `${fieldName}.latency_ms`),
  }
}

function parseDiagnosticsResponse(payload: unknown): DiagnosticsResponse {
  if (!isJsonObject(payload)) {
    throw new Error('Diagnostics response is not a JSON object')
  }

  const rawCameras = payload.cameras
  if (!isJsonObject(rawCameras)) {
    throw new Error('cameras must be an object')
  }

  const cameras: DiagnosticsResponse['cameras'] = {}
  for (const [cameraName, rawStatus] of Object.entries(rawCameras)) {
    if (!isJsonObject(rawStatus)) {
      throw new Error(`cameras.${cameraName} must be an object`)
    }
    cameras[cameraName] = {
      healthy: expectBoolean(rawStatus.healthy, `cameras.${cameraName}.healthy`),
      enabled: expectBoolean(rawStatus.enabled, `cameras.${cameraName}.enabled`),
      last_heartbeat: expectNullableNumber(
        rawStatus.last_heartbeat,
        `cameras.${cameraName}.last_heartbeat`,
      ),
    }
  }

  return {
    status: expectString(payload.status, 'status'),
    uptime_seconds: expectNumber(payload.uptime_seconds, 'uptime_seconds'),
    postgres: parseComponentStatus(payload.postgres, 'postgres'),
    storage: parseComponentStatus(payload.storage, 'storage'),
    cameras,
  }
}

function parseClipResponse(payload: unknown): ClipResponse {
  if (!isJsonObject(payload)) {
    throw new Error('Clip response is not a JSON object')
  }

  return {
    id: expectString(payload.id, 'id'),
    camera: expectString(payload.camera, 'camera'),
    status: expectString(payload.status, 'status'),
    created_at: expectString(payload.created_at, 'created_at'),
    activity_type: expectNullableString(payload.activity_type, 'activity_type'),
    risk_level: expectNullableString(payload.risk_level, 'risk_level'),
    summary: expectNullableString(payload.summary, 'summary'),
    detected_objects:
      payload.detected_objects === undefined
        ? []
        : expectStringArray(payload.detected_objects, 'detected_objects'),
    storage_uri: expectNullableString(payload.storage_uri, 'storage_uri'),
    view_url: expectNullableString(payload.view_url, 'view_url'),
    alerted:
      payload.alerted === undefined ? false : expectBoolean(payload.alerted, 'alerted'),
  }
}

function parseClipListResponse(payload: unknown): ClipListResponse {
  if (!isJsonObject(payload)) {
    throw new Error('Clip list response is not a JSON object')
  }

  const rawClips = payload.clips
  if (!Array.isArray(rawClips)) {
    throw new Error('clips must be an array')
  }

  return {
    clips: rawClips.map((clip, index) => {
      try {
        return parseClipResponse(clip)
      } catch (error) {
        throw new Error(`clips[${index}] invalid: ${(error as Error).message}`)
      }
    }),
    limit: expectNumber(payload.limit, 'limit'),
    next_cursor: expectNullableString(payload.next_cursor, 'next_cursor'),
    has_more: expectBoolean(payload.has_more, 'has_more'),
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

function buildHeaders(apiKey: string | null): HeadersInit {
  const headers: Record<string, string> = {
    Accept: 'application/json',
  }

  if (apiKey && apiKey.trim().length > 0) {
    headers.Authorization = `Bearer ${apiKey}`
  }
  return headers
}

function extractAPIErrorEnvelope(payload: unknown): APIErrorEnvelope {
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

function asSnapshot<TPayload extends object>(
  payload: TPayload,
  status: number,
): TPayload & { httpStatus: number } {
  return {
    ...payload,
    httpStatus: status,
  }
}

export class HomeSecApiClient implements GeneratedHomeSecClient {
  private readonly baseUrl: string

  constructor(baseUrl = DEFAULT_API_BASE_URL) {
    this.baseUrl = baseUrl
  }

  async getCameras(options: ApiRequestOptions = {}): Promise<CameraListResponse> {
    const { status, payload } = await this.requestJson('/api/v1/cameras', options)

    try {
      return parseCameraListResponse(payload)
    } catch {
      throw new APIError('Invalid cameras response payload', status, payload, null)
    }
  }

  async getHealth(options: ApiRequestOptions = {}): Promise<HealthSnapshot> {
    const { status, payload } = await this.requestJson('/api/v1/health', {
      ...options,
      allowStatuses: [503],
    })

    try {
      return asSnapshot(parseHealthResponse(payload), status)
    } catch {
      throw new APIError('Invalid health response payload', status, payload, null)
    }
  }

  async getStats(options: ApiRequestOptions = {}): Promise<StatsSnapshot> {
    const { status, payload } = await this.requestJson('/api/v1/stats', options)

    try {
      return asSnapshot(parseStatsResponse(payload), status)
    } catch {
      throw new APIError('Invalid stats response payload', status, payload, null)
    }
  }

  async getDiagnostics(options: ApiRequestOptions = {}): Promise<DiagnosticsSnapshot> {
    const { status, payload } = await this.requestJson('/api/v1/diagnostics', options)

    try {
      return asSnapshot(parseDiagnosticsResponse(payload), status)
    } catch {
      throw new APIError('Invalid diagnostics response payload', status, payload, null)
    }
  }

  async getClips(
    query: ListClipsQuery | undefined = undefined,
    options: ApiRequestOptions = {},
  ): Promise<ClipListSnapshot> {
    const { status, payload } = await this.requestJson('/api/v1/clips', {
      ...options,
      query: query ?? undefined,
    })

    try {
      return asSnapshot(parseClipListResponse(payload), status)
    } catch {
      throw new APIError('Invalid clips list response payload', status, payload, null)
    }
  }

  async getClip(clipId: string, options: ApiRequestOptions = {}): Promise<ClipSnapshot> {
    const { status, payload } = await this.requestJson(
      `/api/v1/clips/${encodeURIComponent(clipId)}`,
      options,
    )

    try {
      return asSnapshot(parseClipResponse(payload), status)
    } catch {
      throw new APIError('Invalid clip response payload', status, payload, null)
    }
  }

  private async requestJson(
    path: string,
    { signal, apiKey, allowStatuses = [], query }: RequestJsonOptions,
  ): Promise<{ status: number; payload: unknown }> {
    const response = await fetch(joinUrl(this.baseUrl, withQueryString(path, query)), {
      method: 'GET',
      headers: buildHeaders(resolveApiKey(apiKey)),
      signal,
    })

    const payload = await this.parseResponsePayload(response)
    if (!response.ok && !allowStatuses.includes(response.status)) {
      const errorEnvelope = extractAPIErrorEnvelope(payload)
      throw new APIError(errorEnvelope.detail, response.status, payload, errorEnvelope.errorCode)
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
}

export const apiClient = new HomeSecApiClient(import.meta.env.VITE_API_BASE_URL ?? DEFAULT_API_BASE_URL)

export function isAPIError(error: unknown): error is APIError {
  return error instanceof APIError
}

export function isUnauthorizedAPIError(error: unknown): error is APIError {
  return error instanceof APIError && error.status === 401
}

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
