import type {
  CameraListResponse,
  CameraResponse,
  ClipListResponse,
  ClipResponse,
  ConfigChangeResponse,
  DiagnosticsResponse,
  HealthResponse,
  RuntimeReloadResponse,
  RuntimeState,
  RuntimeStatusResponse,
  StatsResponse,
} from './generated/types'

type JsonObject = Record<string, unknown>

export interface ClipMediaTokenResponsePayload {
  media_url: string
  tokenized: boolean
  expires_at: string | null
}

export type ApiSnapshot<TPayload extends object> = TPayload & { httpStatus: number }

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

export function parseCameraResponse(payload: unknown): CameraResponse {
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

export function parseCameraListResponse(payload: unknown): CameraListResponse {
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

export function parseConfigChangeResponse(payload: unknown): ConfigChangeResponse {
  if (!isJsonObject(payload)) {
    throw new Error('Config change response is not a JSON object')
  }

  const camera = payload.camera
  return {
    restart_required: expectBoolean(payload.restart_required, 'restart_required'),
    camera: camera === null || camera === undefined ? null : parseCameraResponse(payload.camera),
  }
}

export function parseHealthResponse(payload: unknown): HealthResponse {
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

export function parseStatsResponse(payload: unknown): StatsResponse {
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

export function parseDiagnosticsResponse(payload: unknown): DiagnosticsResponse {
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

function parseRuntimeState(value: unknown, fieldName: string): RuntimeState {
  if (value === 'idle' || value === 'reloading' || value === 'failed') {
    return value
  }
  throw new Error(`${fieldName} must be one of idle|reloading|failed`)
}

export function parseRuntimeReloadResponse(payload: unknown): RuntimeReloadResponse {
  if (!isJsonObject(payload)) {
    throw new Error('Runtime reload response is not a JSON object')
  }

  return {
    accepted: expectBoolean(payload.accepted, 'accepted'),
    message: expectString(payload.message, 'message'),
    target_generation: expectNumber(payload.target_generation, 'target_generation'),
  }
}

export function parseRuntimeStatusResponse(payload: unknown): RuntimeStatusResponse {
  if (!isJsonObject(payload)) {
    throw new Error('Runtime status response is not a JSON object')
  }

  return {
    state: parseRuntimeState(payload.state, 'state'),
    generation: expectNumber(payload.generation, 'generation'),
    reload_in_progress: expectBoolean(payload.reload_in_progress, 'reload_in_progress'),
    active_config_version: expectNullableString(payload.active_config_version, 'active_config_version'),
    last_reload_at: expectNullableString(payload.last_reload_at, 'last_reload_at'),
    last_reload_error: expectNullableString(payload.last_reload_error, 'last_reload_error'),
  }
}

export function parseClipResponse(payload: unknown): ClipResponse {
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
    alerted: payload.alerted === undefined ? false : expectBoolean(payload.alerted, 'alerted'),
  }
}

export function parseClipListResponse(payload: unknown): ClipListResponse {
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

export function parseClipMediaTokenResponse(payload: unknown): ClipMediaTokenResponsePayload {
  if (!isJsonObject(payload)) {
    throw new Error('Clip media token response is not a JSON object')
  }

  return {
    media_url: expectString(payload.media_url, 'media_url'),
    tokenized: expectBoolean(payload.tokenized, 'tokenized'),
    expires_at: expectNullableString(payload.expires_at, 'expires_at'),
  }
}

export function withHttpStatus<TPayload extends object>(
  payload: TPayload,
  status: number,
): ApiSnapshot<TPayload> {
  return {
    ...payload,
    httpStatus: status,
  }
}
