import type {
  CameraListResponse,
  CameraResponse,
  ClipListResponse,
  ClipResponse,
  ConfigChangeResponse,
  PreviewSessionResponse,
  PreviewState,
  PreviewStatusResponse,
  PreviewStopResponse,
  DeviceInfoResponse,
  DiscoveredCameraResponse,
  DiagnosticsResponse,
  FinalizeResponse,
  HealthResponse,
  MediaProfileResponse,
  PreflightCheckResponse,
  PreflightResponse,
  TestConnectionResponse,
  ProbeResponse,
  PostgresBackupRunResponse,
  PostgresBackupStatusResponse,
  RuntimeReloadResponse,
  RuntimeState,
  RuntimeStatusResponse,
  SetupStatusResponse,
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

export function parseDiscoveredCameraResponse(payload: unknown): DiscoveredCameraResponse {
  if (!isJsonObject(payload)) {
    throw new Error('Discovered camera response is not a JSON object')
  }

  return {
    ip: expectString(payload.ip, 'ip'),
    xaddr: expectString(payload.xaddr, 'xaddr'),
    scopes: expectStringArray(payload.scopes, 'scopes'),
    types: expectStringArray(payload.types, 'types'),
  }
}

export function parseOnvifDiscoverResponse(payload: unknown): DiscoveredCameraResponse[] {
  if (!Array.isArray(payload)) {
    throw new Error('ONVIF discover response must be an array')
  }

  return payload.map((camera, index) => {
    try {
      return parseDiscoveredCameraResponse(camera)
    } catch (error) {
      throw new Error(`discovered_cameras[${index}] invalid: ${(error as Error).message}`)
    }
  })
}

function parseOnvifMediaProfileResponse(payload: unknown): MediaProfileResponse {
  if (!isJsonObject(payload)) {
    throw new Error('Media profile response is not a JSON object')
  }

  return {
    token: expectString(payload.token, 'token'),
    name: expectString(payload.name, 'name'),
    video_encoding: expectNullableString(payload.video_encoding, 'video_encoding'),
    width: expectNullableNumber(payload.width, 'width'),
    height: expectNullableNumber(payload.height, 'height'),
    frame_rate_limit: expectNullableNumber(payload.frame_rate_limit, 'frame_rate_limit'),
    bitrate_limit_kbps: expectNullableNumber(payload.bitrate_limit_kbps, 'bitrate_limit_kbps'),
    stream_uri: expectNullableString(payload.stream_uri, 'stream_uri'),
    stream_error: expectNullableString(payload.stream_error, 'stream_error'),
  }
}

function parseOnvifDeviceInfoResponse(payload: unknown): DeviceInfoResponse {
  if (!isJsonObject(payload)) {
    throw new Error('Device info response is not a JSON object')
  }

  return {
    manufacturer: expectString(payload.manufacturer, 'manufacturer'),
    model: expectString(payload.model, 'model'),
    firmware_version: expectString(payload.firmware_version, 'firmware_version'),
    serial_number: expectString(payload.serial_number, 'serial_number'),
    hardware_id: expectString(payload.hardware_id, 'hardware_id'),
  }
}

export function parseOnvifProbeResponse(payload: unknown): ProbeResponse {
  if (!isJsonObject(payload)) {
    throw new Error('ONVIF probe response is not a JSON object')
  }

  const rawProfiles = payload.profiles
  if (!Array.isArray(rawProfiles)) {
    throw new Error('profiles must be an array')
  }

  return {
    device: parseOnvifDeviceInfoResponse(payload.device),
    profiles: rawProfiles.map((profile, index) => {
      try {
        return parseOnvifMediaProfileResponse(profile)
      } catch (error) {
        throw new Error(`profiles[${index}] invalid: ${(error as Error).message}`)
      }
    }),
  }
}

export function parseConfigChangeResponse(payload: unknown): ConfigChangeResponse {
  if (!isJsonObject(payload)) {
    throw new Error('Config change response is not a JSON object')
  }

  const camera = payload.camera
  const runtimeReload = payload.runtime_reload
  return {
    restart_required: expectBoolean(payload.restart_required, 'restart_required'),
    camera: camera === null || camera === undefined ? null : parseCameraResponse(payload.camera),
    runtime_reload:
      runtimeReload === null || runtimeReload === undefined
        ? null
        : parseRuntimeReloadResponse(runtimeReload),
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
    bootstrap_mode: expectBoolean(payload.bootstrap_mode, 'bootstrap_mode'),
  }
}

function parseSetupState(value: unknown, fieldName: string): SetupStatusResponse['state'] {
  if (value === 'fresh' || value === 'partial' || value === 'complete') {
    return value
  }
  throw new Error(`${fieldName} must be one of fresh|partial|complete`)
}

export function parseSetupStatusResponse(payload: unknown): SetupStatusResponse {
  if (!isJsonObject(payload)) {
    throw new Error('Setup status response is not a JSON object')
  }

  return {
    state: parseSetupState(payload.state, 'state'),
    has_cameras: expectBoolean(payload.has_cameras, 'has_cameras'),
    pipeline_running: expectBoolean(payload.pipeline_running, 'pipeline_running'),
    auth_configured: expectBoolean(payload.auth_configured, 'auth_configured'),
  }
}

export function parseFinalizeResponse(payload: unknown): FinalizeResponse {
  if (!isJsonObject(payload)) {
    throw new Error('Finalize response is not a JSON object')
  }

  const defaultsApplied = payload.defaults_applied
  if (!Array.isArray(defaultsApplied)) {
    throw new Error('defaults_applied must be an array')
  }
  const errors = payload.errors
  if (!Array.isArray(errors)) {
    throw new Error('errors must be an array')
  }

  return {
    success: expectBoolean(payload.success, 'success'),
    config_path: expectString(payload.config_path, 'config_path'),
    restart_requested: expectBoolean(payload.restart_requested, 'restart_requested'),
    defaults_applied: defaultsApplied.map((value, index) =>
      expectString(value, `defaults_applied[${index}]`),
    ),
    errors: errors.map((value, index) => expectString(value, `errors[${index}]`)),
  }
}

function parsePreflightCheckResponse(payload: unknown): PreflightCheckResponse {
  if (!isJsonObject(payload)) {
    throw new Error('Preflight check response is not a JSON object')
  }

  return {
    name: expectString(payload.name, 'name'),
    passed: expectBoolean(payload.passed, 'passed'),
    message: expectString(payload.message, 'message'),
    latency_ms: expectNullableNumber(payload.latency_ms, 'latency_ms'),
  }
}

export function parsePreflightResponse(payload: unknown): PreflightResponse {
  if (!isJsonObject(payload)) {
    throw new Error('Preflight response is not a JSON object')
  }

  const checks = payload.checks
  if (!Array.isArray(checks)) {
    throw new Error('checks must be an array')
  }

  return {
    all_passed: expectBoolean(payload.all_passed, 'all_passed'),
    checks: checks.map((check, index) => {
      try {
        return parsePreflightCheckResponse(check)
      } catch (error) {
        throw new Error(`checks[${index}] invalid: ${(error as Error).message}`)
      }
    }),
  }
}

export function parseTestConnectionResponse(payload: unknown): TestConnectionResponse {
  if (!isJsonObject(payload)) {
    throw new Error('Test-connection response is not a JSON object')
  }

  const details = payload.details
  if (details !== null && details !== undefined && !isJsonObject(details)) {
    throw new Error('details must be an object when provided')
  }

  return {
    success: expectBoolean(payload.success, 'success'),
    message: expectString(payload.message, 'message'),
    latency_ms: expectNullableNumber(payload.latency_ms, 'latency_ms'),
    details: (details ?? null) as TestConnectionResponse['details'],
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

function parsePreviewState(value: unknown, fieldName: string): PreviewState {
  if (
    value === 'idle'
    || value === 'starting'
    || value === 'ready'
    || value === 'degraded'
    || value === 'stopping'
    || value === 'error'
  ) {
    return value
  }
  throw new Error(`${fieldName} must be one of idle|starting|ready|degraded|stopping|error`)
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

export function parsePostgresBackupStatusResponse(
  payload: unknown,
): PostgresBackupStatusResponse {
  if (!isJsonObject(payload)) {
    throw new Error('Postgres backup status response is not a JSON object')
  }

  return {
    enabled: expectBoolean(payload.enabled, 'enabled'),
    running: expectBoolean(payload.running, 'running'),
    available: expectBoolean(payload.available, 'available'),
    unavailable_reason: expectNullableString(payload.unavailable_reason, 'unavailable_reason'),
    last_attempted_at: expectNullableString(payload.last_attempted_at, 'last_attempted_at'),
    last_success_at: expectNullableString(payload.last_success_at, 'last_success_at'),
    last_error: expectNullableString(payload.last_error, 'last_error'),
    last_local_path: expectNullableString(payload.last_local_path, 'last_local_path'),
    last_uploaded_uri: expectNullableString(payload.last_uploaded_uri, 'last_uploaded_uri'),
    next_run_at: expectNullableString(payload.next_run_at, 'next_run_at'),
    pending_remote_delete_count: expectNumber(
      payload.pending_remote_delete_count,
      'pending_remote_delete_count',
    ),
  }
}

export function parsePostgresBackupRunResponse(payload: unknown): PostgresBackupRunResponse {
  if (!isJsonObject(payload)) {
    throw new Error('Postgres backup run response is not a JSON object')
  }

  return {
    accepted: expectBoolean(payload.accepted, 'accepted'),
    message: expectString(payload.message, 'message'),
  }
}

export function parsePreviewStatusResponse(payload: unknown): PreviewStatusResponse {
  if (!isJsonObject(payload)) {
    throw new Error('Preview status response is not a JSON object')
  }

  return {
    camera_name: expectString(payload.camera_name, 'camera_name'),
    enabled: expectBoolean(payload.enabled, 'enabled'),
    state: parsePreviewState(payload.state, 'state'),
    viewer_count: expectNullableNumber(payload.viewer_count, 'viewer_count'),
    degraded_reason: expectNullableString(payload.degraded_reason, 'degraded_reason'),
    last_error: expectNullableString(payload.last_error, 'last_error'),
    idle_shutdown_at: expectNullableNumber(payload.idle_shutdown_at, 'idle_shutdown_at'),
  }
}

export function parsePreviewSessionResponse(payload: unknown): PreviewSessionResponse {
  if (!isJsonObject(payload)) {
    throw new Error('Preview session response is not a JSON object')
  }

  return {
    camera_name: expectString(payload.camera_name, 'camera_name'),
    state: parsePreviewState(payload.state, 'state'),
    viewer_count: expectNullableNumber(payload.viewer_count, 'viewer_count'),
    token: expectNullableString(payload.token, 'token'),
    token_expires_at: expectNullableString(payload.token_expires_at, 'token_expires_at'),
    playlist_url: expectString(payload.playlist_url, 'playlist_url'),
    idle_timeout_s: expectNumber(payload.idle_timeout_s, 'idle_timeout_s'),
    warning: expectNullableString(payload.warning, 'warning'),
  }
}

export function parsePreviewStopResponse(payload: unknown): PreviewStopResponse {
  if (!isJsonObject(payload)) {
    throw new Error('Preview stop response is not a JSON object')
  }

  return {
    accepted: expectBoolean(payload.accepted, 'accepted'),
    state: parsePreviewState(payload.state, 'state'),
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
