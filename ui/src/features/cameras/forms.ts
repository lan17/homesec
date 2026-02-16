import type { CameraCreate } from '../../api/generated/types'

export const CAMERA_BACKEND_OPTIONS = ['rtsp', 'ftp', 'local_folder'] as const

export type CameraBackend = (typeof CAMERA_BACKEND_OPTIONS)[number]

const CAMERA_SOURCE_CONFIG_TEMPLATES: Record<CameraBackend, CameraCreate['source_config']> = {
  rtsp: {
    rtsp_url: 'rtsp://username:password@camera.local/stream',
    output_dir: './recordings',
  },
  ftp: {
    host: '0.0.0.0',
    port: 2121,
    root_dir: './ftp_incoming',
    anonymous: true,
  },
  local_folder: {
    watch_dir: './recordings',
    poll_interval: 1.0,
    stability_threshold_s: 3.0,
  },
}

export interface ParsedSourceConfig {
  ok: true
  value: CameraCreate['source_config']
}

export interface SourceConfigParseError {
  ok: false
  message: string
}

export type SourceConfigParseResult = ParsedSourceConfig | SourceConfigParseError

export function defaultSourceConfigForBackend(backend: CameraBackend): string {
  return JSON.stringify(CAMERA_SOURCE_CONFIG_TEMPLATES[backend], null, 2)
}

export function parseSourceConfigJson(rawValue: string): SourceConfigParseResult {
  const trimmed = rawValue.trim()
  if (!trimmed) {
    return {
      ok: false,
      message: 'Source config JSON is required.',
    }
  }

  let parsed: unknown
  try {
    parsed = JSON.parse(trimmed)
  } catch {
    return {
      ok: false,
      message: 'Source config must be valid JSON.',
    }
  }

  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return {
      ok: false,
      message: 'Source config must be a JSON object.',
    }
  }

  return {
    ok: true,
    value: parsed as CameraCreate['source_config'],
  }
}
