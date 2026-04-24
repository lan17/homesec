import type { TestConnectionRequest } from '../../../../api/generated/types'
import type { CameraAddBackend } from '../types'
import { FtpForm } from './FtpForm'
import { LocalFolderForm } from './LocalFolderForm'
import { OnvifForm } from './OnvifForm'
import { RtspForm } from './RtspForm'
import type { BackendFormDef } from './types'

const RTSP_BACKEND: BackendFormDef = {
  id: 'rtsp',
  label: 'RTSP',
  description: 'Direct stream URL with optional credentials.',
  steps: [{ title: 'Configure RTSP source', component: RtspForm }],
  defaultConfig: {
    rtsp_url: 'rtsp://username:password@camera.local/stream',
    preview_stream: 'main',
    output_dir: './recordings',
  },
  suggestNamePrefix: 'rtsp',
  validateStep: (_stepIndex, config) => {
    const rtspUrl = config.rtsp_url
    if (typeof rtspUrl !== 'string' || rtspUrl.trim().length === 0) {
      return 'RTSP URL is required.'
    }
    return null
  },
  buildCameraSource: (config) => ({
    source_backend: 'rtsp',
    source_config: config,
  }),
  buildTestRequest: (config): TestConnectionRequest => ({
    type: 'camera',
    backend: 'rtsp',
    config,
  }),
}

const FTP_BACKEND: BackendFormDef = {
  id: 'ftp',
  label: 'FTP',
  description: 'Receive uploaded clips from camera FTP server mode.',
  steps: [{ title: 'Configure FTP source', component: FtpForm }],
  defaultConfig: {
    host: '0.0.0.0',
    port: 2121,
    root_dir: './ftp_incoming',
    anonymous: true,
  },
  suggestNamePrefix: 'ftp',
  validateStep: (_stepIndex, config) => {
    const host = config.host
    if (typeof host !== 'string' || host.trim().length === 0) {
      return 'FTP host is required.'
    }
    const port = config.port
    if (typeof port !== 'number' || !Number.isInteger(port) || port < 1 || port > 65535) {
      return 'FTP port must be an integer between 1 and 65535.'
    }
    return null
  },
  buildCameraSource: (config) => ({
    source_backend: 'ftp',
    source_config: config,
  }),
  buildTestRequest: (config): TestConnectionRequest => ({
    type: 'camera',
    backend: 'ftp',
    config,
  }),
}

const LOCAL_FOLDER_BACKEND: BackendFormDef = {
  id: 'local_folder',
  label: 'Local Folder',
  description: 'Watch a local directory for incoming clips.',
  steps: [{ title: 'Configure local folder source', component: LocalFolderForm }],
  defaultConfig: {
    watch_dir: './recordings',
    poll_interval: 1.0,
    stability_threshold_s: 3.0,
  },
  suggestNamePrefix: 'local_folder',
  validateStep: (_stepIndex, config) => {
    const watchDir = config.watch_dir
    if (typeof watchDir !== 'string' || watchDir.trim().length === 0) {
      return 'Watch directory is required.'
    }
    return null
  },
  buildCameraSource: (config) => ({
    source_backend: 'local_folder',
    source_config: config,
  }),
  buildTestRequest: (config): TestConnectionRequest => ({
    type: 'camera',
    backend: 'local_folder',
    config,
  }),
}

const ONVIF_BACKEND: BackendFormDef = {
  id: 'onvif',
  label: 'ONVIF Discovery',
  description: 'Discover ONVIF cameras and generate RTSP config.',
  steps: [{ title: 'Discover and select ONVIF stream', component: OnvifForm }],
  defaultConfig: {
    rtsp_url: '',
    preview_stream: 'main',
    output_dir: './recordings',
  },
  suggestNamePrefix: 'onvif',
  validateStep: (_stepIndex, config) => {
    const rtspUrl = config.rtsp_url
    if (typeof rtspUrl !== 'string' || rtspUrl.trim().length === 0) {
      return 'Complete ONVIF discovery and select a stream profile.'
    }
    return null
  },
  buildCameraSource: (config) => ({
    source_backend: 'rtsp',
    source_config: config,
  }),
  buildTestRequest: (config): TestConnectionRequest => ({
    type: 'camera',
    backend: 'rtsp',
    config,
  }),
}

export const CAMERA_BACKEND_ORDER: readonly CameraAddBackend[] = [
  'rtsp',
  'ftp',
  'local_folder',
  'onvif',
] as const

export const CAMERA_ADD_BACKENDS: Record<CameraAddBackend, BackendFormDef> = {
  rtsp: RTSP_BACKEND,
  ftp: FTP_BACKEND,
  local_folder: LOCAL_FOLDER_BACKEND,
  onvif: ONVIF_BACKEND,
}

export function suggestCameraName(prefix: string, existingNames: readonly string[]): string {
  const used = new Set(existingNames.map((name) => name.trim().toLowerCase()))
  let index = 1
  while (used.has(`${prefix}_${index}`.toLowerCase())) {
    index += 1
  }
  return `${prefix}_${index}`
}
