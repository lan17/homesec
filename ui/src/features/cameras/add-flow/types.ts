import type { CameraBackend } from '../forms'

export type CameraAddBackend = CameraBackend | 'onvif'

export interface CameraAddBackendOption {
  id: CameraAddBackend
  label: string
  description: string
}

export const CAMERA_ADD_BACKEND_OPTIONS: readonly CameraAddBackendOption[] = [
  {
    id: 'rtsp',
    label: 'RTSP',
    description: 'Direct stream URL with optional credentials.',
  },
  {
    id: 'ftp',
    label: 'FTP',
    description: 'Receive uploaded clips from camera FTP server mode.',
  },
  {
    id: 'local_folder',
    label: 'Local Folder',
    description: 'Watch a local directory for incoming clips.',
  },
  {
    id: 'onvif',
    label: 'ONVIF Discovery',
    description: 'Discover ONVIF cameras and generate RTSP config.',
  },
] as const
