import type { StorageBackend, StorageBackendDef } from '../types'
import { DropboxStorageForm } from './DropboxStorageForm'
import { LocalStorageForm } from './LocalStorageForm'

const LOCAL_BACKEND: StorageBackendDef = {
  id: 'local',
  label: 'Local',
  description: 'Store clips on local disk.',
  defaultConfig: {
    root: './storage',
  },
  validate: (config) => {
    const root = config.root
    if (typeof root !== 'string' || root.trim().length === 0) {
      return 'Storage root directory is required.'
    }
    return null
  },
  component: LocalStorageForm,
}

const DROPBOX_BACKEND: StorageBackendDef = {
  id: 'dropbox',
  label: 'Dropbox',
  description: 'Upload clips to Dropbox storage.',
  defaultConfig: {
    root: '/homesec',
    token_env: 'DROPBOX_TOKEN',
  },
  validate: (config) => {
    const root = config.root
    if (typeof root !== 'string' || root.trim().length === 0) {
      return 'Dropbox root path is required.'
    }
    const tokenEnv = config.token_env
    if (typeof tokenEnv !== 'string' || tokenEnv.trim().length === 0) {
      return 'Dropbox token env var is required.'
    }
    return null
  },
  component: DropboxStorageForm,
}

export const STORAGE_BACKEND_ORDER: readonly StorageBackend[] = ['local', 'dropbox'] as const

export const STORAGE_BACKENDS: Record<StorageBackend, StorageBackendDef> = {
  local: LOCAL_BACKEND,
  dropbox: DROPBOX_BACKEND,
}
