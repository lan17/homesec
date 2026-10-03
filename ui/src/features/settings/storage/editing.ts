import type { ConfigPatch, ConfigSnapshot } from '../../../api/client'
import { validateEnvReferences } from '../../shared/envReferences'
import { expectConfigObject, expectConfigString, diffConfig } from '../configEditing'
import { STORAGE_BACKENDS } from './backends'
import type { StorageFormState } from './types'

export interface StorageSettingsDraft {
  expectedVersion: string
  original: StorageFormState
  value: StorageFormState
  originalPaths: Record<string, unknown>
  paths: Record<string, unknown>
}

export function storageSettingsDraft(config: ConfigSnapshot): StorageSettingsDraft | null {
  const storage = expectConfigObject(config.config.storage, 'Storage settings')
  const backend = expectConfigString(storage.backend, 'Storage backend')
  if (backend !== 'local' && backend !== 'dropbox') {
    return null
  }
  const raw = expectConfigObject(storage.config, 'Storage backend settings')
  const value: StorageFormState = {
    backend,
    config: { ...STORAGE_BACKENDS[backend].defaultConfig, ...raw },
  }
  const paths = {
    clips_dir: 'clips', backups_dir: 'backups', artifacts_dir: 'artifacts',
    ...expectConfigObject(storage.paths ?? {}, 'Storage paths'),
  }
  return {
    expectedVersion: config.saved_config_version,
    original: value, value, originalPaths: paths, paths,
  }
}

export function buildStorageSettingsPatch(draft: StorageSettingsDraft): ConfigPatch {
  const validation = STORAGE_BACKENDS[draft.value.backend].validate(draft.value.config)
  if (validation) {
    throw new Error(validation)
  }
  const config = diffConfig(draft.original.config, draft.value.config)
  validateEnvReferences(config)
  const paths = diffConfig(draft.originalPaths, draft.paths)
  for (const value of Object.values(paths)) {
    if (typeof value !== 'string' || value.trim().length === 0 || value.includes('\0')
      || value.startsWith('/') || value.split('/').includes('..')) {
      throw new Error("Storage destination directories must be relative without '..' segments.")
    }
  }
  return {
    expected_config_version: draft.expectedVersion,
    storage: {
      ...(Object.keys(config).length > 0 ? { config } : {}),
      ...(Object.keys(paths).length > 0 ? { paths } : {}),
    },
  }
}
