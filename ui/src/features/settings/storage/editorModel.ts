import type {
  StorageBackendMetadata,
  StorageBackendsResponse,
} from '../../../api/generated/types'
import { STORAGE_BACKENDS, STORAGE_BACKEND_ORDER } from './backends'
import type { StorageBackend } from './types'

export interface SupportedStorageBackendOption {
  backend: StorageBackend
  label: string
  description: string
  metadata: StorageBackendMetadata | null
}

export function isSupportedStorageBackend(backend: string): backend is StorageBackend {
  return Object.prototype.hasOwnProperty.call(STORAGE_BACKENDS, backend)
}

export function cloneStorageConfig(config: Record<string, unknown>): Record<string, unknown> {
  return JSON.parse(JSON.stringify(config)) as Record<string, unknown>
}

export function sameJsonValue(left: unknown, right: unknown): boolean {
  return JSON.stringify(left) === JSON.stringify(right)
}

export function getStorageBackendMetadata(
  backends: StorageBackendsResponse | null | undefined,
  backend: string,
): StorageBackendMetadata | null {
  const metadata = backends?.find((candidate) => candidate.backend === backend)
  return metadata ?? null
}

export function defaultConfigForBackend(
  backend: string,
  metadata: StorageBackendMetadata | null,
): Record<string, unknown> {
  const defaults: Record<string, unknown> = isSupportedStorageBackend(backend)
    ? cloneStorageConfig(STORAGE_BACKENDS[backend].defaultConfig)
    : {}

  if (!metadata) {
    return defaults
  }

  for (const field of metadata.fields) {
    if (field.default === null || field.default === undefined) {
      continue
    }
    defaults[field.name] = field.default
  }

  return defaults
}

export function buildSupportedStorageBackendOptions(
  backends: StorageBackendsResponse | null | undefined,
): SupportedStorageBackendOption[] {
  return STORAGE_BACKEND_ORDER.map((backend) => {
    const metadata = getStorageBackendMetadata(backends, backend)
    return {
      backend,
      label: metadata?.label ?? STORAGE_BACKENDS[backend].label,
      description: metadata?.description ?? STORAGE_BACKENDS[backend].description,
      metadata,
    }
  })
}
