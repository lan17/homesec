import type { JSX } from 'react'

import type { TestConnectionRequest } from '../../../api/generated/types'
import type { CredentialFields } from '../CredentialField'

export type StorageBackend = 'local' | 'dropbox'

export interface StorageFormState {
  backend: StorageBackend
  config: Record<string, unknown>
}

export interface StorageBackendFormProps {
  config: Record<string, unknown>
  onChange: (config: Record<string, unknown>) => void
  credentials?: CredentialFields
}

export interface StorageBackendDef {
  id: StorageBackend
  label: string
  description: string
  defaultConfig: Record<string, unknown>
  validate: (config: Record<string, unknown>) => string | null
  component: (props: StorageBackendFormProps) => JSX.Element
}

export function buildStorageTestRequest(value: StorageFormState): TestConnectionRequest {
  return {
    type: 'storage',
    backend: value.backend,
    config: value.config,
  }
}
