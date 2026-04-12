import { describe, expect, it } from 'vitest'

import type { StorageBackendMetadata } from '../../../api/generated/types'
import {
  buildStorageConfigPatch,
  buildStorageSecretPatch,
  defaultConfigForBackend,
} from './editorModel'

describe('storage editor model helpers', () => {
  it('builds recursive config patches while skipping unchanged redacted placeholders', () => {
    // Given: Nested storage config with one updated field, one removed field, and a redacted secret
    const base = {
      root: '/homesec',
      nested: {
        keep: 'same',
        change: 'before',
        remove: 'drop-me',
      },
      stale: true,
      secret: 'persisted-secret',
    }
    const next = {
      root: '/homesec',
      nested: {
        keep: 'same',
        change: 'after',
      },
      secret: '***redacted***',
    }

    // When: Building a storage config patch
    const patch = buildStorageConfigPatch(base, next, '***redacted***')

    // Then: Only meaningful nested changes are emitted
    expect(patch).toEqual({
      nested: {
        change: 'after',
        remove: null,
      },
      stale: null,
    })
  })

  it('omits blank secret fields while preserving nonblank replacement values', () => {
    // Given: Secret inputs include whitespace-only fields and explicit replacements
    const secretInputs = {
      access_token: '  ',
      refresh_token: 'refresh-me',
      app_secret: '  keep surrounding space  ',
    }

    // When: Building the secret patch
    const patch = buildStorageSecretPatch(secretInputs)

    // Then: Blank fields are omitted and provided replacements are preserved verbatim
    expect(patch).toEqual({
      refresh_token: 'refresh-me',
      app_secret: '  keep surrounding space  ',
    })
  })

  it('preserves built-in defaults when metadata omits a schema default', () => {
    // Given: Dropbox metadata omits root default but overrides other defaults
    const metadata: StorageBackendMetadata = {
      backend: 'dropbox',
      label: 'Dropbox',
      description: 'Upload to Dropbox',
      config_schema: {},
      fields: [
        {
          name: 'root',
          type: 'string',
          required: true,
          description: 'Dropbox root',
          default: null,
          secret: false,
        },
        {
          name: 'token_env',
          type: 'string',
          required: true,
          description: 'Token env var',
          default: 'CUSTOM_DROPBOX_TOKEN',
          secret: false,
        },
        {
          name: 'app_key_env',
          type: 'string',
          required: false,
          description: 'App key env var',
          default: 'DROPBOX_APP_KEY',
          secret: false,
        },
      ],
      secret_fields: [],
    }

    // When: Building default config for the backend
    const defaults = defaultConfigForBackend('dropbox', metadata)

    // Then: Built-in required defaults remain while metadata defaults still overlay
    expect(defaults).toEqual({
      root: '/homesec',
      token_env: 'CUSTOM_DROPBOX_TOKEN',
      app_key_env: 'DROPBOX_APP_KEY',
    })
  })
})
