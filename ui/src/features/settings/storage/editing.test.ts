import { describe, expect, it } from 'vitest'

import type { ConfigSnapshot } from '../../../api/client'
import { buildStorageSettingsPatch, storageSettingsDraft } from './editing'

function config(storage: Record<string, unknown>): ConfigSnapshot {
  return { config: { storage }, saved_config_version: 'original', active_config_version: 'original',
    apply_required: 'none', credentials: {}, credentials_editable: true, httpStatus: 200 }
}

describe('storage settings patches', () => {
  it('hydrates omitted Dropbox defaults without writing them in a root-only patch', () => {
    // Given: Minimal saved Dropbox config has refresh auth and custom web URL
    const draft = storageSettingsDraft(config({ backend: 'dropbox', config: {
      root: '/old', app_key_env: 'CUSTOM_KEY', refresh_token_env: 'CUSTOM_REFRESH',
      web_url_prefix: 'https://www.dropbox.com/home',
    }, paths: { clips_dir: '/recordings' } }))!
    // When: Root changes without touching credentials or paths
    draft.value = { ...draft.value, config: { ...draft.value.config, root: '/new' } }
    const patch = buildStorageSettingsPatch(draft)
    // Then: Only the root and baseline precondition are submitted
    expect(draft.value.config.token_env).toBe('DROPBOX_TOKEN')
    expect(patch).toEqual({ expected_config_version: 'original', storage: { config: { root: '/new' } } })
  })

  it('preserves hidden redacted values when unrelated fields change', () => {
    // Given: Saved backend payload contains a redacted advanced value
    const draft = storageSettingsDraft(config({ backend: 'local', config: {
      root: '/old', credential: '***redacted***',
    } }))!
    // When: Editing the root
    draft.value = { ...draft.value, config: { ...draft.value.config, root: '/new' } }
    // Then: The mask is omitted rather than persisted
    expect(buildStorageSettingsPatch(draft).storage?.config).toEqual({ root: '/new' })
  })

  it('returns unsupported backends as read-only', () => {
    // Given: A deployment uses a backend outside the shipped guided forms
    const saved = config({ backend: 's3', config: { bucket: 'clips' } })
    // When: Resolving a storage editor draft
    const draft = storageSettingsDraft(saved)
    // Then: The payload is preserved without conversion to local storage
    expect(draft).toBeNull()
    expect(saved.config.storage).toEqual({ backend: 's3', config: { bucket: 'clips' } })
  })

  it('rejects raw credentials typed into an env reference', () => {
    // Given: A Dropbox editor with a raw token in the env-name field
    const draft = storageSettingsDraft(config({ backend: 'dropbox', config: { root: '/clips' } }))!
    draft.value = { ...draft.value, config: { ...draft.value.config, token_env: 'sl.example.token' } }
    // When: Building the save patch
    // Then: Raw secrets are refused before any request
    expect(() => buildStorageSettingsPatch(draft)).toThrow('environment variable name')
  })
})
