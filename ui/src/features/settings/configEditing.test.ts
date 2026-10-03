import { describe, expect, it } from 'vitest'

import { diffConfig } from './configEditing'

describe('configuration merge patches', () => {
  it('preserves nested advanced settings and unchanged redacted values', () => {
    // Given: An existing configuration includes secrets and advanced nested settings
    const original = { host: 'old', auth: { password: '***redacted***', username_env: 'USER' }, qos: 2 }
    // When: Only the host and username reference change
    const patch = diffConfig(original, { ...original, host: 'new', auth: { ...original.auth, username_env: 'NEW_USER' } })
    // Then: The patch contains only the edited values
    expect(patch).toEqual({ host: 'new', auth: { username_env: 'NEW_USER' } })
  })

  it('uses null to explicitly clear removed settings', () => {
    // Given: A saved optional setting
    const original = { root: '/clips', optional: 'remove' }
    // When: The setting is removed
    const patch = diffConfig(original, { root: '/clips' })
    // Then: The merge patch explicitly clears that key
    expect(patch).toEqual({ optional: null })
  })

  it('rejects edited URLs and nested arrays that contain redacted placeholders', () => {
    // Given: The original configuration contains a redacted URL
    const original = { url: 'https://***redacted***@old.test' }
    // When/Then: Editing around a placeholder cannot submit it as a real credential
    expect(() => diffConfig(original, { url: 'https://***redacted***@new.test' })).toThrow('Redacted values')
    expect(() => diffConfig({}, { values: [{ token: '***redacted***' }] })).toThrow('Redacted values')
  })
})
