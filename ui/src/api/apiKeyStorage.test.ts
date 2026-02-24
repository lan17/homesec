import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  clearApiKey,
  getStoredApiKey,
  hasStoredApiKey,
  resolveApiKey,
  saveApiKey,
} from './apiKeyStorage'

function installWindowSessionStorageMock(): void {
  const store = new Map<string, string>()
  vi.stubGlobal('window', {
    sessionStorage: {
      getItem: (key: string): string | null => store.get(key) ?? null,
      setItem: (key: string, value: string): void => {
        store.set(key, value)
      },
      removeItem: (key: string): void => {
        store.delete(key)
      },
      clear: (): void => {
        store.clear()
      },
    },
  })
}

describe('apiKeyStorage', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('saves, loads, and clears API key values', () => {
    // Given: A client API key
    installWindowSessionStorageMock()
    const apiKey = 'secret-key'

    // When: Saving and reading key from session storage
    saveApiKey(apiKey)
    const stored = getStoredApiKey()
    const hasBeforeClear = hasStoredApiKey()
    clearApiKey()
    const cleared = getStoredApiKey()

    // Then: Key persistence and clear behavior are consistent
    expect(stored).toBe('secret-key')
    expect(hasBeforeClear).toBe(true)
    expect(cleared).toBeNull()
  })

  it('prefers explicit apiKey and falls back to storage when omitted', () => {
    // Given: A stored API key and an explicit override key
    installWindowSessionStorageMock()
    saveApiKey('stored-secret')

    // When: Resolving keys with explicit and implicit values
    const explicit = resolveApiKey('explicit-secret')
    const implicit = resolveApiKey(undefined)

    // Then: Explicit key wins, otherwise storage key is used
    expect(explicit).toBe('explicit-secret')
    expect(implicit).toBe('stored-secret')
  })
})
