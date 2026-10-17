import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  API_KEY_STORAGE_KEY,
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
    vi.doUnmock('./homeSecAuthPlugin')
    vi.doUnmock('../runtime/nativeRuntime')
  })

  it('saves, loads, and clears API key values', async () => {
    // Given: A client API key
    installWindowSessionStorageMock()
    const apiKey = 'secret-key'

    // When: Saving and reading key from session storage
    await saveApiKey(apiKey)
    const stored = getStoredApiKey()
    const hasBeforeClear = hasStoredApiKey()
    await clearApiKey()
    const cleared = getStoredApiKey()

    // Then: Key persistence and clear behavior are consistent
    expect(stored).toBe('secret-key')
    expect(hasBeforeClear).toBe(true)
    expect(cleared).toBeNull()
  })

  it('prefers explicit apiKey and falls back to storage when omitted', async () => {
    // Given: A stored API key and an explicit override key
    installWindowSessionStorageMock()
    await saveApiKey('stored-secret')

    // When: Resolving keys with explicit and implicit values
    const explicit = resolveApiKey('explicit-secret')
    const implicit = resolveApiKey(undefined)

    // Then: Explicit key wins, otherwise storage key is used
    expect(explicit).toBe('explicit-secret')
    expect(implicit).toBe('stored-secret')
  })

  it('normalizes blank API key values as absent', async () => {
    // Given: A browser storage area and a whitespace API key
    installWindowSessionStorageMock()

    // When: Saving a blank key value
    await saveApiKey('   ')
    const stored = getStoredApiKey()
    const hasKey = hasStoredApiKey()

    // Then: Blank keys are treated as missing credentials
    expect(stored).toBeNull()
    expect(hasKey).toBe(false)
    expect(resolveApiKey('  ')).toBeNull()
  })

  it('uses the native runtime token provider when iOS mode is active', async () => {
    // Given: The runtime is loaded in native iOS mode with browser session storage available
    vi.resetModules()
    installWindowSessionStorageMock()
    const nativeAuthPlugin = {
      getServerBaseUrl: vi.fn(async () => ({ value: 'https://homesec.example.com' })),
      setServerBaseUrl: vi.fn(async () => {}),
      clearServerBaseUrl: vi.fn(async () => {}),
      getApiToken: vi.fn(async () => ({ value: null })),
      setApiToken: vi.fn(async () => {}),
      clearApiToken: vi.fn(async () => {}),
      getAuthDisabledReady: vi.fn(async () => ({ value: false })),
      setAuthDisabledReady: vi.fn(async () => {}),
      clearAuthDisabledReady: vi.fn(async () => {}),
    }
    vi.doMock('./homeSecAuthPlugin', () => ({
      homeSecAuthPlugin: nativeAuthPlugin,
    }))
    vi.doMock('../runtime/nativeRuntime', () => ({
      isIOSNativeApp: () => true,
    }))
    const nativeApiKeyStorage = await import('./apiKeyStorage')
    const tokenProvider = await import('./tokenProvider')

    // When: The shared auth recovery helpers save an API key
    await nativeApiKeyStorage.saveApiKey(' native-secret ')
    const stored = nativeApiKeyStorage.getStoredApiKey()
    const ready = tokenProvider.isRuntimeAuthSessionReady()
    await nativeApiKeyStorage.clearApiKey()

    // Then: The iOS API client token source is updated without writing WebView storage
    expect(nativeAuthPlugin.setApiToken).toHaveBeenCalledWith({ value: 'native-secret' })
    expect(nativeAuthPlugin.clearApiToken).toHaveBeenCalledTimes(1)
    expect(stored).toBe('native-secret')
    expect(ready).toBe(true)
    expect(tokenProvider.nativeAuthTokenProvider.getTokenSync()).toBeNull()
    expect(window.sessionStorage.getItem(API_KEY_STORAGE_KEY)).toBeNull()
  })
})
