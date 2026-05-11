import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  BROWSER_AUTH_DISABLED_SESSION_READY_STORAGE_KEY,
  BROWSER_AUTH_TOKEN_STORAGE_KEY,
  BrowserAuthTokenProvider,
  InMemoryAuthTokenProvider,
  normalizeAuthToken,
  resolveAuthToken,
  type AuthTokenProvider,
} from './tokenProvider'

type TestStorage = Pick<Storage, 'getItem' | 'removeItem' | 'setItem'> & {
  values: Map<string, string>
}

function createStorage(): TestStorage {
  const values = new Map<string, string>()
  return {
    values,
    getItem: (key: string): string | null => values.get(key) ?? null,
    setItem: (key: string, value: string): void => {
      values.set(key, value)
    },
    removeItem: (key: string): void => {
      values.delete(key)
    },
  }
}

function installWindowSessionStorageMock(): TestStorage {
  const storage = createStorage()
  vi.stubGlobal('window', {
    sessionStorage: {
      getItem: storage.getItem,
      setItem: storage.setItem,
      removeItem: storage.removeItem,
      clear: (): void => {
        storage.values.clear()
      },
    },
  })
  return storage
}

afterEach(() => {
  vi.unstubAllGlobals()
  vi.doUnmock('../runtime/nativeRuntime')
})

describe('BrowserAuthTokenProvider', () => {
  it('sets, gets, and clears token values from storage', async () => {
    // Given: A browser token provider backed by session storage
    const storage = createStorage()
    const provider = new BrowserAuthTokenProvider(() => storage)

    // When: Persisting and then clearing a token
    await provider.setToken(' secret-key ')
    const stored = await provider.getToken()
    const persisted = storage.values.get(BROWSER_AUTH_TOKEN_STORAGE_KEY)
    await provider.clearToken()
    const cleared = await provider.getToken()

    // Then: Token values are normalized and removable
    expect(stored).toBe('secret-key')
    expect(persisted).toBe('secret-key')
    expect(cleared).toBeNull()
    expect(storage.values.has(BROWSER_AUTH_TOKEN_STORAGE_KEY)).toBe(false)
  })

  it('treats blank token values as absent', async () => {
    // Given: A provider with a stored whitespace-only token
    const storage = createStorage()
    const provider = new BrowserAuthTokenProvider(() => storage)
    storage.setItem(BROWSER_AUTH_TOKEN_STORAGE_KEY, '   ')

    // When: Reading and then setting a blank token
    const storedBlank = await provider.getToken()
    await provider.setToken('  ')
    const afterSetBlank = await provider.getToken()

    // Then: Blank tokens are not exposed or persisted
    expect(storedBlank).toBeNull()
    expect(afterSetBlank).toBeNull()
    expect(storage.values.has(BROWSER_AUTH_TOKEN_STORAGE_KEY)).toBe(false)
    expect(normalizeAuthToken('\tsecret\n')).toBe('secret')
  })
})

describe('InMemoryAuthTokenProvider', () => {
  it('keeps token values in memory only', async () => {
    // Given: A native-runtime token provider without browser storage
    const provider = new InMemoryAuthTokenProvider()

    // When: Persisting and then clearing a token
    await provider.setToken(' native-secret ')
    const stored = await provider.getToken()
    await provider.clearToken()
    const cleared = await provider.getToken()

    // Then: Token values are normalized without depending on session storage
    expect(stored).toBe('native-secret')
    expect(provider.getTokenSync()).toBeNull()
    expect(cleared).toBeNull()
  })
})

describe('runtime auth session readiness', () => {
  it('persists auth-disabled readiness without a token in native iOS mode', async () => {
    // Given: The runtime is loaded in native iOS mode with browser session storage available
    vi.resetModules()
    const storage = installWindowSessionStorageMock()
    vi.doMock('../runtime/nativeRuntime', () => ({
      isIOSNativeApp: () => true,
    }))
    const tokenProvider = await import('./tokenProvider')

    // When: Setup marks an auth-disabled server as ready
    tokenProvider.markRuntimeAuthSessionReady({ persistAuthDisabled: true })
    const readyBeforeReload = tokenProvider.isRuntimeAuthSessionReady()
    vi.resetModules()
    const reloadedTokenProvider = await import('./tokenProvider')
    const readyAfterReload = reloadedTokenProvider.isRuntimeAuthSessionReady()

    // Then: Readiness survives reload without persisting an API token
    expect(readyBeforeReload).toBe(true)
    expect(readyAfterReload).toBe(true)
    expect(storage.values.get(BROWSER_AUTH_DISABLED_SESSION_READY_STORAGE_KEY)).toBe('true')
    expect(reloadedTokenProvider.nativeAuthTokenProvider.getTokenSync()).toBeNull()
  })

  it('does not persist protected native sessions after reload', async () => {
    // Given: The runtime is loaded in native iOS mode with browser session storage available
    vi.resetModules()
    const storage = installWindowSessionStorageMock()
    vi.doMock('../runtime/nativeRuntime', () => ({
      isIOSNativeApp: () => true,
    }))
    const tokenProvider = await import('./tokenProvider')

    // When: Setup marks a protected server as ready with an in-memory token
    await tokenProvider.runtimeAuthTokenProvider.setToken('native-token')
    tokenProvider.markRuntimeAuthSessionReady({ persistAuthDisabled: false })
    const readyBeforeReload = tokenProvider.isRuntimeAuthSessionReady()
    vi.resetModules()
    const reloadedTokenProvider = await import('./tokenProvider')
    const readyAfterReload = reloadedTokenProvider.isRuntimeAuthSessionReady()

    // Then: Token-backed readiness remains memory-only
    expect(readyBeforeReload).toBe(true)
    expect(readyAfterReload).toBe(false)
    expect(storage.values.has(BROWSER_AUTH_DISABLED_SESSION_READY_STORAGE_KEY)).toBe(false)
    expect(reloadedTokenProvider.nativeAuthTokenProvider.getTokenSync()).toBeNull()
  })
})

describe('resolveAuthToken', () => {
  it('prefers explicit request tokens over provider values', async () => {
    // Given: A provider with a different stored token
    const provider: AuthTokenProvider = {
      getToken: vi.fn().mockResolvedValue('stored-secret'),
      setToken: vi.fn(),
      clearToken: vi.fn(),
    }

    // When: Resolving an explicit API token
    const resolved = await resolveAuthToken(' explicit-secret ', provider)

    // Then: Explicit values win without consulting storage
    expect(resolved).toBe('explicit-secret')
    expect(provider.getToken).not.toHaveBeenCalled()
  })

  it('falls back to the configured provider when no request token is supplied', async () => {
    // Given: A provider with a stored token
    const provider: AuthTokenProvider = {
      getToken: vi.fn().mockResolvedValue('stored-secret'),
      setToken: vi.fn(),
      clearToken: vi.fn(),
    }

    // When: Resolving without an explicit request token
    const resolved = await resolveAuthToken(undefined, provider)

    // Then: The provider supplies the token
    expect(resolved).toBe('stored-secret')
    expect(provider.getToken).toHaveBeenCalledTimes(1)
  })
})
