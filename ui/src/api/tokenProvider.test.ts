import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  BROWSER_AUTH_DISABLED_SESSION_READY_STORAGE_KEY,
  BROWSER_AUTH_TOKEN_STORAGE_KEY,
  BrowserAuthTokenProvider,
  InMemoryAuthTokenProvider,
  NativeAuthTokenProvider,
  normalizeAuthToken,
  resolveAuthToken,
  type AuthTokenProvider,
} from './tokenProvider'
import type { HomeSecAuthPlugin } from './homeSecAuthPlugin'

type TestStorage = Pick<Storage, 'getItem' | 'removeItem' | 'setItem'> & {
  values: Map<string, string>
}

function createNativePluginMock(
  initial: { token?: string | null; authDisabledReady?: boolean } = {},
): HomeSecAuthPlugin {
  let token = initial.token ?? null
  let authDisabledReady = initial.authDisabledReady ?? false
  return {
    getServerBaseUrl: vi.fn(async () => ({ value: 'https://homesec.example.com' })),
    setServerBaseUrl: vi.fn(async () => {}),
    clearServerBaseUrl: vi.fn(async () => {}),
    getApiToken: vi.fn(async () => ({ value: token })),
    setApiToken: vi.fn(async ({ value }) => {
      token = value
    }),
    clearApiToken: vi.fn(async () => {
      token = null
    }),
    getAuthDisabledReady: vi.fn(async () => ({ value: authDisabledReady })),
    setAuthDisabledReady: vi.fn(async ({ value }) => {
      authDisabledReady = value
    }),
    clearAuthDisabledReady: vi.fn(async () => {
      authDisabledReady = false
    }),
  }
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

describe('NativeAuthTokenProvider', () => {
  it('hydrates token and auth-disabled readiness from the native bridge', async () => {
    // Given: A native bridge with stored token and auth-disabled state
    const plugin = createNativePluginMock({
      token: ' native-secret ',
      authDisabledReady: true,
    })
    const provider = new NativeAuthTokenProvider(plugin)

    // When: Hydrating the native provider
    await provider.hydrate()

    // Then: Token and readiness are cached for synchronous route guards
    expect(provider.getTokenSync()).toBe('native-secret')
    expect(provider.isAuthDisabledReadySync()).toBe(true)
  })

  it('sets and clears native token values through the bridge', async () => {
    // Given: A native provider backed by a bridge plugin
    const plugin = createNativePluginMock()
    const provider = new NativeAuthTokenProvider(plugin)

    // When: Persisting then clearing a native token
    await provider.setToken(' native-secret ')
    const stored = provider.getTokenSync()
    await provider.clearToken()
    const cleared = provider.getTokenSync()

    // Then: Writes go through the native bridge and update the sync cache
    expect(plugin.setApiToken).toHaveBeenCalledWith({ value: 'native-secret' })
    expect(stored).toBe('native-secret')
    expect(plugin.clearApiToken).toHaveBeenCalledTimes(1)
    expect(cleared).toBeNull()
  })
})

describe('runtime auth session readiness', () => {
  it('persists auth-disabled readiness without a token in native iOS mode', async () => {
    // Given: The runtime is loaded in native iOS mode with a native auth bridge
    vi.resetModules()
    const storage = installWindowSessionStorageMock()
    const nativePlugin = createNativePluginMock()
    vi.doMock('./homeSecAuthPlugin', () => ({
      homeSecAuthPlugin: nativePlugin,
    }))
    vi.doMock('../runtime/nativeRuntime', () => ({
      isIOSNativeApp: () => true,
    }))
    const tokenProvider = await import('./tokenProvider')

    // When: Setup persists an auth-disabled server as ready
    await tokenProvider.persistRuntimeAuthSessionReady({ persistAuthDisabled: true })
    const readyBeforeReload = tokenProvider.isRuntimeAuthSessionReady()
    vi.resetModules()
    vi.doMock('./homeSecAuthPlugin', () => ({
      homeSecAuthPlugin: nativePlugin,
    }))
    const reloadedTokenProvider = await import('./tokenProvider')
    await reloadedTokenProvider.nativeAuthTokenProvider.hydrate()
    const readyAfterReload = reloadedTokenProvider.isRuntimeAuthSessionReady()

    // Then: Readiness survives reload without persisting an API token
    expect(readyBeforeReload).toBe(true)
    expect(readyAfterReload).toBe(true)
    expect(storage.values.get(BROWSER_AUTH_DISABLED_SESSION_READY_STORAGE_KEY)).toBe('true')
    expect(reloadedTokenProvider.nativeAuthTokenProvider.getTokenSync()).toBeNull()
  })

  it('hydrates protected native sessions after reload', async () => {
    // Given: The runtime is loaded in native iOS mode with a native auth bridge
    vi.resetModules()
    const storage = installWindowSessionStorageMock()
    const nativePlugin = createNativePluginMock()
    vi.doMock('./homeSecAuthPlugin', () => ({
      homeSecAuthPlugin: nativePlugin,
    }))
    vi.doMock('../runtime/nativeRuntime', () => ({
      isIOSNativeApp: () => true,
    }))
    const tokenProvider = await import('./tokenProvider')

    // When: Setup persists a protected server token through native storage
    await tokenProvider.runtimeAuthTokenProvider.setToken('native-token')
    await tokenProvider.persistRuntimeAuthSessionReady({ persistAuthDisabled: false })
    const readyBeforeReload = tokenProvider.isRuntimeAuthSessionReady()
    vi.resetModules()
    vi.doMock('./homeSecAuthPlugin', () => ({
      homeSecAuthPlugin: nativePlugin,
    }))
    const reloadedTokenProvider = await import('./tokenProvider')
    await reloadedTokenProvider.nativeAuthTokenProvider.hydrate()
    const readyAfterReload = reloadedTokenProvider.isRuntimeAuthSessionReady()

    // Then: Token-backed readiness survives reload through the native bridge
    expect(readyBeforeReload).toBe(true)
    expect(readyAfterReload).toBe(true)
    expect(storage.values.has(BROWSER_AUTH_DISABLED_SESSION_READY_STORAGE_KEY)).toBe(false)
    expect(reloadedTokenProvider.nativeAuthTokenProvider.getTokenSync()).toBe('native-token')
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
