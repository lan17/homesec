import { afterEach, describe, expect, it, vi } from 'vitest'

import { HomeSecApiClient } from './client'
import {
  BROWSER_SERVER_BASE_URL_STORAGE_KEY,
  BrowserServerBaseUrlProvider,
} from './serverBaseUrlProvider'
import {
  initializeApiRuntimeConfig,
  WindowApiRuntimeConfigSource,
  type ApiRuntimeConfigSource,
} from './runtimeConfig'

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

function sourceWithServerBaseUrl(serverBaseUrl?: string | null): ApiRuntimeConfigSource {
  return {
    loadRuntimeConfig: () =>
      serverBaseUrl === undefined
        ? {}
        : {
            serverBaseUrl,
          },
  }
}

describe('initializeApiRuntimeConfig', () => {
  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it('leaves browser fallback configuration intact without extra setup', async () => {
    // Given: Browser mode with only a build-time fallback base URL
    const storage = createStorage()
    const serverBaseUrlProvider = new BrowserServerBaseUrlProvider(
      'http://localhost:8081/',
      () => storage,
    )

    // When: Initializing without a runtime server URL
    await initializeApiRuntimeConfig({
      runtimeConfigSource: sourceWithServerBaseUrl(),
      serverBaseUrlProvider,
    })

    // Then: Existing browser fallback behavior is unchanged
    expect(await serverBaseUrlProvider.getBaseUrl()).toBe('http://localhost:8081')
    expect(storage.values.has(BROWSER_SERVER_BASE_URL_STORAGE_KEY)).toBe(false)
  })

  it('applies a native-provided LAN base URL before the first API call', async () => {
    // Given: Runtime config supplied before the React app mounts
    const storage = createStorage()
    const serverBaseUrlProvider = new BrowserServerBaseUrlProvider('', () => storage)
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async () =>
      new Response(
        JSON.stringify({
          status: 'healthy',
          pipeline: 'running',
          postgres: 'connected',
          cameras_online: 1,
          bootstrap_mode: false,
        }),
        {
          status: 200,
          headers: { 'content-type': 'application/json' },
        },
      ),
    )

    // When: App initialization applies the LAN URL before client use
    await initializeApiRuntimeConfig({
      runtimeConfigSource: sourceWithServerBaseUrl(' http://192.168.1.10:8081/// '),
      serverBaseUrlProvider,
    })
    const client = new HomeSecApiClient('', { serverBaseUrlProvider })
    await client.getHealth()

    // Then: First API call resolves against the runtime-configured server
    expect(fetchSpy.mock.calls[0]?.[0]).toBe('http://192.168.1.10:8081/api/v1/health')
  })

  it('supports HTTPS runtime base URLs and empty same-origin mode', async () => {
    // Given: A runtime-configurable provider with a build-time fallback
    const storage = createStorage()
    const serverBaseUrlProvider = new BrowserServerBaseUrlProvider(
      'http://localhost:8081',
      () => storage,
    )
    const client = new HomeSecApiClient('', { serverBaseUrlProvider })

    // When: Initializing with HTTPS and then an empty same-origin value
    await initializeApiRuntimeConfig({
      runtimeConfigSource: sourceWithServerBaseUrl('https://homesec.example.com/'),
      serverBaseUrlProvider,
    })
    const httpsUrl = serverBaseUrlProvider.getBaseUrlSync()

    await initializeApiRuntimeConfig({
      runtimeConfigSource: sourceWithServerBaseUrl('   '),
      serverBaseUrlProvider,
    })
    const sameOriginUrl = serverBaseUrlProvider.getBaseUrlSync()
    const sameOriginPath = client.resolvePath('/api/v1/health')

    // Then: HTTPS URLs normalize and empty values override the fallback with same-origin
    expect(httpsUrl).toBe('https://homesec.example.com')
    expect(sameOriginUrl).toBeNull()
    expect(sameOriginPath).toBe('/api/v1/health')
    expect(storage.values.get(BROWSER_SERVER_BASE_URL_STORAGE_KEY)).toBe('')
  })
})

describe('WindowApiRuntimeConfigSource', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('reads preloaded runtime config from the window object', () => {
    // Given: A native shell preloads configuration before the bundle starts
    vi.stubGlobal('window', {
      __HOMESEC_RUNTIME_CONFIG__: {
        serverBaseUrl: 'http://192.168.1.10:8081',
      },
    })

    // When: Loading browser-visible runtime configuration
    const config = new WindowApiRuntimeConfigSource().loadRuntimeConfig()

    // Then: The preloaded server URL is exposed to app initialization
    expect(config).toEqual({ serverBaseUrl: 'http://192.168.1.10:8081' })
  })
})
