import { describe, expect, it } from 'vitest'

import {
  BROWSER_SERVER_BASE_URL_STORAGE_KEY,
  BrowserServerBaseUrlProvider,
  NativeServerBaseUrlProvider,
  normalizeServerBaseUrl,
} from './serverBaseUrlProvider'
import type { HomeSecAuthPlugin } from './homeSecAuthPlugin'

type TestStorage = Pick<Storage, 'getItem' | 'removeItem' | 'setItem'> & {
  values: Map<string, string>
}

function createNativePluginMock(initialBaseUrl: string | null = null): HomeSecAuthPlugin {
  let baseUrl = initialBaseUrl
  return {
    getServerBaseUrl: async () => ({ value: baseUrl }),
    setServerBaseUrl: async ({ value }) => {
      baseUrl = value
    },
    clearServerBaseUrl: async () => {
      baseUrl = null
    },
    getApiToken: async () => ({ value: null }),
    setApiToken: async () => {},
    clearApiToken: async () => {},
    getAuthDisabledReady: async () => ({ value: false }),
    setAuthDisabledReady: async () => {},
    clearAuthDisabledReady: async () => {},
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

describe('normalizeServerBaseUrl', () => {
  it('normalizes LAN and HTTPS URLs while preserving unset same-origin mode', () => {
    // Given: Candidate server base URL values
    const lanUrl = ' http://192.168.1.10:8081/// '
    const httpsUrl = 'https://homesec.example.com/'
    const pathUrl = 'https://homesec.example.com/homesec///'

    // When / Then: URLs are trimmed and empty values stay unset
    expect(normalizeServerBaseUrl(lanUrl)).toBe('http://192.168.1.10:8081')
    expect(normalizeServerBaseUrl(httpsUrl)).toBe('https://homesec.example.com')
    expect(normalizeServerBaseUrl(pathUrl)).toBe('https://homesec.example.com/homesec')
    expect(normalizeServerBaseUrl('   ')).toBeNull()
    expect(normalizeServerBaseUrl(null)).toBeNull()
  })
})

describe('BrowserServerBaseUrlProvider', () => {
  it('sets, gets, and clears runtime base URL overrides', async () => {
    // Given: A provider with a build-time fallback base URL
    const storage = createStorage()
    const provider = new BrowserServerBaseUrlProvider('http://localhost:8081/', () => storage)

    // When: Overriding and then clearing the runtime base URL
    const fallback = await provider.getBaseUrl()
    await provider.setBaseUrl(' https://homesec.example.com/// ')
    const override = await provider.getBaseUrl()
    const persistedOverride = storage.values.get(BROWSER_SERVER_BASE_URL_STORAGE_KEY)
    await provider.clearBaseUrl()
    const afterClear = await provider.getBaseUrl()

    // Then: Runtime values are normalized and clearing returns to the fallback
    expect(fallback).toBe('http://localhost:8081')
    expect(override).toBe('https://homesec.example.com')
    expect(persistedOverride).toBe('https://homesec.example.com')
    expect(afterClear).toBe('http://localhost:8081')
  })

  it('distinguishes explicit same-origin override from clearing runtime config', async () => {
    // Given: A provider with a stored runtime value and Vite-provided fallback
    const storage = createStorage()
    const provider = new BrowserServerBaseUrlProvider('http://localhost:8081', () => storage)
    await provider.setBaseUrl('http://192.168.1.10:8081')

    // When: Replacing the runtime value with a blank string and then clearing it
    await provider.setBaseUrl('   ')
    const sameOrigin = await provider.getBaseUrl()
    const persistedSameOrigin = storage.values.get(BROWSER_SERVER_BASE_URL_STORAGE_KEY)
    await provider.clearBaseUrl()
    const afterClear = await provider.getBaseUrl()

    // Then: Blank runtime values force same-origin; explicit clearing returns to fallback
    expect(sameOrigin).toBeNull()
    expect(persistedSameOrigin).toBe('')
    expect(afterClear).toBe('http://localhost:8081')
    expect(storage.values.has(BROWSER_SERVER_BASE_URL_STORAGE_KEY)).toBe(false)
  })
})

describe('NativeServerBaseUrlProvider', () => {
  it('hydrates, updates, and clears server URL values through the native bridge', async () => {
    // Given: A native bridge with a stored HomeSec server URL
    const plugin = createNativePluginMock(' http://192.168.1.10:8081/// ')
    const provider = new NativeServerBaseUrlProvider(plugin)

    // When: Hydrating, updating, and clearing the native URL cache
    await provider.hydrate()
    const hydrated = provider.getBaseUrlSync()
    await provider.setBaseUrl('https://homesec.example.com/')
    const updated = await provider.getBaseUrl()
    await provider.clearBaseUrl()
    const cleared = provider.getBaseUrlSync()

    // Then: Values are normalized and remain available synchronously after hydration
    expect(hydrated).toBe('http://192.168.1.10:8081')
    expect(updated).toBe('https://homesec.example.com')
    expect(cleared).toBeNull()
  })
})
