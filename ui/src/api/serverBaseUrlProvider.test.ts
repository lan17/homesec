import { describe, expect, it } from 'vitest'

import {
  BROWSER_SERVER_BASE_URL_STORAGE_KEY,
  BrowserServerBaseUrlProvider,
  normalizeServerBaseUrl,
} from './serverBaseUrlProvider'

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

describe('normalizeServerBaseUrl', () => {
  it('normalizes LAN and HTTPS URLs while preserving unset same-origin mode', () => {
    // Given: Candidate server base URL values
    const lanUrl = ' http://192.168.1.10:8081/// '
    const httpsUrl = 'https://homesec.example.com/'

    // When / Then: URLs are trimmed and empty values stay unset
    expect(normalizeServerBaseUrl(lanUrl)).toBe('http://192.168.1.10:8081')
    expect(normalizeServerBaseUrl(httpsUrl)).toBe('https://homesec.example.com')
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

  it('falls back to the build-time base URL when runtime value is blank', async () => {
    // Given: A provider with a stored runtime value and Vite-provided fallback
    const storage = createStorage()
    const provider = new BrowserServerBaseUrlProvider('http://localhost:8081', () => storage)
    await provider.setBaseUrl('http://192.168.1.10:8081')

    // When: Replacing the runtime value with a blank string
    await provider.setBaseUrl('   ')
    const resolved = await provider.getBaseUrl()

    // Then: Blank runtime values are cleared and the fallback is used
    expect(resolved).toBe('http://localhost:8081')
    expect(storage.values.has(BROWSER_SERVER_BASE_URL_STORAGE_KEY)).toBe(false)
  })
})
