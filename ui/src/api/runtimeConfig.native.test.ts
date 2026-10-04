// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const keychain = vi.hoisted(() => ({
  server: 'https://a.example',
  tokens: new Map<string, string>(),
  ready: new Set<string>(),
}))

vi.mock('../runtime/nativeRuntime', () => ({ isIOSNativeApp: () => true }))
vi.mock('./homeSecAuthPlugin', () => ({
  homeSecAuthPlugin: {
    getServerBaseUrl: async () => ({ value: keychain.server }),
    setServerBaseUrl: async ({ value }: { value: string }) => {
      if (keychain.server !== value) {
        keychain.tokens.delete(keychain.server)
        keychain.ready.delete(keychain.server)
      }
      keychain.server = value
    },
    getApiToken: async () => ({ value: keychain.tokens.get(keychain.server) ?? null }),
    clearApiToken: async () => { keychain.tokens.delete(keychain.server) },
    getAuthDisabledReady: async () => ({ value: keychain.ready.has(keychain.server) }),
    clearAuthDisabledReady: async () => { keychain.ready.delete(keychain.server) },
    setAuthDisabledReady: async ({ value }: { value: boolean }) => {
      if (value) keychain.ready.add(keychain.server)
      else keychain.ready.delete(keychain.server)
    },
  },
}))

import { apiClient } from './client'
import { initializeApiRuntimeConfig } from './runtimeConfig'
import {
  BROWSER_AUTH_DISABLED_SESSION_READY_STORAGE_KEY,
  clearRuntimeAuthSessionReady,
  isRuntimeAuthSessionReady,
  persistRuntimeAuthSessionReady,
} from './tokenProvider'

describe('native runtime server overrides', () => {
  beforeEach(() => {
    keychain.server = 'https://a.example'
    keychain.tokens.clear()
    keychain.ready.clear()
    window.sessionStorage.clear()
    clearRuntimeAuthSessionReady()
  })

  afterEach(() => { vi.restoreAllMocks() })

  it.each(['token', 'auth-disabled', 'webview-ready'])('clears %s auth when overriding the server', async (authMode) => {
    // Given: The old server has a token or an acknowledged auth-disabled session
    if (authMode === 'token') keychain.tokens.set(keychain.server, 'server-a-secret')
    if (authMode === 'auth-disabled') keychain.ready.add(keychain.server)
    if (authMode === 'webview-ready') {
      await persistRuntimeAuthSessionReady({ persistAuthDisabled: true })
    }
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response('[]', { headers: { 'content-type': 'application/json' } }),
    )

    // When: Startup overrides the stored URL with a different server
    await initializeApiRuntimeConfig({
      runtimeConfigSource: { loadRuntimeConfig: () => ({ serverBaseUrl: 'https://b.example' }) },
    })
    await apiClient.getCameras()

    // Then: B receives no A credential and the app requires fresh setup
    expect(fetchSpy.mock.calls[0]?.[0]).toBe('https://b.example/api/v1/cameras')
    expect(fetchSpy.mock.calls[0]?.[1]?.headers).not.toHaveProperty('Authorization')
    expect(isRuntimeAuthSessionReady()).toBe(false)
    expect(window.sessionStorage.getItem(BROWSER_AUTH_DISABLED_SESSION_READY_STORAGE_KEY)).toBeNull()
  })

  it('preserves credentials when the configured URL names the same server', async () => {
    // Given: A stored token for the configured server
    keychain.tokens.set(keychain.server, 'server-a-secret')
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response('[]', { headers: { 'content-type': 'application/json' } }),
    )

    // When: The runtime supplies the same URL with trailing slash/whitespace
    await initializeApiRuntimeConfig({
      runtimeConfigSource: { loadRuntimeConfig: () => ({ serverBaseUrl: ' https://a.example/ ' }) },
    })
    await apiClient.getCameras()

    // Then: URL normalization preserves the valid token
    expect(fetchSpy.mock.calls[0]?.[1]?.headers).toHaveProperty('Authorization', 'Bearer server-a-secret')
    expect(isRuntimeAuthSessionReady()).toBe(true)
  })
})
