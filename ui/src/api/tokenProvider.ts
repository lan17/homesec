import type { ApiRequestOptions } from './generated/client'

import { isIOSNativeApp } from '../runtime/nativeRuntime'

export const BROWSER_AUTH_TOKEN_STORAGE_KEY = 'homesec.apiKey'
export const BROWSER_AUTH_DISABLED_SESSION_READY_STORAGE_KEY =
  'homesec.authDisabledSessionReady'

export interface AuthTokenProvider {
  getToken(): Promise<string | null>
  setToken(token: string | null): Promise<void>
  clearToken(): Promise<void>
}

export interface SyncAuthTokenProvider extends AuthTokenProvider {
  getTokenSync(): string | null
  setTokenSync(token: string | null): void
  clearTokenSync(): void
}

type AuthTokenStorage = Pick<Storage, 'getItem' | 'removeItem' | 'setItem'>

function getWindowSessionStorage(): AuthTokenStorage | null {
  if (typeof window === 'undefined') {
    return null
  }

  try {
    return window.sessionStorage
  } catch {
    return null
  }
}

export function normalizeAuthToken(token: string | null | undefined): string | null {
  const trimmed = token?.trim() ?? ''
  return trimmed.length > 0 ? trimmed : null
}

export class BrowserAuthTokenProvider implements SyncAuthTokenProvider {
  private readonly getStorage: () => AuthTokenStorage | null
  private readonly storageKey: string

  constructor(
    getStorage: () => AuthTokenStorage | null = getWindowSessionStorage,
    storageKey = BROWSER_AUTH_TOKEN_STORAGE_KEY,
  ) {
    this.getStorage = getStorage
    this.storageKey = storageKey
  }

  getTokenSync(): string | null {
    const storage = this.getStorage()
    if (!storage) {
      return null
    }

    return normalizeAuthToken(storage.getItem(this.storageKey))
  }

  setTokenSync(token: string | null): void {
    const storage = this.getStorage()
    if (!storage) {
      return
    }

    const normalized = normalizeAuthToken(token)
    if (!normalized) {
      storage.removeItem(this.storageKey)
      return
    }

    storage.setItem(this.storageKey, normalized)
  }

  clearTokenSync(): void {
    const storage = this.getStorage()
    if (!storage) {
      return
    }

    storage.removeItem(this.storageKey)
  }

  async getToken(): Promise<string | null> {
    return this.getTokenSync()
  }

  async setToken(token: string | null): Promise<void> {
    this.setTokenSync(token)
  }

  async clearToken(): Promise<void> {
    this.clearTokenSync()
  }
}

export class InMemoryAuthTokenProvider implements SyncAuthTokenProvider {
  private token: string | null = null

  getTokenSync(): string | null {
    return this.token
  }

  setTokenSync(token: string | null): void {
    this.token = normalizeAuthToken(token)
  }

  clearTokenSync(): void {
    this.token = null
  }

  async getToken(): Promise<string | null> {
    return this.getTokenSync()
  }

  async setToken(token: string | null): Promise<void> {
    this.setTokenSync(token)
  }

  async clearToken(): Promise<void> {
    this.clearTokenSync()
  }
}

export const browserAuthTokenProvider = new BrowserAuthTokenProvider()
export const nativeAuthTokenProvider = new InMemoryAuthTokenProvider()
export const runtimeAuthTokenProvider: SyncAuthTokenProvider = isIOSNativeApp()
  ? nativeAuthTokenProvider
  : browserAuthTokenProvider
let nativeAuthSessionReady = false

export function hasAuthToken(provider: SyncAuthTokenProvider): boolean {
  return provider.getTokenSync() !== null
}

function hasPersistedAuthDisabledSessionReady(): boolean {
  return getWindowSessionStorage()?.getItem(BROWSER_AUTH_DISABLED_SESSION_READY_STORAGE_KEY) === 'true'
}

function persistAuthDisabledSessionReady(ready: boolean): void {
  const storage = getWindowSessionStorage()
  if (!storage) {
    return
  }

  if (ready) {
    storage.setItem(BROWSER_AUTH_DISABLED_SESSION_READY_STORAGE_KEY, 'true')
    return
  }

  storage.removeItem(BROWSER_AUTH_DISABLED_SESSION_READY_STORAGE_KEY)
}

export function markRuntimeAuthSessionReady(options: { persistAuthDisabled?: boolean } = {}): void {
  if (isIOSNativeApp()) {
    nativeAuthSessionReady = true
    persistAuthDisabledSessionReady(Boolean(options.persistAuthDisabled))
  }
}

export function clearRuntimeAuthSessionReady(): void {
  if (isIOSNativeApp()) {
    nativeAuthSessionReady = false
    persistAuthDisabledSessionReady(false)
  }
}

export function isRuntimeAuthSessionReady(): boolean {
  if (!isIOSNativeApp()) {
    return true
  }

  return (
    nativeAuthSessionReady ||
    hasAuthToken(runtimeAuthTokenProvider) ||
    hasPersistedAuthDisabledSessionReady()
  )
}

export async function resolveAuthToken(
  explicitApiKey: ApiRequestOptions['apiKey'],
  provider: AuthTokenProvider = browserAuthTokenProvider,
): Promise<string | null> {
  if (explicitApiKey !== undefined) {
    return normalizeAuthToken(explicitApiKey)
  }

  return provider.getToken()
}
