import type { ApiRequestOptions } from './generated/client'

export const BROWSER_AUTH_TOKEN_STORAGE_KEY = 'homesec.apiKey'

export interface AuthTokenProvider {
  getToken(): Promise<string | null>
  setToken(token: string | null): Promise<void>
  clearToken(): Promise<void>
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

export class BrowserAuthTokenProvider implements AuthTokenProvider {
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

export const browserAuthTokenProvider = new BrowserAuthTokenProvider()

export async function resolveAuthToken(
  explicitApiKey: ApiRequestOptions['apiKey'],
  provider: AuthTokenProvider = browserAuthTokenProvider,
): Promise<string | null> {
  if (explicitApiKey !== undefined) {
    return normalizeAuthToken(explicitApiKey)
  }

  return provider.getToken()
}
