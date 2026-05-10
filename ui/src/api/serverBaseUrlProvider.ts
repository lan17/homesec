export const BROWSER_SERVER_BASE_URL_STORAGE_KEY = 'homesec.serverBaseUrl'

export interface ServerBaseUrlProvider {
  getBaseUrl(): Promise<string | null>
  setBaseUrl(value: string | null): Promise<void>
  clearBaseUrl(): Promise<void>
}

export interface ClientServerBaseUrlProvider extends ServerBaseUrlProvider {
  getBaseUrlSync(): string | null
}

type ServerBaseUrlStorage = Pick<Storage, 'getItem' | 'removeItem' | 'setItem'>

function getWindowSessionStorage(): ServerBaseUrlStorage | null {
  if (typeof window === 'undefined') {
    return null
  }

  try {
    return window.sessionStorage
  } catch {
    return null
  }
}

export function normalizeServerBaseUrl(value: string | null | undefined): string | null {
  const trimmed = value?.trim() ?? ''
  if (trimmed.length === 0) {
    return null
  }

  return trimmed.replace(/\/+$/, '')
}

export class BrowserServerBaseUrlProvider implements ClientServerBaseUrlProvider {
  private readonly fallbackBaseUrl: string | null
  private readonly getStorage: () => ServerBaseUrlStorage | null
  private readonly storageKey: string

  constructor(
    fallbackBaseUrl: string | null | undefined,
    getStorage: () => ServerBaseUrlStorage | null = getWindowSessionStorage,
    storageKey = BROWSER_SERVER_BASE_URL_STORAGE_KEY,
  ) {
    this.fallbackBaseUrl = normalizeServerBaseUrl(fallbackBaseUrl)
    this.getStorage = getStorage
    this.storageKey = storageKey
  }

  getBaseUrlSync(): string | null {
    const storage = this.getStorage()
    const stored = storage ? normalizeServerBaseUrl(storage.getItem(this.storageKey)) : null
    return stored ?? this.fallbackBaseUrl
  }

  setBaseUrlSync(value: string | null): void {
    const storage = this.getStorage()
    if (!storage) {
      return
    }

    const normalized = normalizeServerBaseUrl(value)
    if (!normalized) {
      storage.removeItem(this.storageKey)
      return
    }

    storage.setItem(this.storageKey, normalized)
  }

  clearBaseUrlSync(): void {
    const storage = this.getStorage()
    if (!storage) {
      return
    }

    storage.removeItem(this.storageKey)
  }

  async getBaseUrl(): Promise<string | null> {
    return this.getBaseUrlSync()
  }

  async setBaseUrl(value: string | null): Promise<void> {
    this.setBaseUrlSync(value)
  }

  async clearBaseUrl(): Promise<void> {
    this.clearBaseUrlSync()
  }
}

export function createBrowserServerBaseUrlProvider(
  fallbackBaseUrl: string | null | undefined,
): BrowserServerBaseUrlProvider {
  return new BrowserServerBaseUrlProvider(fallbackBaseUrl)
}
