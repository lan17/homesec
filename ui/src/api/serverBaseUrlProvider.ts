import { homeSecAuthPlugin, type HomeSecAuthPlugin } from './homeSecAuthPlugin'

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
    const stored = storage?.getItem(this.storageKey)
    if (stored !== null && stored !== undefined) {
      return normalizeServerBaseUrl(stored)
    }

    return this.fallbackBaseUrl
  }

  setBaseUrlSync(value: string | null): void {
    const storage = this.getStorage()
    if (!storage) {
      return
    }

    if (value === null) {
      storage.removeItem(this.storageKey)
      return
    }

    storage.setItem(this.storageKey, normalizeServerBaseUrl(value) ?? '')
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

export class NativeServerBaseUrlProvider implements ClientServerBaseUrlProvider {
  private baseUrl: string | null = null
  private hydrated = false
  private readonly plugin: HomeSecAuthPlugin

  constructor(plugin: HomeSecAuthPlugin = homeSecAuthPlugin) {
    this.plugin = plugin
  }

  async hydrate(): Promise<void> {
    const result = await this.plugin.getServerBaseUrl()
    this.baseUrl = normalizeServerBaseUrl(result.value)
    this.hydrated = true
  }

  getBaseUrlSync(): string | null {
    return this.baseUrl
  }

  async getBaseUrl(): Promise<string | null> {
    if (!this.hydrated) {
      await this.hydrate()
    }

    return this.getBaseUrlSync()
  }

  async setBaseUrl(value: string | null): Promise<void> {
    const normalized = normalizeServerBaseUrl(value)
    if (normalized) {
      await this.plugin.setServerBaseUrl({ value: normalized })
    } else {
      await this.plugin.clearServerBaseUrl()
    }

    this.baseUrl = normalized
    this.hydrated = true
  }

  async clearBaseUrl(): Promise<void> {
    await this.plugin.clearServerBaseUrl()
    this.baseUrl = null
    this.hydrated = true
  }
}

export function createBrowserServerBaseUrlProvider(
  fallbackBaseUrl: string | null | undefined,
): BrowserServerBaseUrlProvider {
  return new BrowserServerBaseUrlProvider(fallbackBaseUrl)
}
