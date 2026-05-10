import { browserServerBaseUrlProvider } from './client'
import type { ClientServerBaseUrlProvider } from './serverBaseUrlProvider'

export interface ApiRuntimeConfig {
  serverBaseUrl?: string | null
}

export interface ApiRuntimeConfigSource {
  loadRuntimeConfig(): ApiRuntimeConfig | Promise<ApiRuntimeConfig>
}

declare global {
  interface Window {
    __HOMESEC_RUNTIME_CONFIG__?: ApiRuntimeConfig
  }
}

export class WindowApiRuntimeConfigSource implements ApiRuntimeConfigSource {
  loadRuntimeConfig(): ApiRuntimeConfig {
    if (typeof window === 'undefined') {
      return {}
    }

    return window.__HOMESEC_RUNTIME_CONFIG__ ?? {}
  }
}

export interface InitializeApiRuntimeConfigOptions {
  runtimeConfigSource?: ApiRuntimeConfigSource
  serverBaseUrlProvider?: ClientServerBaseUrlProvider
}

export async function initializeApiRuntimeConfig({
  runtimeConfigSource = new WindowApiRuntimeConfigSource(),
  serverBaseUrlProvider = browserServerBaseUrlProvider,
}: InitializeApiRuntimeConfigOptions = {}): Promise<void> {
  const config = await runtimeConfigSource.loadRuntimeConfig()
  if (Object.prototype.hasOwnProperty.call(config, 'serverBaseUrl')) {
    await serverBaseUrlProvider.setBaseUrl(config.serverBaseUrl ?? null)
  }
}
