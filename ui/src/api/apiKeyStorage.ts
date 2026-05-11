import type { ApiRequestOptions } from './generated/client'
import {
  BROWSER_AUTH_TOKEN_STORAGE_KEY,
  clearRuntimeAuthSessionReady,
  markRuntimeAuthSessionReady,
  normalizeAuthToken,
  runtimeAuthTokenProvider,
} from './tokenProvider'

export const API_KEY_STORAGE_KEY = BROWSER_AUTH_TOKEN_STORAGE_KEY

export function saveApiKey(apiKey: string): void {
  runtimeAuthTokenProvider.setTokenSync(apiKey)
  if (getStoredApiKey()) {
    markRuntimeAuthSessionReady()
  }
}

export function getStoredApiKey(): string | null {
  return runtimeAuthTokenProvider.getTokenSync()
}

export function hasStoredApiKey(): boolean {
  return getStoredApiKey() !== null
}

export function clearApiKey(): void {
  runtimeAuthTokenProvider.clearTokenSync()
  clearRuntimeAuthSessionReady()
}

export function resolveApiKey(explicitApiKey: ApiRequestOptions['apiKey']): string | null {
  if (explicitApiKey !== undefined) {
    return normalizeAuthToken(explicitApiKey)
  }

  return getStoredApiKey()
}
