import type { ApiRequestOptions } from './generated/client'
import {
  BROWSER_AUTH_TOKEN_STORAGE_KEY,
  browserAuthTokenProvider,
  normalizeAuthToken,
} from './tokenProvider'

export const API_KEY_STORAGE_KEY = BROWSER_AUTH_TOKEN_STORAGE_KEY

export function saveApiKey(apiKey: string): void {
  browserAuthTokenProvider.setTokenSync(apiKey)
}

export function getStoredApiKey(): string | null {
  return browserAuthTokenProvider.getTokenSync()
}

export function hasStoredApiKey(): boolean {
  return getStoredApiKey() !== null
}

export function clearApiKey(): void {
  browserAuthTokenProvider.clearTokenSync()
}

export function resolveApiKey(explicitApiKey: ApiRequestOptions['apiKey']): string | null {
  if (explicitApiKey !== undefined) {
    return normalizeAuthToken(explicitApiKey)
  }

  return getStoredApiKey()
}
