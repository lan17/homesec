import type { ApiRequestOptions } from './generated/client'
import {
  BROWSER_AUTH_TOKEN_STORAGE_KEY,
  clearPersistedRuntimeAuthSessionReady,
  normalizeAuthToken,
  persistRuntimeAuthSessionReady,
  runtimeAuthTokenProvider,
} from './tokenProvider'

export const API_KEY_STORAGE_KEY = BROWSER_AUTH_TOKEN_STORAGE_KEY

export async function saveApiKey(apiKey: string): Promise<void> {
  await runtimeAuthTokenProvider.setToken(apiKey)
  if (getStoredApiKey()) {
    await persistRuntimeAuthSessionReady()
  }
}

export function getStoredApiKey(): string | null {
  return runtimeAuthTokenProvider.getTokenSync()
}

export function hasStoredApiKey(): boolean {
  return getStoredApiKey() !== null
}

export async function clearApiKey(): Promise<void> {
  await runtimeAuthTokenProvider.clearToken()
  await clearPersistedRuntimeAuthSessionReady()
}

export function resolveApiKey(explicitApiKey: ApiRequestOptions['apiKey']): string | null {
  if (explicitApiKey !== undefined) {
    return normalizeAuthToken(explicitApiKey)
  }

  return getStoredApiKey()
}
