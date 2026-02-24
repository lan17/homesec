import type { ApiRequestOptions } from './generated/client'

const API_KEY_STORAGE_KEY = 'homesec.apiKey'

export function saveApiKey(apiKey: string): void {
  if (typeof window === 'undefined') {
    return
  }

  window.sessionStorage.setItem(API_KEY_STORAGE_KEY, apiKey)
}

export function getStoredApiKey(): string | null {
  if (typeof window === 'undefined') {
    return null
  }
  return window.sessionStorage.getItem(API_KEY_STORAGE_KEY)
}

export function hasStoredApiKey(): boolean {
  const value = getStoredApiKey()
  return Boolean(value && value.trim().length > 0)
}

export function clearApiKey(): void {
  if (typeof window === 'undefined') {
    return
  }

  window.sessionStorage.removeItem(API_KEY_STORAGE_KEY)
}

export function resolveApiKey(explicitApiKey: ApiRequestOptions['apiKey']): string | null {
  if (explicitApiKey !== undefined) {
    return explicitApiKey
  }

  return getStoredApiKey()
}
