import { registerPlugin } from '@capacitor/core'

export interface HomeSecAuthStoredValue {
  value: string | null
}

export interface HomeSecAuthStoredFlag {
  value: boolean
}

export interface HomeSecAuthPlugin {
  getServerBaseUrl(): Promise<HomeSecAuthStoredValue>
  setServerBaseUrl(input: { value: string }): Promise<void>
  clearServerBaseUrl(): Promise<void>
  getApiToken(): Promise<HomeSecAuthStoredValue>
  setApiToken(input: { value: string }): Promise<void>
  clearApiToken(): Promise<void>
  getAuthDisabledReady(): Promise<HomeSecAuthStoredFlag>
  setAuthDisabledReady(input: { value: boolean }): Promise<void>
  clearAuthDisabledReady(): Promise<void>
}

export const homeSecAuthPlugin = registerPlugin<HomeSecAuthPlugin>('HomeSecAuth')
