import { registerPlugin } from '@capacitor/core'

export type HomeSecAPNSEnvironment = 'sandbox' | 'production'

export interface HomeSecDeviceRegistrationInfo {
  apnsEnvironment: HomeSecAPNSEnvironment
  appVersion: string | null
  bundleId: string
  deviceName: string | null
}

export interface HomeSecDevicePlugin {
  getRegistrationInfo(): Promise<HomeSecDeviceRegistrationInfo>
}

export const homeSecDevicePlugin = registerPlugin<HomeSecDevicePlugin>('HomeSecDevice')
