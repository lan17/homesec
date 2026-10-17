import { beforeEach, describe, expect, it, vi } from 'vitest'

const capacitorMock = vi.hoisted(() => ({
  getPlatform: vi.fn(() => 'web'),
  isNativePlatform: vi.fn(() => false),
}))

vi.mock('@capacitor/core', () => ({
  Capacitor: capacitorMock,
}))

import { isIOSNativeApp, isNativeApp } from './nativeRuntime'

describe('native runtime detection', () => {
  beforeEach(() => {
    capacitorMock.getPlatform.mockReturnValue('web')
    capacitorMock.isNativePlatform.mockReturnValue(false)
  })

  it('reports browser mode as non-native', () => {
    // Given: Capacitor is running on the web platform
    capacitorMock.getPlatform.mockReturnValue('web')
    capacitorMock.isNativePlatform.mockReturnValue(false)

    // When: The app checks the runtime mode
    const native = isNativeApp()
    const iosNative = isIOSNativeApp()

    // Then: Browser mode is not treated as native iOS
    expect(native).toBe(false)
    expect(iosNative).toBe(false)
  })

  it('reports iOS Capacitor mode as native iOS', () => {
    // Given: Capacitor is running inside the iOS native shell
    capacitorMock.getPlatform.mockReturnValue('ios')
    capacitorMock.isNativePlatform.mockReturnValue(true)

    // When: The app checks the runtime mode
    const native = isNativeApp()
    const iosNative = isIOSNativeApp()

    // Then: The iOS native shell is detected
    expect(native).toBe(true)
    expect(iosNative).toBe(true)
  })

  it('does not report non-iOS native platforms as iOS', () => {
    // Given: Capacitor is running on a different native platform
    capacitorMock.getPlatform.mockReturnValue('android')
    capacitorMock.isNativePlatform.mockReturnValue(true)

    // When: The app checks the runtime mode
    const native = isNativeApp()
    const iosNative = isIOSNativeApp()

    // Then: Native and iOS-native detection remain distinct
    expect(native).toBe(true)
    expect(iosNative).toBe(false)
  })
})
