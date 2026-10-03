// @vitest-environment jsdom

import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type {
  PermissionStatus,
  RegistrationError,
  Token,
} from '@capacitor/push-notifications'
import type { PluginListenerHandle } from '@capacitor/core'

const nativeRuntimeMock = vi.hoisted(() => ({
  isIOSNativeApp: vi.fn(() => false),
}))

const nativeAppMock = vi.hoisted(() => {
  const listeners = new Map<string, Set<(event?: { isActive: boolean }) => void>>()
  return {
    listeners,
    getState: vi.fn(async () => ({ isActive: true })),
    addListener: vi.fn(async (
      eventName: string,
      listener: (event?: { isActive: boolean }) => void,
    ) => {
      const eventListeners = listeners.get(eventName) ?? new Set()
      eventListeners.add(listener)
      listeners.set(eventName, eventListeners)
      return {
        remove: vi.fn(async () => { eventListeners.delete(listener) }),
      }
    }),
  }
})

vi.mock('./nativeRuntime', () => ({
  isIOSNativeApp: () => nativeRuntimeMock.isIOSNativeApp(),
}))

vi.mock('@capacitor/app', () => ({ App: nativeAppMock }))

import {
  BROWSER_AUTH_TOKEN_STORAGE_KEY,
  BROWSER_SERVER_BASE_URL_STORAGE_KEY,
} from '../api/client'
import type { MobileDeviceRegisterRequest } from '../api/generated/types'
import type { HomeSecDevicePlugin } from './homeSecDevicePlugin'
import {
  registerNativePushDevice,
  resetNativePushRegistrationForTests,
  useNativePushRegistration,
  type NativePushRegistrationOptions,
} from './nativePushRegistration'

type PushAdapter = NonNullable<NativePushRegistrationOptions['pushNotifications']>
type PushRegistrationMode = 'error' | 'success'
type TestStorage = Pick<Storage, 'getItem' | 'removeItem' | 'setItem'> & {
  values: Map<string, string>
}

function listenerHandle(): PluginListenerHandle {
  return {
    remove: vi.fn(async () => {}),
  }
}

function createPushAdapter({
  initialPermission = 'granted',
  mode = 'success',
  requestedPermission = 'granted',
}: {
  initialPermission?: PermissionStatus['receive']
  mode?: PushRegistrationMode
  requestedPermission?: PermissionStatus['receive']
} = {}): PushAdapter {
  const registrationListeners: Array<(token: Token) => void> = []
  const registrationErrorListeners: Array<(error: RegistrationError) => void> = []

  return {
    addListener: vi.fn(async (eventName: string, listener: unknown) => {
      if (eventName === 'registration') {
        registrationListeners.push(listener as (token: Token) => void)
      }
      if (eventName === 'registrationError') {
        registrationErrorListeners.push(listener as (error: RegistrationError) => void)
      }
      return listenerHandle()
    }),
    checkPermissions: vi.fn(async () => ({ receive: initialPermission })),
    register: vi.fn(async () => {
      queueMicrotask(() => {
        if (mode === 'success') {
          registrationListeners.forEach((listener) => listener({ value: 'apns-token-123' }))
          return
        }
        registrationErrorListeners.forEach((listener) =>
          listener({ error: 'registration rejected' }),
        )
      })
    }),
    requestPermissions: vi.fn(async () => ({ receive: requestedPermission })),
  }
}

function createDevicePlugin(): HomeSecDevicePlugin {
  return {
    getRegistrationInfo: vi.fn(async () => ({
      apnsEnvironment: 'sandbox' as const,
      appVersion: '1.0.0',
      bundleId: 'com.levneiman.homesec',
      deviceName: "Lev's iPhone",
    })),
  }
}

function createRegistrationClient() {
  return {
    registerMobileDevice: vi.fn(async (payload: MobileDeviceRegisterRequest) => ({
      id: 'dev_1',
      platform: 'ios' as const,
      environment: payload.environment,
      bundle_id: payload.bundle_id,
      device_name: payload.device_name ?? null,
      app_version: payload.app_version ?? null,
      capabilities: payload.capabilities ?? {
        deep_links: true,
        rich_notifications: false,
      },
      enabled: true,
      token_fingerprint: 'abcdef123456',
      created_at: '2026-06-14T00:00:00Z',
      updated_at: '2026-06-14T00:00:00Z',
      last_seen_at: '2026-06-14T00:00:00Z',
      last_push_at: null,
      last_push_error: null,
      httpStatus: 201,
    })),
  }
}

function installWindowSessionStorageMock(): TestStorage {
  const storage: TestStorage = {
    values: new Map<string, string>(),
    getItem: (key: string): string | null => storage.values.get(key) ?? null,
    setItem: (key: string, value: string): void => {
      storage.values.set(key, value)
    },
    removeItem: (key: string): void => {
      storage.values.delete(key)
    },
  }
  vi.stubGlobal('window', { sessionStorage: storage })
  return storage
}

function mobileDeviceResponse() {
  return {
    id: 'dev_1',
    platform: 'ios' as const,
    environment: 'sandbox' as const,
    bundle_id: 'com.levneiman.homesec',
    device_name: "Lev's iPhone",
    app_version: '1.0.0',
    capabilities: {
      deep_links: true,
      rich_notifications: false,
    },
    enabled: true,
    token_fingerprint: 'abcdef123456',
    created_at: '2026-06-14T00:00:00Z',
    updated_at: '2026-06-14T00:00:00Z',
    last_seen_at: '2026-06-14T00:00:00Z',
    last_push_at: null,
    last_push_error: null,
  }
}

async function backgroundNativeApp(): Promise<void> {
  await act(async () => {
    nativeAppMock.listeners.get('appStateChange')?.forEach((listener) => {
      listener({ isActive: false })
    })
    nativeAppMock.listeners.get('pause')?.forEach((listener) => { listener() })
  })
}

async function resumeNativeApp(): Promise<void> {
  await act(async () => {
    nativeAppMock.listeners.get('appStateChange')?.forEach((listener) => {
      listener({ isActive: true })
    })
    nativeAppMock.listeners.get('resume')?.forEach((listener) => { listener() })
  })
}

describe('native push registration', () => {
  beforeEach(() => {
    resetNativePushRegistrationForTests()
    nativeRuntimeMock.isIOSNativeApp.mockReturnValue(true)
    nativeAppMock.listeners.clear()
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it('skips registration outside iOS native mode', async () => {
    // Given: The app is running outside the iOS native shell
    const pushNotifications = createPushAdapter()
    const client = createRegistrationClient()

    // When: Native push registration runs
    const result = await registerNativePushDevice({
      client,
      isIOSNative: () => false,
      pushNotifications,
    })

    // Then: No permission prompt, APNs registration, or backend request is attempted
    expect(result).toEqual({ status: 'skipped', reason: 'not_ios_native' })
    expect(pushNotifications.checkPermissions).not.toHaveBeenCalled()
    expect(client.registerMobileDevice).not.toHaveBeenCalled()
  })

  it('posts the APNs registration result to HomeSec when permission is granted', async () => {
    // Given: iOS has notification permission and APNs returns a token
    const pushNotifications = createPushAdapter()
    const devicePlugin = createDevicePlugin()
    const client = createRegistrationClient()

    // When: Native push registration runs
    const result = await registerNativePushDevice({
      client,
      devicePlugin,
      isIOSNative: () => true,
      pushNotifications,
    })

    // Then: The device is registered with redacted app/device metadata and current capabilities
    expect(result).toEqual({ status: 'registered' })
    expect(client.registerMobileDevice).toHaveBeenCalledWith({
      platform: 'ios',
      apns_token: 'apns-token-123',
      environment: 'sandbox',
      bundle_id: 'com.levneiman.homesec',
      device_name: "Lev's iPhone",
      app_version: '1.0.0',
      capabilities: {
        deep_links: true,
        rich_notifications: false,
      },
    })
  })

  it('uses the configured runtime API client when no client is injected', async () => {
    // Given: Native setup has stored a server URL and API token in the runtime providers
    const storage = installWindowSessionStorageMock()
    storage.setItem(BROWSER_SERVER_BASE_URL_STORAGE_KEY, 'http://192.168.1.10:8081')
    storage.setItem(BROWSER_AUTH_TOKEN_STORAGE_KEY, 'secret-token')
    const pushNotifications = createPushAdapter()
    const devicePlugin = createDevicePlugin()
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify(mobileDeviceResponse()), {
        status: 201,
        headers: { 'content-type': 'application/json' },
      }),
    )

    // When: Native push registration runs without an injected test client
    const result = await registerNativePushDevice({
      devicePlugin,
      isIOSNative: () => true,
      pushNotifications,
    })

    // Then: Registration posts through the runtime-configured HomeSec origin with auth
    expect(result).toEqual({ status: 'registered' })
    expect(fetchSpy).toHaveBeenCalledTimes(1)
    expect(fetchSpy.mock.calls[0]?.[0]).toBe(
      'http://192.168.1.10:8081/api/v1/mobile/devices',
    )
    expect(fetchSpy.mock.calls[0]?.[1]).toMatchObject({
      headers: {
        Accept: 'application/json',
        Authorization: 'Bearer secret-token',
        'content-type': 'application/json',
      },
    })
  })

  it('requests permission once and skips backend registration when denied', async () => {
    // Given: iOS has not prompted yet and the user denies notification permission
    const pushNotifications = createPushAdapter({
      initialPermission: 'prompt',
      requestedPermission: 'denied',
    })
    const client = createRegistrationClient()

    // When: Native push registration runs
    const result = await registerNativePushDevice({
      client,
      isIOSNative: () => true,
      pushNotifications,
    })

    // Then: The denial is handled without APNs registration or a backend request
    expect(result).toEqual({ status: 'skipped', reason: 'permission_not_granted' })
    expect(pushNotifications.requestPermissions).toHaveBeenCalledTimes(1)
    expect(pushNotifications.register).not.toHaveBeenCalled()
    expect(client.registerMobileDevice).not.toHaveBeenCalled()
  })

  it('registers on native resume after notification permission is granted in Settings', async () => {
    // Given: The running app has notification permission denied
    const pushNotifications = createPushAdapter({ initialPermission: 'denied' })
    const devicePlugin = createDevicePlugin()
    const client = createRegistrationClient()
    const options = {
      client,
      devicePlugin,
      enabled: true,
      pushNotifications,
      registrationKey: 'same-device',
    }
    renderHook(() => useNativePushRegistration(options))

    await waitFor(() => {
      expect(pushNotifications.checkPermissions).toHaveBeenCalledTimes(1)
    })
    expect(client.registerMobileDevice).not.toHaveBeenCalled()

    // When: The user grants permission in Settings and resumes the same app instance
    await backgroundNativeApp()
    vi.mocked(pushNotifications.checkPermissions).mockResolvedValue({ receive: 'granted' })
    await resumeNativeApp()

    // Then: The unchanged push and device plugins register without restarting the app
    await waitFor(() => {
      expect(client.registerMobileDevice).toHaveBeenCalledTimes(1)
    })
    expect(pushNotifications.checkPermissions).toHaveBeenCalledTimes(2)
    expect(pushNotifications.requestPermissions).not.toHaveBeenCalled()
    expect(pushNotifications.register).toHaveBeenCalledTimes(1)
  })

  it('retries a failed backend registration on native resume', async () => {
    // Given: Startup gets an APNs token but the HomeSec server is unreachable
    const pushNotifications = createPushAdapter()
    const devicePlugin = createDevicePlugin()
    const client = createRegistrationClient()
    client.registerMobileDevice.mockRejectedValueOnce(new Error('HomeSec server unreachable'))
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const options = {
      client,
      devicePlugin,
      enabled: true,
      pushNotifications,
      registrationKey: 'same-device',
    }
    renderHook(() => useNativePushRegistration(options))

    await waitFor(() => {
      expect(warn).toHaveBeenCalledWith('iOS push registration failed: HomeSec server unreachable')
    })

    // When: The server becomes reachable and the same app backgrounds then resumes
    await backgroundNativeApp()
    expect(client.registerMobileDevice).toHaveBeenCalledTimes(1)
    await resumeNativeApp()

    // Then: Registration retries with the original adapters and succeeds
    await waitFor(() => {
      expect(client.registerMobileDevice).toHaveBeenCalledTimes(2)
    })
    expect(warn).toHaveBeenCalledTimes(1)
    expect(pushNotifications.register).toHaveBeenCalledTimes(2)
  })

  it('does not duplicate a successful registration on native resume', async () => {
    // Given: The same app instance has successfully registered its device
    const pushNotifications = createPushAdapter()
    const devicePlugin = createDevicePlugin()
    const client = createRegistrationClient()
    const options = {
      client,
      devicePlugin,
      enabled: true,
      pushNotifications,
      registrationKey: 'same-device',
    }
    renderHook(() => useNativePushRegistration(options))
    await waitFor(() => {
      expect(client.registerMobileDevice).toHaveBeenCalledTimes(1)
    })

    // When: iOS backgrounds and resumes the app twice
    await backgroundNativeApp()
    await resumeNativeApp()
    await backgroundNativeApp()
    await resumeNativeApp()

    // Then: Permission and APNs/backend registration are not repeated after success
    expect(pushNotifications.checkPermissions).toHaveBeenCalledTimes(1)
    expect(pushNotifications.register).toHaveBeenCalledTimes(1)
    expect(client.registerMobileDevice).toHaveBeenCalledTimes(1)
  })

  it('shares an in-flight registration across native resumes', async () => {
    // Given: The backend registration response is pending for the running app
    const pushNotifications = createPushAdapter()
    const devicePlugin = createDevicePlugin()
    const client = createRegistrationClient()
    let resolveRegistration: (() => void) | undefined
    const registrationResponse = new Promise<Awaited<ReturnType<typeof client.registerMobileDevice>>>((resolve) => {
      resolveRegistration = () => resolve({ ...mobileDeviceResponse(), httpStatus: 201 })
    })
    client.registerMobileDevice.mockReturnValueOnce(registrationResponse)
    const options = {
      client,
      devicePlugin,
      enabled: true,
      pushNotifications,
      registrationKey: 'same-device',
    }
    renderHook(() => useNativePushRegistration(options))
    await waitFor(() => {
      expect(client.registerMobileDevice).toHaveBeenCalledTimes(1)
    })

    // When: The app resumes before the first request completes
    await backgroundNativeApp()
    await resumeNativeApp()
    await act(async () => { resolveRegistration?.() })

    // Then: The original request is shared and later resumes use its successful result
    await backgroundNativeApp()
    await resumeNativeApp()
    expect(pushNotifications.register).toHaveBeenCalledTimes(1)
    expect(client.registerMobileDevice).toHaveBeenCalledTimes(1)
  })

  it('handles APNs registration errors without posting a device', async () => {
    // Given: APNs registration fails after notification permission is granted
    const pushNotifications = createPushAdapter({ mode: 'error' })
    const devicePlugin = createDevicePlugin()
    const client = createRegistrationClient()

    // When: Native push registration runs
    const result = await registerNativePushDevice({
      client,
      devicePlugin,
      isIOSNative: () => true,
      pushNotifications,
    })

    // Then: The failure is reported and the raw token registration endpoint is not called
    expect(result).toEqual({ status: 'failed', reason: 'registration rejected' })
    expect(client.registerMobileDevice).not.toHaveBeenCalled()
  })
})
