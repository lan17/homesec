// @vitest-environment jsdom

import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type {
  PermissionStatus,
  RegistrationError,
  Token,
} from '@capacitor/push-notifications'
import type { PluginListenerHandle } from '@capacitor/core'

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

describe('native push registration', () => {
  beforeEach(() => {
    resetNativePushRegistrationForTests()
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

  it('does not cache denied permission as a completed registration', async () => {
    // Given: The first startup sees denied permission and a later startup has permission
    const deniedPushNotifications = createPushAdapter({ initialPermission: 'denied' })
    const grantedPushNotifications = createPushAdapter()
    const devicePlugin = createDevicePlugin()
    const client = createRegistrationClient()

    const { rerender } = renderHook(
      ({ pushNotifications }) =>
        useNativePushRegistration({
          client,
          devicePlugin,
          enabled: true,
          isIOSNative: () => true,
          pushNotifications,
          registrationKey: 'same-device',
        }),
      { initialProps: { pushNotifications: deniedPushNotifications } },
    )

    await waitFor(() => {
      expect(deniedPushNotifications.checkPermissions).toHaveBeenCalledTimes(1)
    })

    // When: The hook runs again for the same device key after permission becomes available
    await act(async () => {
      rerender({ pushNotifications: grantedPushNotifications })
    })

    // Then: The second attempt is allowed to register the device
    await waitFor(() => {
      expect(client.registerMobileDevice).toHaveBeenCalledTimes(1)
    })
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
