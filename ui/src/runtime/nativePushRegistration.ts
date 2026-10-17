import { useEffect, useRef } from 'react'
import { PushNotifications } from '@capacitor/push-notifications'
import type {
  PushNotificationsPlugin,
  RegistrationError,
  Token,
} from '@capacitor/push-notifications'
import type { PluginListenerHandle } from '@capacitor/core'

import { apiClient, type HomeSecApiClient } from '../api/client'
import type { MobileDeviceRegisterRequest } from '../api/generated/types'
import { homeSecDevicePlugin, type HomeSecDevicePlugin } from './homeSecDevicePlugin'
import { useNativeAppLifecycleState } from './nativeAppLifecycle'
import { isIOSNativeApp } from './nativeRuntime'

type MobileDeviceRegistrationClient = Pick<HomeSecApiClient, 'registerMobileDevice'>
type NativePushNotifications = Pick<
  PushNotificationsPlugin,
  'addListener' | 'checkPermissions' | 'register' | 'requestPermissions'
>

type NativePushRegistrationStatus = 'failed' | 'registered' | 'skipped'

export interface NativePushRegistrationResult {
  reason?: string
  status: NativePushRegistrationStatus
}

export interface NativePushRegistrationOptions {
  client?: MobileDeviceRegistrationClient
  devicePlugin?: HomeSecDevicePlugin
  isIOSNative?: () => boolean
  pushNotifications?: NativePushNotifications
  signal?: AbortSignal
  timeoutMs?: number
}

export interface UseNativePushRegistrationOptions extends NativePushRegistrationOptions {
  enabled: boolean
  registrationKey: string
}

const completedRegistrationKeys = new Set<string>()
const inFlightRegistrations = new Map<string, Promise<NativePushRegistrationResult>>()
const REGISTRATION_CANCELLED = 'registration_cancelled'

function describeError(error: unknown): string {
  if (error instanceof Error && error.message.trim().length > 0) {
    return error.message
  }
  if (typeof error === 'string' && error.trim().length > 0) {
    return error
  }
  return 'APNs registration failed'
}

function nullableTrimmed(value: string | null | undefined): string | null {
  const normalized = value?.trim() ?? ''
  return normalized.length > 0 ? normalized : null
}

function shouldRequestPermission(receive: string): boolean {
  return receive === 'prompt' || receive === 'prompt-with-rationale'
}

async function hasPushPermission(pushNotifications: NativePushNotifications): Promise<boolean> {
  let permission = await pushNotifications.checkPermissions()
  if (shouldRequestPermission(permission.receive)) {
    permission = await pushNotifications.requestPermissions()
  }
  return permission.receive === 'granted'
}

async function requestAPNSToken(
  pushNotifications: NativePushNotifications,
  timeoutMs: number,
  signal?: AbortSignal,
): Promise<string> {
  let registrationHandle: PluginListenerHandle | null = null
  let registrationErrorHandle: PluginListenerHandle | null = null

  return await new Promise<string>((resolve, reject) => {
    let settled = false
    const timer = globalThis.setTimeout(() => {
      rejectOnce(new Error('APNs registration timed out'))
    }, timeoutMs)

    function cleanup(): void {
      globalThis.clearTimeout(timer)
      signal?.removeEventListener('abort', onAbort)
      void registrationHandle?.remove()
      void registrationErrorHandle?.remove()
    }

    function resolveOnce(token: Token): void {
      if (settled) {
        return
      }
      const value = token.value.trim()
      if (!value) {
        rejectOnce(new Error('APNs registration returned an empty token'))
        return
      }
      settled = true
      cleanup()
      resolve(value)
    }

    function rejectOnce(error: Error): void {
      if (settled) {
        return
      }
      settled = true
      cleanup()
      reject(error)
    }

    function onAbort(): void {
      rejectOnce(new Error(REGISTRATION_CANCELLED))
    }

    signal?.addEventListener('abort', onAbort, { once: true })
    if (signal?.aborted) {
      onAbort()
      return
    }

    async function register(): Promise<void> {
      registrationHandle = await pushNotifications.addListener('registration', resolveOnce)
      if (settled) {
        await registrationHandle.remove()
        return
      }
      registrationErrorHandle = await pushNotifications.addListener(
        'registrationError',
        (error: RegistrationError) => {
          rejectOnce(new Error(error.error || 'APNs registration failed'))
        },
      )
      if (settled) {
        await registrationErrorHandle.remove()
        return
      }
      await pushNotifications.register()
    }

    void register().catch((error: unknown) => rejectOnce(new Error(describeError(error))))
  })
}

function buildMobileDeviceRegistration(
  apnsToken: string,
  info: Awaited<ReturnType<HomeSecDevicePlugin['getRegistrationInfo']>>,
): MobileDeviceRegisterRequest {
  return {
    platform: 'ios',
    apns_token: apnsToken,
    environment: info.apnsEnvironment,
    bundle_id: info.bundleId,
    device_name: nullableTrimmed(info.deviceName),
    app_version: nullableTrimmed(info.appVersion),
    capabilities: {
      deep_links: true,
      rich_notifications: false,
    },
  }
}

export async function registerNativePushDevice(
  options: NativePushRegistrationOptions = {},
): Promise<NativePushRegistrationResult> {
  const isIOSNative = options.isIOSNative ?? isIOSNativeApp
  if (!isIOSNative()) {
    return { status: 'skipped', reason: 'not_ios_native' }
  }

  try {
    if (options.signal?.aborted) {
      return { status: 'skipped', reason: REGISTRATION_CANCELLED }
    }
    const pushNotifications = options.pushNotifications ?? PushNotifications
    const permitted = await hasPushPermission(pushNotifications)
    if (options.signal?.aborted) {
      return { status: 'skipped', reason: REGISTRATION_CANCELLED }
    }
    if (!permitted) {
      return { status: 'skipped', reason: 'permission_not_granted' }
    }

    const devicePlugin = options.devicePlugin ?? homeSecDevicePlugin
    const [info, apnsToken] = await Promise.all([
      devicePlugin.getRegistrationInfo(),
      requestAPNSToken(pushNotifications, options.timeoutMs ?? 30_000, options.signal),
    ])
    if (options.signal?.aborted) {
      return { status: 'skipped', reason: REGISTRATION_CANCELLED }
    }
    const client = options.client ?? apiClient
    const payload = buildMobileDeviceRegistration(apnsToken, info)
    if (options.signal) {
      await client.registerMobileDevice(payload, { signal: options.signal })
    } else {
      await client.registerMobileDevice(payload)
    }
    if (options.signal?.aborted) {
      return { status: 'skipped', reason: REGISTRATION_CANCELLED }
    }
    return { status: 'registered' }
  } catch (error) {
    if (options.signal?.aborted) {
      return { status: 'skipped', reason: REGISTRATION_CANCELLED }
    }
    const reason = describeError(error)
    if (reason.includes('plugin is not implemented on web')) {
      return { status: 'skipped', reason: 'push_plugin_unavailable' }
    }
    return { status: 'failed', reason }
  }
}

async function registerOnceForKey(
  registrationKey: string,
  options: NativePushRegistrationOptions,
): Promise<NativePushRegistrationResult> {
  if (completedRegistrationKeys.has(registrationKey)) {
    return Promise.resolve({ status: 'skipped', reason: 'already_registered' })
  }

  const existing = inFlightRegistrations.get(registrationKey)
  if (existing) {
    const result = await existing
    // A remount can arrive before the obsolete owner's cancellation settles.
    if (result.reason === REGISTRATION_CANCELLED && !options.signal?.aborted) {
      return registerOnceForKey(registrationKey, options)
    }
    return result
  }

  const registration = registerNativePushDevice(options).then((result) => {
    inFlightRegistrations.delete(registrationKey)
    if (result.status === 'registered' && !options.signal?.aborted) {
      completedRegistrationKeys.add(registrationKey)
    }
    return result
  })
  inFlightRegistrations.set(registrationKey, registration)
  return registration
}

export function resetNativePushRegistrationForTests(): void {
  completedRegistrationKeys.clear()
  inFlightRegistrations.clear()
}

export function useNativePushRegistration({
  client,
  devicePlugin,
  enabled,
  isIOSNative,
  pushNotifications,
  registrationKey,
  timeoutMs,
}: UseNativePushRegistrationOptions): void {
  const nativeLifecycle = useNativeAppLifecycleState()
  const controllerRef = useRef<AbortController | null>(null)

  // Native pause/resume shares pending registration. Cancel only when the
  // owning setup/auth context changes or the hook unmounts.
  useEffect(() => {
    const controller = new AbortController()
    controllerRef.current = controller
    return () => { controller.abort() }
  }, [client, devicePlugin, enabled, isIOSNative, pushNotifications, registrationKey, timeoutMs])

  useEffect(() => {
    if (!enabled || !nativeLifecycle.isActive || nativeLifecycle.isBackgrounded) {
      return
    }

    const controller = controllerRef.current
    if (!controller) {
      return
    }
    void registerOnceForKey(registrationKey, {
      client,
      devicePlugin,
      isIOSNative,
      pushNotifications,
      signal: controller.signal,
      timeoutMs,
    }).then((result) => {
      if (!controller.signal.aborted && result.status === 'failed') {
        console.warn(`iOS push registration failed: ${result.reason ?? 'unknown error'}`)
      }
    })
  }, [
    client,
    devicePlugin,
    enabled,
    isIOSNative,
    nativeLifecycle.isActive,
    nativeLifecycle.isBackgrounded,
    nativeLifecycle.resumeCount,
    pushNotifications,
    registrationKey,
    timeoutMs,
  ])
}
