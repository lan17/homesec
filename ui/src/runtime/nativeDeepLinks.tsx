import { useCallback, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { App } from '@capacitor/app'
import { PushNotifications } from '@capacitor/push-notifications'
import type { PluginListenerHandle } from '@capacitor/core'
import type { ActionPerformed } from '@capacitor/push-notifications'

import { parseNativeDeepLinkRoute, parseNativeNotificationRoute } from './nativeDeepLinkRoutes'
import { isIOSNativeApp } from './nativeRuntime'

interface NativeDeepLinkEvent {
  url?: string | null
}

interface NativeDeepLinkApp {
  getLaunchUrl: () => Promise<NativeDeepLinkEvent | null | undefined>
  addListener: (
    eventName: 'appUrlOpen',
    listenerFunc: (event: NativeDeepLinkEvent) => void,
  ) => Promise<PluginListenerHandle>
}

interface NativePushNotificationActions {
  addListener: (
    eventName: 'pushNotificationActionPerformed',
    listenerFunc: (notification: ActionPerformed) => void,
  ) => Promise<PluginListenerHandle>
}

export function NativeDeepLinkRouter({
  app = App,
  pushNotifications = PushNotifications,
}: {
  app?: NativeDeepLinkApp
  pushNotifications?: NativePushNotificationActions
}) {
  const navigate = useNavigate()
  const navigateRef = useRef(navigate)
  const isIOS = isIOSNativeApp()

  useEffect(() => {
    navigateRef.current = navigate
  }, [navigate])

  const navigateToDeepLink = useCallback((
    rawUrl: string | null | undefined,
    options: { replace: boolean },
  ) => {
    if (!rawUrl) {
      return
    }
    const route = parseNativeDeepLinkRoute(rawUrl)
    if (route === null) {
      return
    }
    navigateRef.current(route, { replace: options.replace })
  }, [])

  const navigateToNotificationRoute = useCallback((
    action: ActionPerformed,
    options: { replace: boolean },
  ) => {
    const route = parseNativeNotificationRoute(action.notification.data)
    if (route === null) {
      return
    }
    navigateRef.current(route, { replace: options.replace })
  }, [])

  useEffect(() => {
    if (!isIOS) {
      return
    }

    let cancelled = false
    const handles: PluginListenerHandle[] = []

    void app.getLaunchUrl()
      .then((event) => {
        if (!cancelled) {
          navigateToDeepLink(event?.url, { replace: true })
        }
      })
      .catch(() => {})

    void app.addListener('appUrlOpen', (event) => {
      if (cancelled) {
        return
      }
      navigateToDeepLink(event.url, { replace: false })
    })
      .then((nextHandle) => {
        if (cancelled) {
          void nextHandle.remove()
          return
        }
        handles.push(nextHandle)
      })
      .catch(() => {})

    void pushNotifications.addListener('pushNotificationActionPerformed', (action) => {
      if (cancelled) {
        return
      }
      navigateToNotificationRoute(action, { replace: false })
    })
      .then((nextHandle) => {
        if (cancelled) {
          void nextHandle.remove()
          return
        }
        handles.push(nextHandle)
      })
      .catch(() => {})

    return () => {
      cancelled = true
      for (const handle of handles) {
        void handle.remove()
      }
    }
  }, [app, isIOS, navigateToDeepLink, navigateToNotificationRoute, pushNotifications])

  return null
}
