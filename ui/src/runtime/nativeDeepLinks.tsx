import { useCallback, useEffect, useLayoutEffect, useRef } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
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
  const location = useLocation()
  const navigateRef = useRef(navigate)
  const locationRef = useRef(location)
  const isIOS = isIOSNativeApp()

  useLayoutEffect(() => {
    navigateRef.current = navigate
    locationRef.current = location
  }, [location, navigate])

  const navigateToRoute = useCallback((route: string, options: { replace: boolean }) => {
    const currentLocation = locationRef.current
    if (currentLocation.pathname === '/native-setup') {
      // Keep pending validation and credential writes owned by the mounted form.
      const state = currentLocation.state && typeof currentLocation.state === 'object'
        ? currentLocation.state
        : {}
      navigateRef.current('/native-setup', {
        replace: true,
        state: { ...state, nativeSetupReturnTo: route },
      })
      return
    }
    navigateRef.current(route, options)
  }, [])

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
    navigateToRoute(route, options)
  }, [navigateToRoute])

  const navigateToNotificationRoute = useCallback((
    action: ActionPerformed,
    options: { replace: boolean },
  ) => {
    const route = parseNativeNotificationRoute(action.notification.data)
    if (route === null) {
      return
    }
    navigateToRoute(route, options)
  }, [navigateToRoute])

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
