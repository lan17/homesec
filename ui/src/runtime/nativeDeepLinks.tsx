import { useCallback, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { App } from '@capacitor/app'
import type { PluginListenerHandle } from '@capacitor/core'

import { parseNativeDeepLinkRoute } from './nativeDeepLinkRoutes'
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

export function NativeDeepLinkRouter({ app = App }: { app?: NativeDeepLinkApp }) {
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

  useEffect(() => {
    if (!isIOS) {
      return
    }

    let cancelled = false
    let handle: PluginListenerHandle | null = null

    void app.getLaunchUrl()
      .then((event) => {
        if (!cancelled) {
          navigateToDeepLink(event?.url, { replace: true })
        }
      })
      .catch(() => {})

    void app.addListener('appUrlOpen', (event) => {
      navigateToDeepLink(event.url, { replace: false })
    })
      .then((nextHandle) => {
        if (cancelled) {
          void nextHandle.remove()
          return
        }
        handle = nextHandle
      })
      .catch(() => {})

    return () => {
      cancelled = true
      if (handle !== null) {
        void handle.remove()
      }
    }
  }, [app, isIOS, navigateToDeepLink])

  return null
}
