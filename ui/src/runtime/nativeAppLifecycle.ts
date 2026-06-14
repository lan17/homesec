import { useEffect, useState } from 'react'
import { App } from '@capacitor/app'
import type { PluginListenerHandle } from '@capacitor/core'

import { isIOSNativeApp } from './nativeRuntime'

export interface NativeAppLifecycleState {
  isActive: boolean
  pauseCount: number
  resumeCount: number
}

const ACTIVE_BROWSER_LIFECYCLE_STATE: NativeAppLifecycleState = {
  isActive: true,
  pauseCount: 0,
  resumeCount: 0,
}

export function useNativeAppLifecycleState(): NativeAppLifecycleState {
  const isIOS = isIOSNativeApp()
  const [state, setState] = useState<NativeAppLifecycleState>(ACTIVE_BROWSER_LIFECYCLE_STATE)

  useEffect(() => {
    if (!isIOS) {
      return
    }

    let cancelled = false
    const handles: PluginListenerHandle[] = []

    const trackHandle = async (listener: Promise<PluginListenerHandle>): Promise<void> => {
      const handle = await listener.catch(() => null)
      if (handle === null) {
        return
      }
      if (cancelled) {
        void handle.remove()
        return
      }
      handles.push(handle)
    }

    void App.getState()
      .then((appState) => {
        if (!cancelled) {
          setState((previous) => ({ ...previous, isActive: appState.isActive }))
        }
      })
      .catch(() => {})

    void trackHandle(
      App.addListener('appStateChange', (appState) => {
        setState((previous) => ({ ...previous, isActive: appState.isActive }))
      }),
    )
    void trackHandle(
      App.addListener('pause', () => {
        setState((previous) => ({
          ...previous,
          isActive: false,
          pauseCount: previous.pauseCount + 1,
        }))
      }),
    )
    void trackHandle(
      App.addListener('resume', () => {
        setState((previous) => ({
          ...previous,
          isActive: true,
          resumeCount: previous.resumeCount + 1,
        }))
      }),
    )

    return () => {
      cancelled = true
      handles.forEach((handle) => {
        void handle.remove()
      })
    }
  }, [isIOS])

  return isIOS ? state : ACTIVE_BROWSER_LIFECYCLE_STATE
}
