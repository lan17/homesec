// @vitest-environment happy-dom

import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'

const nativeRuntimeMock = vi.hoisted(() => ({
  isIOSNativeApp: vi.fn<() => boolean>(() => false),
}))

vi.mock('./nativeRuntime', () => ({
  isIOSNativeApp: () => nativeRuntimeMock.isIOSNativeApp(),
}))

import { parseNativeDeepLinkRoute } from './nativeDeepLinkRoutes'
import { NativeDeepLinkRouter } from './nativeDeepLinks'

type DeepLinkEvent = {
  url?: string | null
}

function LocationProbe() {
  const location = useLocation()
  return <p data-testid="location">{`${location.pathname}${location.search}${location.hash}`}</p>
}

function createNativeDeepLinkApp(launchUrl?: string | null) {
  let appUrlOpenListener: ((event: DeepLinkEvent) => void) | null = null
  const remove = vi.fn()
  const app = {
    getLaunchUrl: vi.fn(async () => (launchUrl === undefined ? null : { url: launchUrl })),
    addListener: vi.fn(async (
      eventName: 'appUrlOpen',
      listenerFunc: (event: DeepLinkEvent) => void,
    ) => {
      expect(eventName).toBe('appUrlOpen')
      appUrlOpenListener = listenerFunc
      return { remove }
    }),
  }

  return {
    app,
    emitUrlOpen(rawUrl: string) {
      appUrlOpenListener?.({ url: rawUrl })
    },
    remove,
  }
}

function renderNativeDeepLinkRouter(
  app: ReturnType<typeof createNativeDeepLinkApp>['app'],
  initialPath = '/live',
) {
  render(
    <MemoryRouter initialEntries={[initialPath]}>
      <NativeDeepLinkRouter app={app} />
      <Routes>
        <Route path="*" element={<LocationProbe />} />
      </Routes>
    </MemoryRouter>,
  )
}

describe('parseNativeDeepLinkRoute', () => {
  it('translates homesec event links into React routes', () => {
    // Given: A notification deep link with route and source context
    const route = parseNativeDeepLinkRoute('homesec://events/test-id?from=notification')

    // Then: The custom scheme is stripped and the React route is preserved
    expect(route).toBe('/events/test-id?from=notification')
  })

  it('preserves triple-slash path links, query strings, and hashes', () => {
    // Given: A custom-scheme URL using path form instead of host form
    const route = parseNativeDeepLinkRoute('homesec:///events/clip-42?camera=front#summary')

    // Then: The parser keeps the route details needed by React Router
    expect(route).toBe('/events/clip-42?camera=front#summary')
  })

  it('ignores non-HomeSec URLs', () => {
    // Given: A URL that was not issued for the HomeSec app scheme
    const route = parseNativeDeepLinkRoute('https://homesec.example.com/events/test-id')

    // Then: The native listener leaves unrelated URLs alone
    expect(route).toBeNull()
  })

  it('falls back safely for unsupported HomeSec routes', () => {
    // Given: A HomeSec-scheme URL that does not map to a known app route
    const route = parseNativeDeepLinkRoute('homesec://admin/secrets?token=leak')

    // Then: The app opens a safe default route instead of an arbitrary path
    expect(route).toBe('/live')
  })
})

describe('NativeDeepLinkRouter', () => {
  beforeEach(() => {
    nativeRuntimeMock.isIOSNativeApp.mockReturnValue(true)
  })

  afterEach(() => {
    cleanup()
    nativeRuntimeMock.isIOSNativeApp.mockReset()
  })

  it('routes a cold-start launch URL into the React app', async () => {
    // Given: iOS launched the app from a notification deep link
    const { app } = createNativeDeepLinkApp('homesec://events/test-id?from=notification')

    // When: The deep-link router mounts
    renderNativeDeepLinkRouter(app, '/live')

    // Then: The launch URL becomes the active React route
    await waitFor(() => {
      expect(screen.getByTestId('location').textContent).toBe(
        '/events/test-id?from=notification',
      )
    })
    expect(app.getLaunchUrl).toHaveBeenCalledTimes(1)
    expect(app.addListener).toHaveBeenCalledTimes(1)
  })

  it('routes warm appUrlOpen events into the React app', async () => {
    // Given: The app is already open and listening for URL events
    const nativeApp = createNativeDeepLinkApp()
    renderNativeDeepLinkRouter(nativeApp.app, '/live')
    await waitFor(() => {
      expect(nativeApp.app.addListener).toHaveBeenCalledTimes(1)
    })

    // When: iOS sends a custom-scheme URL to the running app
    await act(async () => {
      nativeApp.emitUrlOpen('homesec://events/clip-99?from=notification')
    })

    // Then: React Router navigates to the event detail route
    expect(screen.getByTestId('location').textContent).toBe(
      '/events/clip-99?from=notification',
    )
  })

  it('falls back to Live for unsupported HomeSec appUrlOpen routes', async () => {
    // Given: The app receives an invalid route under the HomeSec scheme
    const nativeApp = createNativeDeepLinkApp()
    renderNativeDeepLinkRouter(nativeApp.app, '/events')
    await waitFor(() => {
      expect(nativeApp.app.addListener).toHaveBeenCalledTimes(1)
    })

    // When: The invalid route opens
    await act(async () => {
      nativeApp.emitUrlOpen('homesec://admin')
    })

    // Then: The app navigates to a safe default route
    expect(screen.getByTestId('location').textContent).toBe('/live')
  })

  it('does not register native listeners outside iOS native mode', () => {
    // Given: The React app is running in the browser
    nativeRuntimeMock.isIOSNativeApp.mockReturnValue(false)
    const { app } = createNativeDeepLinkApp('homesec://events/test-id')

    // When: The deep-link router mounts
    renderNativeDeepLinkRouter(app, '/live')

    // Then: Capacitor deep-link APIs are not invoked
    expect(app.getLaunchUrl).not.toHaveBeenCalled()
    expect(app.addListener).not.toHaveBeenCalled()
    expect(screen.getByTestId('location').textContent).toBe('/live')
  })

  it('removes the appUrlOpen listener on unmount', async () => {
    // Given: The deep-link router registered a native listener
    const nativeApp = createNativeDeepLinkApp()
    renderNativeDeepLinkRouter(nativeApp.app, '/live')
    await waitFor(() => {
      expect(nativeApp.app.addListener).toHaveBeenCalledTimes(1)
    })

    // When: React unmounts the router
    cleanup()

    // Then: The native listener is removed
    expect(nativeApp.remove).toHaveBeenCalledTimes(1)
  })
})
