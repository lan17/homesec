// @vitest-environment happy-dom

import type { PropsWithChildren } from 'react'
import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, expect, it, vi } from 'vitest'

import { browserServerBaseUrlProvider } from '../../../api/client'
import { runtimeAuthTokenProvider } from '../../../api/tokenProvider'
import { useCameraPreview } from './useCameraPreview'

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { 'content-type': 'application/json' },
  })
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  window.sessionStorage.clear()
})

it.each([false, true])('ignores an unmounted activation after changing servers, explicit stop=%s', async (explicitStop) => {
  // Given: Server A has a pending HLS activation, optionally superseded by explicit Stop.
  await browserServerBaseUrlProvider.setBaseUrl('https://a.example.test')
  await runtimeAuthTokenProvider.setToken('synthetic-a')
  let resolveActivation!: (response: Response) => void
  const requests: Array<{ url: string; method: string; authorization: string | null }> = []
  vi.spyOn(globalThis, 'fetch').mockImplementation(async (url, options) => {
    const method = options?.method ?? 'GET'
    requests.push({
      url: String(url),
      method,
      authorization: new Headers(options?.headers).get('authorization'),
    })
    if (method === 'POST') {
      return new Promise<Response>((resolve) => { resolveActivation = resolve })
    }
    if (method === 'DELETE') {
      return jsonResponse({ accepted: true, state: 'idle' }, 202)
    }
    return jsonResponse({
      camera_name: 'front', enabled: true, state: 'ready', viewer_count: 1,
      degraded_reason: null, last_error: null, idle_shutdown_at: null,
    })
  })
  const queryClient = new QueryClient({ defaultOptions: {
    queries: { retry: false }, mutations: { retry: false },
  } })
  function Wrapper({ children }: PropsWithChildren) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  }
  const { result, unmount } = renderHook(() => useCameraPreview('front'), { wrapper: Wrapper })
  await waitFor(() => expect(result.current.canStop).toBe(true))
  let activation!: Promise<void>
  await act(async () => { activation = result.current.start() })
  await waitFor(() => expect(requests.some((request) => request.method === 'POST')).toBe(true))
  if (explicitStop) {
    await act(async () => { await result.current.stop() })
  }

  // When: Connection setup unmounts the viewer, switches to B, and A's response arrives late.
  unmount()
  await queryClient.cancelQueries()
  queryClient.clear()
  await runtimeAuthTokenProvider.clearToken()
  await browserServerBaseUrlProvider.setBaseUrl('https://b.example.test')
  await runtimeAuthTokenProvider.setToken('synthetic-b')
  const requestsBeforeResponse = [...requests]
  await act(async () => {
    resolveActivation(jsonResponse({
      camera_name: 'front', state: 'ready', viewer_count: 1, transport: 'hls',
      token: 'synthetic-preview', token_expires_at: null, lease_expires_at: null,
      playlist_url: '/api/v1/preview/cameras/front/index.m3u8?token=synthetic-preview',
      signaling_url: null, ice_servers: [], idle_timeout_s: 30, warning: null,
    }))
    await activation
  })

  // Then: The detached viewer neither stops B's publisher nor repopulates its cleared cache.
  expect(requests).toEqual(requestsBeforeResponse)
  expect(requests.filter((request) => request.method === 'DELETE')).toEqual(explicitStop ? [{
    url: 'https://a.example.test/api/v1/preview/cameras/front',
    method: 'DELETE',
    authorization: 'Bearer synthetic-a',
  }] : [])
  expect(queryClient.getQueryCache().getAll()).toEqual([])
})
