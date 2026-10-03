// @vitest-environment happy-dom

import { afterEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { PropsWithChildren } from 'react'

import { useConfigSettings } from './useConfigSettings'

const saved = { config: {}, saved_config_version: 'target', active_config_version: 'old', apply_required: 'restart' }

function response(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), { status, headers: { 'content-type': 'application/json' } })
}

function wrapper({ children }: PropsWithChildren) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
}

describe('useConfigSettings', () => {
  afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.useRealTimers() })

  it('bounds a stalled application request and keeps the saved outcome explicit', async () => {
    // Given: Saved configuration is available but the application endpoint never responds
    const fetch = vi.spyOn(globalThis, 'fetch').mockImplementation(async (_url, request) => {
      if (request?.method === 'POST') {
        return new Promise<Response>((_resolve, reject) => {
          request.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')))
        })
      }
      return response(saved)
    })
    const { result } = renderHook(useConfigSettings, { wrapper })
    await waitFor(() => expect(result.current.configQuery.data).toBeTruthy())
    vi.useFakeTimers()
    let apply = Promise.resolve()
    // When: Applying the saved version and waiting through the entire time budget
    await act(async () => { apply = result.current.applyConfig() })
    await act(async () => { await vi.advanceTimersByTimeAsync(45_000); await apply })
    // Then: The pending request is aborted, recording is not falsely reported active, and retry becomes available
    expect(result.current.applyPending).toBe(false)
    expect(result.current.applyMessage).toBe('Settings remain saved. Activation has not been confirmed.')
    expect(result.current.applyError).toMatchObject({ message: expect.stringContaining('timeout') })
    const signal = fetch.mock.calls.find(([, request]) => request?.method === 'POST')?.[1]?.signal
    expect(signal?.aborted).toBe(true)
    expect(result.current.configQuery.data?.active_config_version).toBe('old')
  })
})
