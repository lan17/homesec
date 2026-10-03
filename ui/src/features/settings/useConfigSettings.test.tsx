// @vitest-environment happy-dom

import { afterEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { PropsWithChildren } from 'react'

import { QUERY_KEYS } from '../../api/hooks/queryKeys'
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

  it('keeps the saved snapshot when a GET started during PATCH completes late', async () => {
    // Given: A draft save and a subsequent background GET are both deferred at the HTTP boundary
    const old = { ...saved, saved_config_version: 'old', apply_required: 'none' }
    let finishSave: ((value: Response) => void) | undefined
    let finishRead: ((value: Response) => void) | undefined
    let reads = 0
    const fetch = vi.spyOn(globalThis, 'fetch').mockImplementation(async (_url, request) => {
      if (request?.method === 'PATCH') {
        return new Promise<Response>((resolve) => { finishSave = resolve })
      }
      if (++reads === 1) { return response(old) }
      return new Promise<Response>((resolve) => { finishRead = resolve })
    })
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const { result } = renderHook(useConfigSettings, { wrapper: ({ children }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    ) })
    await waitFor(() => expect(result.current.configQuery.data?.saved_config_version).toBe('old'))
    let save: ReturnType<typeof result.current.saveConfig> | undefined
    await act(async () => { save = result.current.saveConfig({ expected_config_version: 'old' }) })
    await waitFor(() => expect(finishSave).toBeTruthy())
    const read = client.refetchQueries({ queryKey: QUERY_KEYS.config })
    await waitFor(() => expect(finishRead).toBeTruthy())

    // When: The authoritative save finishes first, followed by the stale GET
    await act(async () => { finishSave?.(response(saved)); await save })
    expect(fetch.mock.calls.find(([, request]) => request?.method === 'GET' && request.signal?.aborted)?.[1]?.signal?.aborted).toBe(true)
    await act(async () => { finishRead?.(response(old)); await read })

    // Then: The form and public cache retain the saved version and its activation requirement
    expect(result.current.configQuery.data?.saved_config_version).toBe('target')
    expect(result.current.configQuery.data?.apply_required).toBe('restart')
    expect(client.getQueryData(QUERY_KEYS.config)).toMatchObject({ saved_config_version: 'target' })
  })

  it('keeps the activated snapshot when a background GET completes late', async () => {
    // Given: Activation polling and a background config GET overlap
    const activated = { ...saved, active_config_version: 'target', apply_required: 'none' }
    let finishRead: ((value: Response) => void) | undefined
    let finishApply: ((value: Response) => void) | undefined
    let reads = 0
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (_url, request) => {
      if (request?.method === 'POST') {
        return new Promise<Response>((resolve) => { finishApply = resolve })
      }
      if (++reads === 2) { return new Promise<Response>((resolve) => { finishRead = resolve }) }
      return response(reads === 1 ? saved : activated)
    })
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const { result } = renderHook(useConfigSettings, { wrapper: ({ children }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    ) })
    await waitFor(() => expect(result.current.configQuery.data).toBeTruthy())
    let apply = Promise.resolve()
    await act(async () => { apply = result.current.applyConfig() })
    await waitFor(() => expect(finishApply).toBeTruthy())
    const read = client.refetchQueries({ queryKey: QUERY_KEYS.config })
    await waitFor(() => expect(finishRead).toBeTruthy())

    // When: Runtime activation is confirmed before the earlier GET returns its old status
    await act(async () => {
      finishApply?.(response({ accepted: true, action: 'restart', message: 'Restart scheduled',
        target_config_version: 'target', target_generation: null }, 202))
      await apply
    })
    await act(async () => { finishRead?.(response(saved)); await read })

    // Then: The editor continues to report the confirmed active version
    expect(result.current.configQuery.data?.active_config_version).toBe('target')
    expect(result.current.configQuery.data?.apply_required).toBe('none')
    expect(result.current.applyMessage).toBe('Saved settings are active.')
  })

})
