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


  it('does not publish an older save response after its editor unmounts', async () => {
    // Given: The server commits v2 but navigation occurs before its PATCH response arrives
    const snapshot = (version: string) => ({ ...saved, saved_config_version: version,
      active_config_version: version, apply_required: 'none' })
    let latest = snapshot('v1')
    let finishFirst: ((value: Response) => void) | undefined
    let patches = 0
    const fetch = vi.spyOn(globalThis, 'fetch').mockImplementation(async (_url, request) => {
      if (request?.method === 'PATCH') {
        latest = snapshot(++patches === 1 ? 'v2' : 'v3')
        if (patches === 1) { return new Promise<Response>((resolve) => { finishFirst = resolve }) }
      }
      return response(latest)
    })
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const provider = ({ children }: PropsWithChildren) => <QueryClientProvider client={client}>{children}</QueryClientProvider>
    const first = renderHook(useConfigSettings, { wrapper: provider })
    await waitFor(() => expect(first.result.current.configQuery.data?.saved_config_version).toBe('v1'))
    let firstSave: Promise<unknown> | undefined
    await act(async () => {
      firstSave = first.result.current.saveConfig({ expected_config_version: 'v1',
        credentials: { 'vlm.config.api_key_env': 'private-first-key' } }).catch((error: unknown) => error)
    })
    await waitFor(() => expect(finishFirst).toBeTruthy())

    // When: A new editor refetches the committed v2 and saves v3 before the old response returns
    first.unmount()
    const second = renderHook(useConfigSettings, { wrapper: provider })
    await waitFor(() => expect(second.result.current.configQuery.data?.saved_config_version).toBe('v2'))
    await act(async () => { await second.result.current.saveConfig({ expected_config_version: 'v2' }) })
    await act(async () => { finishFirst?.(response(snapshot('v2'))); await firstSave })

    // Then: Navigation cancels observation, preserves the committed server write, and cannot regress v3
    expect(fetch.mock.calls.find(([, request]) => request?.method === 'PATCH')?.[1]?.signal?.aborted).toBe(true)
    expect(second.result.current.configQuery.data?.saved_config_version).toBe('v3')
    expect(client.getQueryData(QUERY_KEYS.config)).toMatchObject({ saved_config_version: 'v3' })
    expect(client.getMutationCache().getAll()).toHaveLength(0)
    expect(JSON.stringify(client.getQueryData(QUERY_KEYS.config))).not.toContain('private-first-key')
    expect(await firstSave).toMatchObject({ name: 'AbortError' })
  })

  it('does not publish an old activation response after its monitor unmounts', async () => {
    // Given: An activation monitor has a deferred confirmation GET that ignores cancellation
    let latest = saved
    let finishPoll: ((value: Response) => void) | undefined
    let reads = 0
    const fetch = vi.spyOn(globalThis, 'fetch').mockImplementation(async (_url, request) => {
      if (request?.method === 'POST') {
        return response({ accepted: true, action: 'restart', message: 'Restart scheduled',
          target_config_version: 'target', target_generation: null }, 202)
      }
      if (++reads === 2) { return new Promise<Response>((resolve) => { finishPoll = resolve }) }
      return response(latest)
    })
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const provider = ({ children }: PropsWithChildren) => <QueryClientProvider client={client}>{children}</QueryClientProvider>
    const first = renderHook(useConfigSettings, { wrapper: provider })
    await waitFor(() => expect(first.result.current.configQuery.data).toBeTruthy())
    let apply = Promise.resolve()
    await act(async () => { apply = first.result.current.applyConfig() })
    await waitFor(() => expect(finishPoll).toBeTruthy())

    // When: Navigation mounts a newer v3 editor before the old monitor learns that target activated
    first.unmount()
    latest = { ...saved, saved_config_version: 'v3', active_config_version: 'v3', apply_required: 'none' }
    const second = renderHook(useConfigSettings, { wrapper: provider })
    await waitFor(() => expect(second.result.current.configQuery.data?.saved_config_version).toBe('v3'))
    await act(async () => {
      finishPoll?.(response({ ...saved, active_config_version: 'target', apply_required: 'none' }))
      await apply
    })

    // Then: The closed monitor cannot overwrite the newer saved/active snapshot
    expect(fetch.mock.calls.find(([, request]) => request?.method === 'POST')?.[1]?.signal?.aborted).toBe(true)
    expect(second.result.current.configQuery.data?.saved_config_version).toBe('v3')
    expect(second.result.current.configQuery.data?.active_config_version).toBe('v3')
    expect(client.getQueryData(QUERY_KEYS.config)).toMatchObject({ saved_config_version: 'v3' })
  })


  it('refetches an intervening writer revision instead of publishing a late mounted Save', async () => {
    // Given: PATCH commits v2 but a background GET observes a newer writer's v3 first
    const snapshot = (version: string) => ({ ...saved, saved_config_version: version,
      active_config_version: 'v1', apply_required: 'restart' })
    let latest = snapshot('v1')
    let finishSave: ((value: Response) => void) | undefined
    let reads = 0
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (_url, request) => {
      if (request?.method === 'PATCH') {
        latest = snapshot('v2')
        return new Promise<Response>((resolve) => { finishSave = resolve })
      }
      ++reads
      return response(latest)
    })
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const { result } = renderHook(useConfigSettings, { wrapper: ({ children }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    ) })
    await waitFor(() => expect(result.current.configQuery.data?.saved_config_version).toBe('v1'))
    let save: Promise<unknown> | undefined
    await act(async () => { save = result.current.saveConfig({ expected_config_version: 'v1' }) })
    await waitFor(() => expect(finishSave).toBeTruthy())
    latest = snapshot('v3')
    await act(async () => { await result.current.configQuery.refetch() })
    await waitFor(() => expect(result.current.configQuery.data?.saved_config_version).toBe('v3'))

    // When: The delayed v2 PATCH response reaches the still-mounted editor
    await act(async () => { finishSave?.(response(snapshot('v2'))); await save })

    // Then: The ambiguity triggers an authoritative read and v3 stays visible
    expect(reads).toBe(3)
    expect(result.current.configQuery.data?.saved_config_version).toBe('v3')
    expect(client.getQueryData(QUERY_KEYS.config)).toMatchObject({ saved_config_version: 'v3' })
  })

  it.each(['newer', 'older'])('refetches an intervening %s revision before reporting a late mounted Apply', async (intervening) => {
    // Given: An activation confirmation GET is delayed while a newer saved revision is observed
    let latest = saved
    let finishPoll: ((value: Response) => void) | undefined
    let reads = 0
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (_url, request) => {
      if (request?.method === 'POST') {
        return response({ accepted: true, action: 'restart', message: 'Restart scheduled',
          target_config_version: 'target', target_generation: null }, 202)
      }
      if (++reads === 2) { return new Promise<Response>((resolve) => { finishPoll = resolve }) }
      return response(latest)
    })
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const { result } = renderHook(useConfigSettings, { wrapper: ({ children }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    ) })
    await waitFor(() => expect(result.current.configQuery.data).toBeTruthy())
    let apply = Promise.resolve()
    await act(async () => { apply = result.current.applyConfig() })
    await waitFor(() => expect(finishPoll).toBeTruthy())
    latest = { ...saved, saved_config_version: intervening === 'newer' ? 'v3' : 'stale' }
    await act(async () => { await result.current.configQuery.refetch() })
    await waitFor(() => expect(result.current.configQuery.data?.saved_config_version).toBe(latest.saved_config_version))
    if (intervening === 'older') { latest = { ...saved, active_config_version: 'target', apply_required: 'none' } }

    // When: The old monitor confirms its target after the newer saved revision
    await act(async () => {
      finishPoll?.(response({ ...saved, active_config_version: 'target', apply_required: 'none' }))
      await apply
    })

    // Then: The latest saved revision is refetched and never reported as active from the old target
    expect(reads).toBe(4)
    expect(result.current.configQuery.data?.saved_config_version).toBe(intervening === 'newer' ? 'v3' : 'target')
    expect(result.current.configQuery.data?.apply_required).toBe(intervening === 'newer' ? 'restart' : 'none')
    if (intervening === 'newer') {
      expect(result.current.applyMessage).toContain('saved settings changed')
    } else {
      expect(result.current.applyMessage).toBe('Saved settings are active.')
    }
  })


  it('refetches observed activation of the same saved version instead of regressing it with a delayed PATCH', async () => {
    // Given: A saved v2 PATCH response is delayed while another tab activates that same revision
    const initial = { ...saved, saved_config_version: 'v1', active_config_version: 'v1', apply_required: 'none' }
    const committed = { ...initial, saved_config_version: 'v2', apply_required: 'reload' }
    const activated = { ...committed, active_config_version: 'v2', apply_required: 'none' }
    let latest = initial
    let finishSave: ((value: Response) => void) | undefined
    let reads = 0
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (_url, request) => {
      if (request?.method === 'PATCH') { return new Promise<Response>((resolve) => { finishSave = resolve }) }
      ++reads
      return response(latest)
    })
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const { result } = renderHook(useConfigSettings, { wrapper: ({ children }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    ) })
    await waitFor(() => expect(result.current.configQuery.data?.saved_config_version).toBe('v1'))
    let save: Promise<unknown> | undefined
    await act(async () => { save = result.current.saveConfig({ expected_config_version: 'v1' }) })
    await waitFor(() => expect(finishSave).toBeTruthy())
    latest = activated
    await act(async () => { await result.current.configQuery.refetch() })
    await waitFor(() => expect(result.current.configQuery.data?.apply_required).toBe('none'))

    // When: The old response arrives with the same saved version and an earlier activation status
    await act(async () => { finishSave?.(response(committed)); await save })

    // Then: Another authoritative read resolves the ambiguity and preserves observed activation
    expect(reads).toBe(3)
    expect(result.current.configQuery.data?.saved_config_version).toBe('v2')
    expect(result.current.configQuery.data?.active_config_version).toBe('v2')
    expect(result.current.configQuery.data?.apply_required).toBe('none')
    expect(result.current.applyMessage).toBe('Saved settings are active.')
  })

})
