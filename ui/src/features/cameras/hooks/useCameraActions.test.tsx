// @vitest-environment happy-dom

import { afterEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { PropsWithChildren } from 'react'

import type { CameraResponse, ConfigChangeResponse } from '../../../api/generated/types'
import { useCamerasQuery } from '../../../api/hooks/useCamerasQuery'
import type { CameraCreateActionResult } from '../actions'
import { useCameraActions } from './useCameraActions'

const camera: CameraResponse = {
  name: 'front', enabled: true, source_backend: 'local_folder',
  source_config: { path: '/clips' }, healthy: false, last_heartbeat: null,
}

function response(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), { status, headers: { 'content-type': 'application/json' } })
}

function setup(applyError: NonNullable<ConfigChangeResponse['apply_error']>, initial: CameraResponse[] = []) {
  let cameras = initial
  const fetch = vi.spyOn(globalThis, 'fetch').mockImplementation(async (_url, request) => {
    if (request?.method === 'GET') {
      return response(cameras)
    }
    const savedCamera = { ...camera, enabled: request?.method === 'PATCH' ? false : true }
    cameras = [savedCamera]
    return response({
      restart_required: true, camera: savedCamera, runtime_reload: null, apply_error: applyError,
    }, request?.method === 'POST' ? 201 : 200)
  })
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const wrapper = ({ children }: PropsWithChildren) =>
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  const onRuntimeStatusRefresh = vi.fn().mockResolvedValue(undefined)
  const hook = renderHook(() => ({
    actions: useCameraActions({ onRuntimeStatusRefresh }), cameras: useCamerasQuery(),
  }), { wrapper })
  return { ...hook, fetch, onRuntimeStatusRefresh }
}

describe('useCameraActions saved application refusals', () => {
  afterEach(() => { cleanup(); vi.restoreAllMocks() })

  it('acknowledges creation and refreshes saved cameras when a process restart is required', async () => {
    // Given: A camera save succeeds while parent-owned settings prevent worker reload
    const detail = 'Saved storage settings require a process restart'
    const { result, fetch, onRuntimeStatusRefresh } = setup({ detail, error_code: 'CONFIG_RESTART_REQUIRED' })
    await waitFor(() => expect(result.current.cameras.isSuccess).toBe(true))
    let outcome: CameraCreateActionResult | undefined
    // When: Creating the camera with immediate activation requested
    await act(async () => {
      outcome = await result.current.actions.createCamera({
        name: camera.name, enabled: camera.enabled, source_backend: camera.source_backend, source_config: camera.source_config,
      }, true)
    })
    // Then: Creation is acknowledged, normal invalidation refreshes saved data, and activation remains pending
    expect(outcome).toEqual({ ok: true })
    expect(result.current.actions.hasPendingReload).toBe(true)
    expect(result.current.actions.actionFeedback).toContain(`saved but not applied: ${detail}`)
    expect(result.current.actions.pendingReloadMessage).toContain('Saved changes have not been applied')
    await waitFor(() => expect(result.current.cameras.data?.map((item) => item.name)).toEqual(['front']))
    expect(fetch.mock.calls.filter(([, request]) => request?.method === 'GET')).toHaveLength(2)
    expect(fetch.mock.calls.filter(([, request]) => request?.method === 'POST')).toHaveLength(1)
    expect(onRuntimeStatusRefresh).not.toHaveBeenCalled()
    expect(result.current.actions.errors.create).toBeNull()
  })

  it('acknowledges an update when runtime activation is busy', async () => {
    // Given: Runtime reload is busy after the camera update has already been saved
    const detail = 'A runtime reload is already in progress'
    const { result, fetch, onRuntimeStatusRefresh } = setup({ detail, error_code: 'RELOAD_IN_PROGRESS' }, [camera])
    await waitFor(() => expect(result.current.cameras.isSuccess).toBe(true))
    let outcome = false
    // When: Disabling the camera with immediate activation requested
    await act(async () => { outcome = await result.current.actions.toggleCameraEnabled(camera, true) })
    // Then: Saved update remains successful and query refresh does not imply an accepted reload
    expect(outcome).toBe(true)
    expect(result.current.actions.hasPendingReload).toBe(true)
    expect(result.current.actions.actionFeedback).toContain(`saved but not applied: ${detail}`)
    await waitFor(() => expect(result.current.cameras.data?.[0]?.enabled).toBe(false))
    expect(fetch.mock.calls.filter(([, request]) => request?.method === 'GET')).toHaveLength(2)
    expect(fetch.mock.calls.filter(([, request]) => request?.method === 'PATCH')).toHaveLength(1)
    expect(onRuntimeStatusRefresh).not.toHaveBeenCalled()
    expect(result.current.actions.errors.update).toBeNull()
  })
})
