// @vitest-environment happy-dom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import type { RuntimeStatusSnapshot } from '../../api/client'
import type { CameraResponse } from '../../api/generated/types'
import type { useRuntimeStatusQuery } from '../../api/hooks/useRuntimeStatusQuery'
import type { useCamerasQuery } from '../../api/hooks/useCamerasQuery'
import type { useCameraActions } from './hooks/useCameraActions'
import { CamerasPage } from './CamerasPage'

const useCamerasQueryMock = vi.fn()
const useRuntimeStatusQueryMock = vi.fn()
const useCameraActionsMock = vi.fn()

vi.mock('../../api/hooks/useCamerasQuery', () => ({
  useCamerasQuery: () => useCamerasQueryMock(),
}))

vi.mock('../../api/hooks/useRuntimeStatusQuery', () => ({
  useRuntimeStatusQuery: () => useRuntimeStatusQueryMock(),
}))

vi.mock('./hooks/useCameraActions', () => ({
  useCameraActions: (options: unknown) => useCameraActionsMock(options),
}))

interface CamerasPageHarness {
  createCamera: ReturnType<typeof vi.fn>
  toggleCameraEnabled: ReturnType<typeof vi.fn>
  deleteCamera: ReturnType<typeof vi.fn>
  applyRuntimeReload: ReturnType<typeof vi.fn>
}

function makeDefaultRuntimeStatus(): RuntimeStatusSnapshot {
  return {
    httpStatus: 200,
    state: 'idle',
    generation: 3,
    reload_in_progress: false,
    active_config_version: 'cfg-v3',
    last_reload_at: null,
    last_reload_error: null,
  }
}

function makeDefaultCamera(name: string): CameraResponse {
  return {
    name,
    enabled: true,
    source_backend: 'rtsp',
    healthy: true,
    last_heartbeat: 1_739_590_400,
    source_config: { rtsp_url: 'rtsp://***redacted***@camera.local/stream' },
  }
}

function setupPage({
  cameras = [],
  runtimeStatus = makeDefaultRuntimeStatus(),
  hasPendingReload = false,
  pendingReloadMessage = null,
}: {
  cameras?: CameraResponse[]
  runtimeStatus?: RuntimeStatusSnapshot | undefined
  hasPendingReload?: boolean
  pendingReloadMessage?: string | null
} = {}): CamerasPageHarness {
  const camerasRefetch = vi.fn().mockResolvedValue(undefined)
  const runtimeRefetch = vi.fn().mockResolvedValue(undefined)

  useCamerasQueryMock.mockReturnValue({
    data: cameras,
    isPending: false,
    isFetching: false,
    error: null,
    refetch: camerasRefetch,
  } as unknown as ReturnType<typeof useCamerasQuery>)

  useRuntimeStatusQueryMock.mockReturnValue({
    data: runtimeStatus,
    isPending: false,
    isFetching: false,
    error: null,
    refetch: runtimeRefetch,
  } as unknown as ReturnType<typeof useRuntimeStatusQuery>)

  const createCamera = vi.fn().mockResolvedValue(true)
  const toggleCameraEnabled = vi.fn().mockResolvedValue(true)
  const deleteCamera = vi.fn().mockResolvedValue(true)
  const applyRuntimeReload = vi.fn().mockResolvedValue(true)

  useCameraActionsMock.mockReturnValue({
    createCamera,
    toggleCameraEnabled,
    deleteCamera,
    applyRuntimeReload,
    hasPendingReload,
    pendingReloadMessage,
    actionFeedback: null,
    isMutating: false,
    pending: { create: false, update: false, delete: false, reload: false },
    errors: { create: null, update: null, delete: null, reload: null },
  } as unknown as ReturnType<typeof useCameraActions>)

  render(<CamerasPage />)

  return {
    createCamera,
    toggleCameraEnabled,
    deleteCamera,
    applyRuntimeReload,
  }
}

describe('CamerasPage', () => {
  beforeEach(() => {
    useCamerasQueryMock.mockReset()
    useRuntimeStatusQueryMock.mockReset()
    useCameraActionsMock.mockReset()
    vi.restoreAllMocks()
  })

  afterEach(() => {
    cleanup()
  })

  it('renders empty state when no cameras are configured', () => {
    // Given: The cameras query returns an empty list
    setupPage({ cameras: [] })

    // When: The cameras page is rendered
    const emptyState = screen.getByText('No cameras configured yet. Create your first camera above.')

    // Then: The empty-state guidance is shown
    expect(emptyState).toBeTruthy()
  })

  it('submits create camera payload from form values', async () => {
    // Given: A ready page with no mutation in flight
    const harness = setupPage()
    const user = userEvent.setup()

    // When: Operator enters camera name and submits create form
    await user.type(screen.getByLabelText('Camera name'), 'front_door')
    await user.click(screen.getByRole('button', { name: 'Create camera' }))

    // Then: The create mutation receives a typed camera payload
    expect(harness.createCamera).toHaveBeenCalledTimes(1)
    expect(harness.createCamera).toHaveBeenCalledWith(
      expect.objectContaining({
        name: 'front_door',
        enabled: true,
        source_backend: 'rtsp',
        source_config: expect.objectContaining({
          rtsp_url: expect.any(String),
        }),
      }),
    )
  })

  it('rejects invalid source config JSON before mutation', async () => {
    // Given: A ready page and malformed source config input
    const harness = setupPage()
    const user = userEvent.setup()

    // When: Operator submits an invalid JSON source_config
    await user.type(screen.getByLabelText('Camera name'), 'garage')
    fireEvent.change(screen.getByLabelText('Source config (JSON)'), {
      target: { value: '{invalid-json' },
    })
    await user.click(screen.getByRole('button', { name: 'Create camera' }))

    // Then: Validation error is surfaced and no create mutation is sent
    expect(screen.getByText('Source config must be valid JSON.')).toBeTruthy()
    expect(harness.createCamera).not.toHaveBeenCalled()
  })

  it('runs enable toggle and delete flows from camera row actions', async () => {
    // Given: A rendered camera row with enable/disable and delete actions
    const camera = makeDefaultCamera('front')
    const harness = setupPage({ cameras: [camera] })
    const user = userEvent.setup()
    Object.defineProperty(window, 'confirm', {
      configurable: true,
      value: vi.fn(() => true),
    })

    // When: Operator disables camera and confirms delete
    await user.click(screen.getByRole('button', { name: 'Disable' }))
    await user.click(screen.getByRole('button', { name: 'Delete' }))

    // Then: Toggle and delete mutations are called with the selected camera
    expect(harness.toggleCameraEnabled).toHaveBeenCalledTimes(1)
    expect(harness.toggleCameraEnabled).toHaveBeenCalledWith(camera)
    expect(harness.deleteCamera).toHaveBeenCalledTimes(1)
    expect(harness.deleteCamera).toHaveBeenCalledWith('front')
  })

  it('triggers runtime reload from pending-reload banner', async () => {
    // Given: A page state with restart-required camera changes pending
    const harness = setupPage({
      hasPendingReload: true,
      pendingReloadMessage: 'Camera updates require runtime reload.',
    })
    const user = userEvent.setup()

    // When: Operator applies runtime reload
    await user.click(screen.getByRole('button', { name: 'Apply runtime reload' }))

    // Then: Runtime reload mutation is triggered once
    expect(harness.applyRuntimeReload).toHaveBeenCalledTimes(1)
  })
})
