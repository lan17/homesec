// @vitest-environment happy-dom

import type { PropsWithChildren } from 'react'
import { renderHook, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { SetupStatusSnapshot } from '../../api/client'
import { WIZARD_STATE_STORAGE_KEY } from './useWizardState'
import { useSetupRedirect } from './useSetupRedirect'

const useSetupStatusQueryMock = vi.fn()
const navigateMock = vi.fn()

vi.mock('../../api/hooks/useSetupStatusQuery', () => ({
  useSetupStatusQuery: () => useSetupStatusQueryMock(),
}))

vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom')
  return {
    ...actual,
    useNavigate: () => navigateMock,
  }
})

function createWrapper() {
  return function Wrapper({ children }: PropsWithChildren) {
    return <MemoryRouter initialEntries={['/']}>{children}</MemoryRouter>
  }
}

function setupStatusSnapshot(state: SetupStatusSnapshot['state']): SetupStatusSnapshot {
  return {
    httpStatus: 200,
    state,
    has_cameras: state !== 'fresh',
    pipeline_running: state === 'complete',
    auth_configured: true,
  }
}

describe('useSetupRedirect', () => {
  beforeEach(() => {
    useSetupStatusQueryMock.mockReset()
    navigateMock.mockReset()
    window.localStorage.clear()
  })

  it('redirects to setup route when setup state is fresh', async () => {
    // Given: Setup status returns fresh and no local wizard progress exists
    useSetupStatusQueryMock.mockReturnValue({
      data: setupStatusSnapshot('fresh'),
      error: null,
      isPending: false,
    })

    // When: Running setup redirect hook
    const { result } = renderHook(() => useSetupRedirect(), { wrapper: createWrapper() })

    // Then: Hook requests a replace navigation to /setup
    await waitFor(() => {
      expect(result.current.shouldRedirect).toBe(true)
      expect(navigateMock).toHaveBeenCalledWith('/setup', { replace: true })
    })
  })

  it('does not redirect when setup state is partial or complete', async () => {
    // Given: Setup status is not fresh
    useSetupStatusQueryMock.mockReturnValue({
      data: setupStatusSnapshot('partial'),
      error: null,
      isPending: false,
    })

    // When: Running setup redirect hook
    const { result } = renderHook(() => useSetupRedirect(), { wrapper: createWrapper() })

    // Then: Redirect remains disabled and navigate is not called
    await waitFor(() => {
      expect(result.current.shouldRedirect).toBe(false)
      expect(navigateMock).not.toHaveBeenCalled()
    })
  })

  it('does not redirect when setup state is complete', async () => {
    // Given: Setup status reports complete system bootstrap state
    useSetupStatusQueryMock.mockReturnValue({
      data: setupStatusSnapshot('complete'),
      error: null,
      isPending: false,
    })

    // When: Running setup redirect hook
    const { result } = renderHook(() => useSetupRedirect(), { wrapper: createWrapper() })

    // Then: Redirect remains disabled and navigation is not triggered
    await waitFor(() => {
      expect(result.current.shouldRedirect).toBe(false)
      expect(navigateMock).not.toHaveBeenCalled()
    })
  })

  it('does not redirect when setup status query fails', async () => {
    // Given: Setup status query has an API error
    useSetupStatusQueryMock.mockReturnValue({
      data: undefined,
      error: new Error('network'),
      isPending: false,
    })

    // When: Running setup redirect hook
    const { result } = renderHook(() => useSetupRedirect(), { wrapper: createWrapper() })

    // Then: Hook fails open and does not navigate
    await waitFor(() => {
      expect(result.current.shouldRedirect).toBe(false)
      expect(navigateMock).not.toHaveBeenCalled()
    })
  })

  it('does not redirect when wizard progress exists in localStorage', async () => {
    // Given: Setup is fresh but wizard local state indicates user already started setup
    window.localStorage.setItem(
      WIZARD_STATE_STORAGE_KEY,
      JSON.stringify({
        schemaVersion: 1,
        currentStep: 2,
        stepData: {},
        completedSteps: ['welcome'],
        skippedSteps: [],
      }),
    )
    useSetupStatusQueryMock.mockReturnValue({
      data: setupStatusSnapshot('fresh'),
      error: null,
      isPending: false,
    })

    // When: Running setup redirect hook
    const { result } = renderHook(() => useSetupRedirect(), { wrapper: createWrapper() })

    // Then: Redirect is suppressed and no navigation occurs
    await waitFor(() => {
      expect(result.current.shouldRedirect).toBe(false)
      expect(navigateMock).not.toHaveBeenCalled()
    })
  })

  it('reports checking state while setup status query is pending', async () => {
    // Given: Setup status query is still loading
    useSetupStatusQueryMock.mockReturnValue({
      data: undefined,
      error: null,
      isPending: true,
    })

    // When: Running setup redirect hook
    const { result } = renderHook(() => useSetupRedirect(), { wrapper: createWrapper() })

    // Then: Hook reports checking state and does not navigate yet
    await waitFor(() => {
      expect(result.current.isChecking).toBe(true)
      expect(result.current.shouldRedirect).toBe(false)
      expect(navigateMock).not.toHaveBeenCalled()
    })
  })
})
