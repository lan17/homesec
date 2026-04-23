// @vitest-environment happy-dom

import type { PropsWithChildren } from 'react'
import { act, renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { apiClient } from '../../../api/client'
import { useCameraPreview } from './useCameraPreview'

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
      mutations: {
        retry: false,
      },
    },
  })

  return function Wrapper({ children }: PropsWithChildren) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  }
}

describe('useCameraPreview', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('swallows start mutation rejections and exposes the failure via hook state', async () => {
    // Given: Status loads but preview activation fails
    vi.spyOn(apiClient, 'getCameraPreviewStatus').mockResolvedValue({
      camera_name: 'front',
      enabled: true,
      state: 'idle',
      viewer_count: null,
      degraded_reason: null,
      last_error: null,
      idle_shutdown_at: null,
      httpStatus: 200,
    })
    vi.spyOn(apiClient, 'ensureCameraPreviewActive').mockRejectedValue(new Error('preview boom'))

    const { result } = renderHook(() => useCameraPreview('front'), {
      wrapper: createWrapper(),
    })

    await waitFor(() => {
      expect(result.current.status?.state).toBe('idle')
    })

    // When: Starting preview from the hook
    await act(async () => {
      await expect(result.current.start()).resolves.toBeUndefined()
    })

    // Then: The hook stores the failure without surfacing an unhandled rejection
    await waitFor(() => {
      expect(result.current.error?.message).toBe('preview boom')
      expect(result.current.session).toBeNull()
    })
  })
})
