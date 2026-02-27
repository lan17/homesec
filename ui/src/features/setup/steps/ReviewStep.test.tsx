// @vitest-environment happy-dom

import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { APIError, apiClient } from '../../../api/client'
import type { FinalizeSnapshot } from '../../../api/client'
import { ReviewStep } from './ReviewStep'

const finalizeMutationState = vi.hoisted(() => ({
  mutateAsync: vi.fn(),
  isPending: false,
}))

vi.mock('../../../api/hooks/useFinalizeMutation', () => ({
  useFinalizeMutation: () => finalizeMutationState,
}))

describe('ReviewStep', () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    finalizeMutationState.mutateAsync.mockReset()
    finalizeMutationState.isPending = false
  })

  it('renders summary cards and routes edit actions by step id', async () => {
    // Given: Review step with camera data and one skipped section
    const onGoToStep = vi.fn()
    render(
      <ReviewStep
        wizardData={{
          camera: {
            name: 'front_door',
            enabled: true,
            source_backend: 'rtsp',
            source_config: { rtsp_url: 'rtsp://front-door' },
          },
          storage: null,
          detection: null,
          notifications: null,
        }}
        skippedSteps={new Set(['storage'])}
        onGoToStep={onGoToStep}
        onLaunchSuccess={vi.fn()}
        onGoDashboard={vi.fn()}
      />,
    )
    const user = userEvent.setup()

    // When: Operator clicks Edit on the camera summary card
    await user.click(screen.getAllByRole('button', { name: 'Edit' })[0] as HTMLButtonElement)

    // Then: Edit callback receives the corresponding wizard step id
    expect(screen.getByText('Camera')).toBeTruthy()
    expect(screen.getByText('Configured')).toBeTruthy()
    expect(screen.getByText('Skipped')).toBeTruthy()
    expect(onGoToStep).toHaveBeenCalledWith('camera')
  })

  it('launches setup and transitions to started state when pipeline health becomes ready', async () => {
    // Given: Finalize succeeds and setup status poll reports running pipeline
    const onLaunchSuccess = vi.fn()
    const onGoDashboard = vi.fn()
    const finalizeResponse: FinalizeSnapshot = {
      success: true,
      config_path: '/tmp/config.yaml',
      restart_requested: true,
      defaults_applied: [],
      errors: [],
      httpStatus: 200,
    }
    finalizeMutationState.mutateAsync.mockResolvedValue(finalizeResponse)
    vi.spyOn(apiClient, 'getSetupStatus').mockResolvedValue({
      state: 'complete',
      has_cameras: true,
      pipeline_running: true,
      auth_configured: false,
      httpStatus: 200,
    })

    render(
      <ReviewStep
        wizardData={{
          camera: null,
          storage: null,
          detection: null,
          notifications: null,
        }}
        skippedSteps={new Set<string>()}
        onGoToStep={vi.fn()}
        onLaunchSuccess={onLaunchSuccess}
        onGoDashboard={onGoDashboard}
      />,
    )
    const user = userEvent.setup()

    // When: Launch is requested from review step
    await user.click(screen.getByRole('button', { name: 'Launch pipeline' }))

    // Then: Finalize runs, launch success callback fires, and dashboard action becomes available
    await waitFor(() => {
      expect(finalizeMutationState.mutateAsync).toHaveBeenCalledTimes(1)
      expect(onLaunchSuccess).toHaveBeenCalledTimes(1)
      expect(screen.getByRole('button', { name: 'Go to Dashboard' })).toBeTruthy()
    })

    await user.click(screen.getByRole('button', { name: 'Go to Dashboard' }))
    expect(onGoDashboard).toHaveBeenCalledTimes(1)
  })

  it('surfaces finalize errors and exposes retry action', async () => {
    // Given: Finalize mutation rejects with canonical API error
    finalizeMutationState.mutateAsync.mockRejectedValue(
      new APIError(
        'Finalize validation failed',
        422,
        { detail: 'Finalize validation failed', error_code: 'SETUP_FINALIZE_INVALID' },
        'SETUP_FINALIZE_INVALID',
      ),
    )

    render(
      <ReviewStep
        wizardData={{
          camera: null,
          storage: null,
          detection: null,
          notifications: null,
        }}
        skippedSteps={new Set<string>()}
        onGoToStep={vi.fn()}
        onLaunchSuccess={vi.fn()}
        onGoDashboard={vi.fn()}
      />,
    )
    const user = userEvent.setup()

    // When: Launch is triggered and finalize rejects
    await user.click(screen.getByRole('button', { name: 'Launch pipeline' }))

    // Then: Error details and retry action are shown to the user
    await waitFor(() => {
      const alert = screen.getByRole('alert')
      expect(alert.textContent).toContain('SETUP_FINALIZE_INVALID')
      expect(screen.getByRole('button', { name: 'Retry launch' })).toBeTruthy()
    })
  })
})
