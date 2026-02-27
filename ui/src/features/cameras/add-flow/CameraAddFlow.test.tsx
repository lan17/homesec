// @vitest-environment happy-dom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { CameraAddFlow } from './CameraAddFlow'

const onvifWizardMock = vi.fn()
const testConnectionMutateAsyncMock = vi.fn()

vi.mock('../../../api/hooks/useSetupTestConnectionMutation', () => ({
  useSetupTestConnectionMutation: () => ({
    mutateAsync: testConnectionMutateAsyncMock,
    isPending: false,
  }),
}))

vi.mock('../components/OnvifDiscoveryWizard', () => ({
  OnvifDiscoveryWizard: (props: unknown) => {
    onvifWizardMock(props)
    return <div data-testid="onvif-wizard">ONVIF Wizard</div>
  },
}))

async function advanceToConfirm(user: ReturnType<typeof userEvent.setup>): Promise<void> {
  await user.click(screen.getByRole('button', { name: 'RTSP' }))
  await user.click(screen.getByRole('button', { name: 'Next' }))
  await user.click(screen.getByRole('button', { name: 'Continue' }))
}

describe('CameraAddFlow', () => {
  beforeEach(() => {
    onvifWizardMock.mockReset()
    testConnectionMutateAsyncMock.mockReset()
    testConnectionMutateAsyncMock.mockResolvedValue({
      httpStatus: 200,
      success: true,
      message: 'RTSP probe succeeded.',
      latency_ms: 12.4,
      details: null,
    })
  })

  afterEach(() => {
    cleanup()
  })

  it('renders backend picker options before configuration starts', () => {
    // Given: Flow initialized in backend-picker stage
    render(
      <CameraAddFlow existingCameraNames={[]} onComplete={vi.fn()} onDone={vi.fn()} onCancel={vi.fn()} />,
    )

    // When: Operator inspects available backend options
    const rtspButton = screen.getByRole('button', { name: 'RTSP' })
    const onvifButton = screen.getByRole('button', { name: 'ONVIF Discovery' })

    // Then: Picker exposes both manual and ONVIF onboarding paths
    expect(rtspButton).toBeTruthy()
    expect(onvifButton).toBeTruthy()
  })

  it('submits RTSP payload through confirm step and closes via onDone', async () => {
    // Given: Flow with one existing rtsp camera and a successful create callback
    const onComplete = vi.fn().mockResolvedValue({ ok: true })
    const onDone = vi.fn()
    const onCancel = vi.fn()
    const user = userEvent.setup()
    render(
      <CameraAddFlow
        existingCameraNames={['rtsp_1']}
        onComplete={onComplete}
        onDone={onDone}
        onCancel={onCancel}
      />,
    )

    // When: Operator walks through configure -> test -> confirm and submits
    await advanceToConfirm(user)
    await user.clear(screen.getByLabelText('Camera name'))
    await user.type(screen.getByLabelText('Camera name'), 'front_door')
    await user.click(screen.getByLabelText('Apply changes immediately (runtime reload)'))
    await user.click(screen.getByRole('button', { name: 'Create camera' }))

    // Then: Flow submits typed payload and closes via onDone (not cancel)
    expect(onComplete).toHaveBeenCalledTimes(1)
    expect(onComplete).toHaveBeenCalledWith(
      expect.objectContaining({
        name: 'front_door',
        enabled: true,
        source_backend: 'rtsp',
        source_config: expect.objectContaining({
          rtsp_url: expect.any(String),
        }),
      }),
      { applyChangesImmediately: true },
    )
    expect(onDone).toHaveBeenCalledTimes(1)
    expect(onCancel).not.toHaveBeenCalled()
  })

  it('rejects camera names with spaces or special characters', async () => {
    // Given: Flow progressed to confirm stage with an invalid camera name entered
    const onComplete = vi.fn().mockResolvedValue({ ok: true })
    const user = userEvent.setup()
    render(
      <CameraAddFlow existingCameraNames={[]} onComplete={onComplete} onDone={vi.fn()} onCancel={vi.fn()} />,
    )
    await advanceToConfirm(user)
    await user.clear(screen.getByLabelText('Camera name'))
    await user.type(screen.getByLabelText('Camera name'), 'front door!')

    // When: Operator submits confirm step
    await user.click(screen.getByRole('button', { name: 'Create camera' }))

    // Then: Validation error is shown and create callback is not called
    expect(onComplete).not.toHaveBeenCalled()
    expect(
      screen.getByText('Camera name may contain only letters, numbers, underscores, and hyphens.'),
    ).toBeTruthy()
  })

  it('runs setup test-connection during test step and shows result', async () => {
    // Given: Flow in test step with mocked successful setup test endpoint
    const user = userEvent.setup()
    render(
      <CameraAddFlow existingCameraNames={[]} onComplete={vi.fn()} onDone={vi.fn()} onCancel={vi.fn()} />,
    )
    await user.click(screen.getByRole('button', { name: 'RTSP' }))
    await user.click(screen.getByRole('button', { name: 'Next' }))

    // When: Operator runs connection test
    await user.click(screen.getByRole('button', { name: 'Run connection test' }))

    // Then: Request is dispatched and PASS status is displayed
    expect(testConnectionMutateAsyncMock).toHaveBeenCalledTimes(1)
    expect(screen.getByText('PASS')).toBeTruthy()
    expect(screen.getByText('RTSP probe succeeded.')).toBeTruthy()
  })

  it('renders ONVIF discovery flow when ONVIF backend is selected', async () => {
    // Given: Flow initialized in picker stage
    const user = userEvent.setup()
    render(
      <CameraAddFlow existingCameraNames={[]} onComplete={vi.fn()} onDone={vi.fn()} onCancel={vi.fn()} />,
    )

    // When: Operator selects ONVIF discovery backend
    await user.click(screen.getByRole('button', { name: 'ONVIF Discovery' }))

    // Then: ONVIF wizard component is mounted as backend configure step
    expect(screen.getByTestId('onvif-wizard')).toBeTruthy()
    expect(onvifWizardMock).toHaveBeenCalledTimes(1)
  })
})

