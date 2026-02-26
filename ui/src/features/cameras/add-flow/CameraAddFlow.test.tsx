// @vitest-environment happy-dom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { CameraAddFlow } from './CameraAddFlow'

const onvifWizardMock = vi.fn()

vi.mock('../components/OnvifDiscoveryWizard', () => ({
  OnvifDiscoveryWizard: (props: unknown) => {
    onvifWizardMock(props)
    return <div data-testid="onvif-wizard">ONVIF Wizard</div>
  },
}))

describe('CameraAddFlow', () => {
  beforeEach(() => {
    onvifWizardMock.mockReset()
  })

  afterEach(() => {
    cleanup()
  })

  it('renders backend picker options before configuration starts', () => {
    // Given: Flow initialized in idle picker stage
    render(
      <CameraAddFlow
        onComplete={vi.fn()}
        onCancel={vi.fn()}
      />,
    )

    // When: Operator inspects available backend cards
    const rtspButton = screen.getByRole('button', { name: /RTSP/i })
    const onvifButton = screen.getByRole('button', { name: /ONVIF Discovery/i })

    // Then: Picker exposes both manual and ONVIF-driven onboarding paths
    expect(rtspButton).toBeTruthy()
    expect(onvifButton).toBeTruthy()
  })

  it('submits manual backend payload when form is valid', async () => {
    // Given: Flow with create callback ready to accept manual config payload
    const onCreateCamera = vi.fn().mockResolvedValue({ ok: true })
    const onCancel = vi.fn()
    const user = userEvent.setup()
    render(
      <CameraAddFlow
        onComplete={onCreateCamera}
        onCancel={onCancel}
      />,
    )

    // When: Operator picks RTSP backend, fills camera name, and submits
    await user.click(screen.getByRole('button', { name: /RTSP/i }))
    await user.type(screen.getByLabelText('Camera name'), 'front_door')
    await user.click(screen.getByRole('button', { name: 'Create camera' }))

    // Then: Flow sends typed camera payload and closes after successful submit
    expect(onCreateCamera).toHaveBeenCalledTimes(1)
    expect(onCreateCamera).toHaveBeenCalledWith(
      expect.objectContaining({
        name: 'front_door',
        enabled: true,
        source_backend: 'rtsp',
        source_config: expect.objectContaining({
          rtsp_url: expect.any(String),
        }),
      }),
      { applyChangesImmediately: false },
    )
    expect(onCancel).toHaveBeenCalledTimes(1)
  })

  it('blocks submit when source config JSON is invalid', async () => {
    // Given: Flow with malformed JSON entered in source config field
    const onCreateCamera = vi.fn().mockResolvedValue({ ok: true })
    const user = userEvent.setup()
    render(
      <CameraAddFlow
        onComplete={onCreateCamera}
        onCancel={vi.fn()}
      />,
    )

    // When: Operator selects RTSP and submits malformed JSON payload
    await user.click(screen.getByRole('button', { name: /RTSP/i }))
    await user.type(screen.getByLabelText('Camera name'), 'garage')
    fireEvent.change(screen.getByLabelText('Source config (JSON)'), {
      target: { value: '{invalid-json' },
    })
    await user.click(screen.getByRole('button', { name: 'Create camera' }))

    // Then: Flow reports validation error and avoids create callback
    expect(onCreateCamera).not.toHaveBeenCalled()
    expect(screen.getByText('Source config must be valid JSON.')).toBeTruthy()
  })

  it('switches to ONVIF wizard when ONVIF backend is selected', async () => {
    // Given: Flow initialized at backend picker stage
    const user = userEvent.setup()
    render(
      <CameraAddFlow
        defaultApplyChangesImmediately
        onApplyChangesImmediatelyChange={vi.fn()}
        onComplete={vi.fn()}
        onCancel={vi.fn()}
      />,
    )

    // When: Operator selects ONVIF discovery backend
    await user.click(screen.getByRole('button', { name: /ONVIF Discovery/i }))

    // Then: Flow renders ONVIF discovery wizard and forwards onboarding callbacks
    expect(screen.getByTestId('onvif-wizard')).toBeTruthy()
    expect(onvifWizardMock).toHaveBeenCalledTimes(1)
  })
})
