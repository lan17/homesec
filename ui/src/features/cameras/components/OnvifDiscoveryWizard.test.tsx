// @vitest-environment happy-dom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { APIError } from '../../../api/errors'
import { OnvifDiscoveryWizard } from './OnvifDiscoveryWizard'

const discoverOnvifCamerasMock = vi.fn()
const probeOnvifCameraMock = vi.fn()

vi.mock('../onvifApi', () => ({
  discoverOnvifCameras: (...args: unknown[]) => discoverOnvifCamerasMock(...args),
  probeOnvifCamera: (...args: unknown[]) => probeOnvifCameraMock(...args),
}))

describe('OnvifDiscoveryWizard', () => {
  beforeEach(() => {
    discoverOnvifCamerasMock.mockReset()
    probeOnvifCameraMock.mockReset()
  })

  afterEach(() => {
    cleanup()
  })

  it('moves from discovery step to probe step after camera selection', async () => {
    // Given: Discovery endpoint returns one ONVIF camera
    discoverOnvifCamerasMock.mockResolvedValue([
      {
        ip: '192.168.1.20',
        xaddr: 'http://192.168.1.20:8899/onvif/device_service',
        scopes: ['onvif://scope/location/garage'],
        types: ['dn:NetworkVideoTransmitter'],
      },
    ])
    const onCreateCamera = vi.fn().mockResolvedValue({ ok: true })
    const onClose = vi.fn()
    const user = userEvent.setup()
    render(
      <OnvifDiscoveryWizard
        applyChangesImmediately={false}
        createPending={false}
        isMutating={false}
        onCreateCamera={onCreateCamera}
        onClose={onClose}
      />,
    )

    // When: Operator scans and selects discovered camera
    await user.click(screen.getByRole('button', { name: 'Run discovery scan' }))
    await user.click(await screen.findByRole('button', { name: /192.168.1.20/ }))

    // Then: Wizard should advance to credential/probe step and prefill discovered ONVIF port
    expect(screen.getByText('Step 2: Authenticate and probe')).toBeTruthy()
    expect((screen.getByLabelText('ONVIF port') as HTMLInputElement).value).toBe('8899')
  })

  it('creates camera with injected RTSP credentials and applyChanges forwarding', async () => {
    // Given: Discovery + probe return a stream URI without embedded credentials
    discoverOnvifCamerasMock.mockResolvedValue([
      {
        ip: '192.168.1.21',
        xaddr: 'http://192.168.1.21/onvif/device_service',
        scopes: [],
        types: [],
      },
    ])
    probeOnvifCameraMock.mockResolvedValue({
      device: {
        manufacturer: 'Acme',
        model: 'CamPro',
        firmware_version: '1.0.0',
        serial_number: 'SN123',
        hardware_id: 'HW456',
      },
      profiles: [
        {
          token: 'main',
          name: 'Main stream',
          video_encoding: 'H264',
          width: 1920,
          height: 1080,
          frame_rate_limit: 15,
          bitrate_limit_kbps: 4096,
          stream_uri: 'rtsp://camera.local/stream',
          stream_error: null,
        },
      ],
    })
    const onCreateCamera = vi.fn().mockResolvedValue({ ok: true })
    const onClose = vi.fn()
    const user = userEvent.setup()
    render(
      <OnvifDiscoveryWizard
        applyChangesImmediately={true}
        createPending={false}
        isMutating={false}
        onCreateCamera={onCreateCamera}
        onClose={onClose}
      />,
    )

    // When: Operator completes discover -> probe -> create flow
    await user.click(screen.getByRole('button', { name: 'Run discovery scan' }))
    await user.click(await screen.findByRole('button', { name: /192.168.1.21/ }))
    await user.type(screen.getByLabelText('ONVIF username'), 'admin')
    await user.type(screen.getByLabelText('ONVIF password'), 'secret')
    await user.click(screen.getByRole('button', { name: 'Probe camera' }))
    await user.click(await screen.findByRole('button', { name: 'Create camera' }))

    // Then: Create payload includes RTSP URI with credentials and applyChanges=true
    expect(onCreateCamera).toHaveBeenCalledTimes(1)
    expect(onCreateCamera.mock.calls[0]?.[0]).toMatchObject({
      enabled: true,
      source_backend: 'rtsp',
      source_config: {
        rtsp_url: 'rtsp://admin:secret@camera.local/stream',
      },
    })
    expect(onCreateCamera.mock.calls[0]?.[1]).toBe(true)
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('shows unusable-profile guidance when probe returns no stream URIs', async () => {
    // Given: Probe succeeds but all profiles are missing stream_uri values
    discoverOnvifCamerasMock.mockResolvedValue([
      {
        ip: '192.168.1.22',
        xaddr: 'http://192.168.1.22/onvif/device_service',
        scopes: [],
        types: [],
      },
    ])
    probeOnvifCameraMock.mockResolvedValue({
      device: {
        manufacturer: 'Acme',
        model: 'CamLite',
        firmware_version: '1.0.0',
        serial_number: 'SN999',
        hardware_id: 'HW999',
      },
      profiles: [
        {
          token: 'main',
          name: 'Main stream',
          video_encoding: 'H264',
          width: 1280,
          height: 720,
          frame_rate_limit: 10,
          bitrate_limit_kbps: 2048,
          stream_uri: null,
          stream_error: 'Unauthorized',
        },
      ],
    })
    const onCreateCamera = vi.fn().mockResolvedValue({ ok: true })
    const onClose = vi.fn()
    const user = userEvent.setup()
    render(
      <OnvifDiscoveryWizard
        applyChangesImmediately={false}
        createPending={false}
        isMutating={false}
        onCreateCamera={onCreateCamera}
        onClose={onClose}
      />,
    )

    // When: Operator probes selected camera and receives no usable stream URI
    await user.click(screen.getByRole('button', { name: 'Run discovery scan' }))
    await user.click(await screen.findByRole('button', { name: /192.168.1.22/ }))
    await user.type(screen.getByLabelText('ONVIF username'), 'admin')
    await user.type(screen.getByLabelText('ONVIF password'), 'secret')
    await user.click(screen.getByRole('button', { name: 'Probe camera' }))

    // Then: Wizard surfaces guidance and keeps create blocked
    expect(
      await screen.findByText(
        'Probe succeeded, but no usable stream URI was returned. Retry with different credentials or choose another camera.',
      ),
    ).toBeTruthy()
    expect((screen.getByRole('button', { name: 'Create camera' }) as HTMLButtonElement).disabled).toBe(
      true,
    )
    expect(onCreateCamera).not.toHaveBeenCalled()
  })

  it('keeps the last valid ONVIF port when probe port input is cleared', async () => {
    // Given: Discovery returns one camera and probe endpoint succeeds
    discoverOnvifCamerasMock.mockResolvedValue([
      {
        ip: '192.168.1.24',
        xaddr: 'http://192.168.1.24/onvif/device_service',
        scopes: [],
        types: [],
      },
    ])
    probeOnvifCameraMock.mockResolvedValue({
      device: {
        manufacturer: 'Acme',
        model: 'CamPro',
        firmware_version: '1.0.0',
        serial_number: 'SN123',
        hardware_id: 'HW456',
      },
      profiles: [],
    })
    const onCreateCamera = vi.fn().mockResolvedValue({ ok: true })
    const onClose = vi.fn()
    const user = userEvent.setup()
    render(
      <OnvifDiscoveryWizard
        applyChangesImmediately={false}
        createPending={false}
        isMutating={false}
        onCreateCamera={onCreateCamera}
        onClose={onClose}
      />,
    )

    // When: Operator sets a valid port, then clears the port field, then probes
    await user.click(screen.getByRole('button', { name: 'Run discovery scan' }))
    await user.click(await screen.findByRole('button', { name: /192.168.1.24/ }))
    const portInput = screen.getByLabelText('ONVIF port')
    fireEvent.change(portInput, { target: { value: '554' } })
    fireEvent.change(portInput, { target: { value: '' } })
    await user.type(screen.getByLabelText('ONVIF username'), 'admin')
    await user.type(screen.getByLabelText('ONVIF password'), 'secret')
    await user.click(screen.getByRole('button', { name: 'Probe camera' }))

    // Then: Probe request uses the last valid port instead of silently resetting to 80
    expect(probeOnvifCameraMock).toHaveBeenCalledTimes(1)
    expect(probeOnvifCameraMock.mock.calls[0]?.[0]).toMatchObject({
      host: '192.168.1.24',
      port: 554,
      username: 'admin',
      password: 'secret',
    })
  })

  it('does not show discover empty-state copy before first scan request', () => {
    // Given: Wizard opened at initial discovery step
    const onCreateCamera = vi.fn().mockResolvedValue({ ok: true })
    const onClose = vi.fn()
    render(
      <OnvifDiscoveryWizard
        applyChangesImmediately={false}
        createPending={false}
        isMutating={false}
        onCreateCamera={onCreateCamera}
        onClose={onClose}
      />,
    )

    // When: No discovery scan has been requested yet
    const emptyStateText = screen.queryByText(
      'No cameras found. Make sure cameras are on the same subnet.',
    )

    // Then: Wizard should avoid showing "no cameras found" preemptively
    expect(emptyStateText).toBeNull()
  })

  it('shows name-focused conflict guidance when camera create returns CAMERA_ALREADY_EXISTS', async () => {
    // Given: Create call fails with camera-already-exists API error
    discoverOnvifCamerasMock.mockResolvedValue([
      {
        ip: '192.168.1.23',
        xaddr: 'http://192.168.1.23/onvif/device_service',
        scopes: [],
        types: [],
      },
    ])
    probeOnvifCameraMock.mockResolvedValue({
      device: {
        manufacturer: 'Acme',
        model: 'CamPro',
        firmware_version: '1.0.0',
        serial_number: 'SN123',
        hardware_id: 'HW456',
      },
      profiles: [
        {
          token: 'main',
          name: 'Main stream',
          video_encoding: 'H264',
          width: 1920,
          height: 1080,
          frame_rate_limit: 15,
          bitrate_limit_kbps: 4096,
          stream_uri: 'rtsp://camera.local/stream',
          stream_error: null,
        },
      ],
    })
    const duplicateNameError = new APIError(
      'Camera already exists: acme-campro-192-168-1-23',
      409,
      { detail: 'Camera already exists', error_code: 'CAMERA_ALREADY_EXISTS' },
      'CAMERA_ALREADY_EXISTS',
    )
    const onCreateCamera = vi.fn().mockResolvedValue({ ok: false, error: duplicateNameError })
    const onClose = vi.fn()
    const user = userEvent.setup()
    render(
      <OnvifDiscoveryWizard
        applyChangesImmediately={false}
        createPending={false}
        isMutating={false}
        onCreateCamera={onCreateCamera}
        onClose={onClose}
      />,
    )

    // When: Operator completes wizard and create mutation fails
    await user.click(screen.getByRole('button', { name: 'Run discovery scan' }))
    await user.click(await screen.findByRole('button', { name: /192.168.1.23/ }))
    await user.type(screen.getByLabelText('ONVIF username'), 'admin')
    await user.type(screen.getByLabelText('ONVIF password'), 'secret')
    await user.click(screen.getByRole('button', { name: 'Probe camera' }))
    await user.click(await screen.findByRole('button', { name: 'Create camera' }))
    // Then: UI should guide operator to rename camera instead of generic failure text
    expect(
      await screen.findByText('Camera name already exists. Choose a different name and retry.'),
    ).toBeTruthy()
    expect(onClose).not.toHaveBeenCalled()
  })
})
