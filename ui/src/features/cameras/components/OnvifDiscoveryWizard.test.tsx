// @vitest-environment happy-dom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

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
        xaddr: 'http://192.168.1.20/onvif/device_service',
        scopes: ['onvif://scope/location/garage'],
        types: ['dn:NetworkVideoTransmitter'],
      },
    ])
    const onCreateCamera = vi.fn().mockResolvedValue(true)
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

    // Then: Wizard should advance to credential/probe step
    expect(screen.getByText('Step 2: Authenticate and probe')).toBeTruthy()
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
    const onCreateCamera = vi.fn().mockResolvedValue(true)
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
    const onCreateCamera = vi.fn().mockResolvedValue(true)
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
})
