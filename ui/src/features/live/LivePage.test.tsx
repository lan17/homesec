// @vitest-environment happy-dom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'

import type { CameraResponse } from '../../api/generated/types'
import { LivePage } from './LivePage'

const useCamerasQueryMock = vi.fn()
const useSetupRedirectMock = vi.fn()

vi.mock('../../api/hooks/useCamerasQuery', () => ({
  useCamerasQuery: () => useCamerasQueryMock(),
}))

vi.mock('../setup/useSetupRedirect', () => ({
  useSetupRedirect: () => useSetupRedirectMock(),
}))

vi.mock('../cameras/components/CameraPreviewPanel', () => ({
  CameraPreviewPanel: ({
    cameraName,
    showTalkControl,
    title,
  }: {
    cameraName: string
    showTalkControl?: boolean
    title?: string
  }) => (
    <div data-testid={`preview-${cameraName}`} data-talk={String(showTalkControl)}>
      {title} for {cameraName}
    </div>
  ),
}))

function makeCamera(overrides: Partial<CameraResponse> = {}): CameraResponse {
  return {
    name: 'front',
    enabled: true,
    healthy: true,
    source_backend: 'rtsp',
    last_heartbeat: 1_739_590_400,
    source_config: {},
    ...overrides,
  }
}

function renderLivePage(cameras: CameraResponse[]) {
  useSetupRedirectMock.mockReturnValue({ isChecking: false, shouldRedirect: false })
  useCamerasQueryMock.mockReturnValue({
    data: cameras,
    isPending: false,
    isFetching: false,
    error: null,
    refetch: vi.fn().mockResolvedValue(undefined),
  })

  render(
    <MemoryRouter initialEntries={['/live']}>
      <LivePage />
    </MemoryRouter>,
  )
}

function setCompactViewport(matches: boolean): void {
  Object.defineProperty(window, 'matchMedia', {
    configurable: true,
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches,
      media: query,
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })),
  })
}

describe('LivePage camera cards', () => {
  beforeEach(() => {
    setCompactViewport(false)
    useCamerasQueryMock.mockReset()
    useSetupRedirectMock.mockReset()
  })

  afterEach(() => {
    cleanup()
  })

  it('renders a camera-first grid with status, preview, and event actions', () => {
    // Given: Existing camera API data includes an RTSP camera
    renderLivePage([makeCamera({ name: 'front_door' })])

    // When: The Live page renders camera cards
    const card = screen.getByRole('heading', { name: 'front_door' }).closest('article')

    // Then: The card exposes status, preview, and event navigation using existing data
    expect(card ? within(card).getByText('Online') : null).toBeTruthy()
    expect(card ? within(card).getByTestId('preview-front_door') : null).toBeTruthy()
    expect(screen.getByTestId('preview-front_door').getAttribute('data-talk')).toBe('false')
    expect(card ? within(card).getByText('Last seen') : null).toBeTruthy()
    expect(
      card ? within(card).getByRole('link', { name: 'View Events' }).getAttribute('href') : null,
    ).toBe('/events?camera=front_door')
    expect(
      card
        ? within(card).getByRole('link', { name: 'Camera controls' }).getAttribute('href')
        : null,
    ).toBe('/cameras')
  })

  it('uses lightweight placeholders for cameras without live preview support', () => {
    // Given: Existing camera data includes a non-RTSP source and a disabled camera
    renderLivePage([
      makeCamera({ name: 'dropbox_uploads', source_backend: 'local_folder' }),
      makeCamera({ name: 'garage', enabled: false, healthy: false }),
    ])

    // When: The Live page renders the cards
    const disabledCard = screen.getByRole('heading', { name: 'garage' }).closest('article')

    // Then: The list does not auto-start preview streams and still shows clear status
    expect(screen.getByText('Live preview is not available for this camera source.')).toBeTruthy()
    expect(disabledCard ? within(disabledCard).getByText('Disabled') : null).toBeTruthy()
    expect(
      disabledCard
        ? within(disabledCard).getByText('Enable this camera to use live preview.')
        : null,
    ).toBeTruthy()
    expect(screen.queryByTestId('preview-dropbox_uploads')).toBeNull()
  })

  it('keeps only one live preview panel focused on compact viewports', async () => {
    // Given: A mobile viewport with multiple RTSP cameras
    const user = userEvent.setup()
    setCompactViewport(true)
    renderLivePage([
      makeCamera({ name: 'front_door' }),
      makeCamera({ name: 'driveway' }),
    ])

    // When: The compact Live view renders and the homeowner switches cameras
    expect(await screen.findByTestId('preview-front_door')).toBeTruthy()
    expect(screen.queryByTestId('preview-driveway')).toBeNull()
    await user.click(screen.getByRole('button', { name: 'Open live view' }))

    // Then: Only the selected camera exposes preview controls
    expect(await screen.findByTestId('preview-driveway')).toBeTruthy()
    expect(screen.queryByTestId('preview-front_door')).toBeNull()
    expect(screen.getByText('Open live view to start preview controls.')).toBeTruthy()
  })
})
