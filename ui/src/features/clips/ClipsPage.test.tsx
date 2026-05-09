// @vitest-environment happy-dom

import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'

import type { ClipListSnapshot } from '../../api/client'
import type { CameraResponse } from '../../api/generated/types'
import type { useCamerasQuery } from '../../api/hooks/useCamerasQuery'
import type { useClipsQuery } from '../../api/hooks/useClipsQuery'
import { ClipsPage } from './ClipsPage'

const useClipsQueryMock = vi.fn()
const useCamerasQueryMock = vi.fn()

vi.mock('../../api/hooks/useClipsQuery', () => ({
  useClipsQuery: (...args: unknown[]) => useClipsQueryMock(...args),
}))

vi.mock('../../api/hooks/useCamerasQuery', () => ({
  useCamerasQuery: () => useCamerasQueryMock(),
}))

function makeCamera(name: string): CameraResponse {
  return {
    name,
    enabled: true,
    healthy: true,
    last_heartbeat: 1_739_590_400,
    source_backend: 'rtsp',
    source_config: { rtsp_url: 'rtsp://***redacted***@camera.local/stream' },
  }
}

function renderEventsPage({
  clipList,
  cameras = [makeCamera('front_door')],
  route = '/events?detected=any',
}: {
  clipList: ClipListSnapshot
  cameras?: CameraResponse[]
  route?: string
}) {
  useClipsQueryMock.mockReturnValue({
    data: clipList,
    isPending: false,
    isFetching: false,
    error: null,
    refetch: vi.fn().mockResolvedValue(undefined),
  } as unknown as ReturnType<typeof useClipsQuery>)

  useCamerasQueryMock.mockReturnValue({
    data: cameras,
    isPending: false,
    isFetching: false,
    error: null,
    refetch: vi.fn().mockResolvedValue(undefined),
  } as unknown as ReturnType<typeof useCamerasQuery>)

  render(
    <MemoryRouter initialEntries={[route]}>
      <ClipsPage />
    </MemoryRouter>,
  )
}

describe('ClipsPage event list', () => {
  afterEach(() => {
    cleanup()
    useClipsQueryMock.mockReset()
    useCamerasQueryMock.mockReset()
  })

  it('renders grouped media-first event cards without the technical table default', () => {
    // Given: Existing clip list API data with event-facing metadata
    renderEventsPage({
      clipList: {
        httpStatus: 200,
        limit: 25,
        next_cursor: null,
        has_more: false,
        clips: [
          {
            id: 'clip-1',
            camera: 'front_door',
            status: 'done',
            created_at: '2026-02-14T18:30:00.000Z',
            activity_type: 'package_drop',
            risk_level: 'high',
            summary: 'Package left near the front door.',
            detected_objects: ['person', 'package'],
            alerted: true,
            storage_uri: 'dropbox:/clips/clip-1.mp4',
            view_url: null,
          },
          {
            id: 'clip-2',
            camera: 'driveway',
            status: 'uploaded',
            created_at: '2026-02-14T18:10:00.000Z',
            activity_type: 'vehicle',
            risk_level: 'low',
            summary: null,
            detected_objects: ['car'],
            alerted: false,
            storage_uri: null,
            view_url: null,
          },
        ],
      },
      cameras: [makeCamera('front_door'), makeCamera('driveway')],
    })

    // When: Reading the refreshed event list
    const cards = screen.getAllByRole('article')
    const firstCard = cards[0]
    const links = screen.getAllByRole('link', { name: 'View event' })

    // Then: The default list should prioritize what happened over raw clip fields
    expect(screen.queryByRole('columnheader', { name: 'ID' })).toBeNull()
    expect(screen.queryByText('clip-1')).toBeNull()
    expect(screen.queryByText('dropbox:/clips/clip-1.mp4')).toBeNull()
    expect(firstCard ? within(firstCard).getByText('Package Drop') : null).toBeTruthy()
    expect(firstCard ? within(firstCard).getByText('Package left near the front door.') : null).toBeTruthy()
    expect(firstCard ? within(firstCard).getByText('front_door') : null).toBeTruthy()
    expect(firstCard ? within(firstCard).getByText('person, package') : null).toBeTruthy()
    expect(firstCard ? within(firstCard).getByText('Alert sent') : null).toBeTruthy()
    expect(links[0]?.getAttribute('href')).toBe('/events/clip-1?detected=any')
  })

  it('shows a no-results state without a table shell', () => {
    // Given: The existing clip query returns no matches
    renderEventsPage({
      clipList: {
        httpStatus: 200,
        limit: 25,
        next_cursor: null,
        has_more: false,
        clips: [],
      },
    })

    // When: The Events page renders
    const emptyState = screen.getByRole('heading', { name: 'No events found' })

    // Then: Empty results should be homeowner-readable and avoid the old table shell
    expect(emptyState).toBeTruthy()
    expect(screen.queryByRole('table')).toBeNull()
  })
})
