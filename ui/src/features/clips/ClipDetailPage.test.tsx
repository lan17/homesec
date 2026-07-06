// @vitest-environment happy-dom

import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter, Route, Routes } from 'react-router-dom'

import { APIError, type ClipListSnapshot } from '../../api/client'
import type { ClipResponse } from '../../api/generated/types'
import { QUERY_KEYS } from '../../api/hooks/queryKeys'
import type { useClipMediaUrl } from '../../api/hooks/useClipMediaUrl'
import type { useClipQuery } from '../../api/hooks/useClipQuery'
import { ClipDetailPage } from './ClipDetailPage'
import { parseClipsQuery } from './queryParams'

const useClipQueryMock = vi.fn()
const useClipMediaUrlMock = vi.fn()

vi.mock('../../api/hooks/useClipQuery', () => ({
  useClipQuery: (...args: unknown[]) => useClipQueryMock(...args),
}))

vi.mock('../../api/hooks/useClipMediaUrl', () => ({
  useClipMediaUrl: (...args: unknown[]) => useClipMediaUrlMock(...args),
}))

function makeClip(id: string, overrides: Partial<ClipResponse> = {}): ClipResponse {
  return {
    id,
    camera: 'front_door',
    status: 'done',
    created_at: '2026-02-14T18:30:00.000Z',
    activity_type: 'package_drop',
    risk_level: 'high',
    summary: 'Package left near the front door.',
    detected_objects: ['person', 'package'],
    alerted: true,
    storage_uri: `dropbox:/clips/${id}.mp4`,
    view_url: `https://storage.example/${id}`,
    ...overrides,
  }
}

function renderDetail({
  route = '/events/clip-2?detected=any',
  clip = makeClip('clip-2'),
  cachedClips,
  clipQuery = {},
  mediaQuery = {},
}: {
  route?: string
  clip?: ClipResponse | undefined
  cachedClips?: ClipResponse[]
  clipQuery?: Partial<ReturnType<typeof useClipQuery>>
  mediaQuery?: Partial<ReturnType<typeof useClipMediaUrl>>
} = {}) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  })
  if (cachedClips) {
    const query = parseClipsQuery(new URLSearchParams('detected=any'))
    queryClient.setQueryData<ClipListSnapshot>(QUERY_KEYS.clips(query), {
      httpStatus: 200,
      clips: cachedClips,
      limit: 25,
      next_cursor: null,
      has_more: false,
    })
  }

  useClipQueryMock.mockReturnValue({
    data: clip,
    isPending: false,
    isFetching: false,
    error: null,
    refetch: vi.fn().mockResolvedValue(undefined),
    ...clipQuery,
  } as unknown as ReturnType<typeof useClipQuery>)

  useClipMediaUrlMock.mockReturnValue({
    mediaUrl: '/api/v1/clips/clip-2/media',
    expiresAt: null,
    usesToken: false,
    isPending: false,
    error: null,
    refresh: vi.fn().mockResolvedValue('/api/v1/clips/clip-2/media'),
    ...mediaQuery,
  } as unknown as ReturnType<typeof useClipMediaUrl>)

  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[route]}>
        <Routes>
          <Route path="/events/:clipId" element={<ClipDetailPage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('ClipDetailPage', () => {
  afterEach(() => {
    cleanup()
    useClipQueryMock.mockReset()
    useClipMediaUrlMock.mockReset()
  })

  it('puts video, summary, and navigation ahead of technical metadata', () => {
    // Given: A detail page opened from a loaded filtered event list
    renderDetail({
      cachedClips: [
        makeClip('clip-1', { summary: 'Earlier package event.' }),
        makeClip('clip-2'),
        makeClip('clip-3', { summary: 'Next package event.' }),
      ],
    })

    // When: Reading the refreshed detail page
    const technicalSummary = screen.getByText('Technical event details')
    const technicalDetails = technicalSummary.closest('details')

    // Then: Core review information is prominent and list context is preserved
    expect(screen.getByRole('heading', { name: 'Package Drop' })).toBeTruthy()
    expect(screen.getByText('Package left near the front door.')).toBeTruthy()
    expect(screen.getByText('person, package')).toBeTruthy()
    expect(screen.getByRole('link', { name: 'Back to Events' }).getAttribute('href')).toBe(
      '/events?detected=any',
    )
    expect(screen.getByRole('link', { name: 'Previous event' }).getAttribute('href')).toBe(
      '/events/clip-1?detected=any',
    )
    expect(screen.getByRole('link', { name: 'Next event' }).getAttribute('href')).toBe(
      '/events/clip-3?detected=any',
    )
    expect(technicalDetails?.hasAttribute('open')).toBe(false)
    expect(technicalDetails ? within(technicalDetails).getByText('clip-2') : null).toBeTruthy()
    expect(technicalDetails ? within(technicalDetails).getByText('dropbox:/clips/clip-2.mp4') : null).toBeTruthy()
  })

  it('disables previous and next controls when no client result window is loaded', () => {
    // Given: A user opens an event detail URL directly
    renderDetail()

    // When: There is no clips query cache containing the current event
    const previous = screen.getByText('Previous event')
    const next = screen.getByText('Next event')

    // Then: The page does not call a backend neighbor endpoint or invent navigation
    expect(previous.getAttribute('aria-disabled')).toBe('true')
    expect(next.getAttribute('aria-disabled')).toBe('true')
  })

  it('opens notification deep links on the event detail page with a list fallback', () => {
    // Given: iOS opens an event detail route from a notification
    renderDetail({
      route: '/events/clip-2?from=notification',
    })

    // When: The event detail renders
    const backToEvents = screen.getByRole('link', { name: 'Back to Events' })

    // Then: The detail content opens and the list fallback drops notification-only routing state
    expect(screen.getByRole('heading', { name: 'Package Drop' })).toBeTruthy()
    expect(screen.getByText('Package left near the front door.')).toBeTruthy()
    expect(backToEvents.getAttribute('href')).toBe('/events')
    expect(useClipMediaUrlMock).toHaveBeenCalledWith('clip-2')
  })

  it('strips notification source state from neighbor navigation', () => {
    // Given: A notification opens an event with cached neighboring events
    renderDetail({
      route: '/events/clip-2?from=notification&detected=any',
      cachedClips: [
        makeClip('clip-1', { summary: 'Earlier package event.' }),
        makeClip('clip-2'),
        makeClip('clip-3', { summary: 'Next package event.' }),
      ],
    })

    // When: Navigating around the cached event window
    const previous = screen.getByRole('link', { name: 'Previous event' })
    const next = screen.getByRole('link', { name: 'Next event' })

    // Then: Neighbor routes preserve list filters without carrying notification-only source state
    expect(previous.getAttribute('href')).toBe('/events/clip-1?detected=any')
    expect(next.getAttribute('href')).toBe('/events/clip-3?detected=any')
  })

  it('shows a notification-specific fallback when a cached event no longer exists', () => {
    // Given: A notification points at an event that the API no longer has but React Query has cached
    const missingEventError = new APIError(
      'Clip not found',
      404,
      { detail: 'Clip not found' },
      'CLIP_NOT_FOUND',
    )
    renderDetail({
      route: '/events/deleted-clip?from=notification',
      clip: makeClip('deleted-clip', { summary: 'Stale cached summary.' }),
      clipQuery: {
        error: missingEventError,
      },
    })

    // When: The detail route handles the missing event response
    const fallback = screen.getByRole('heading', { name: 'Event no longer available' })

    // Then: The page stays useful instead of rendering a blank detail view
    expect(fallback).toBeTruthy()
    expect(screen.getByText(/opened from this notification is no longer available/i)).toBeTruthy()
    expect(screen.getByRole('link', { name: 'Back to Events' }).getAttribute('href')).toBe('/events')
    expect(screen.queryByText('Event video')).toBeNull()
    expect(screen.queryByText('Stale cached summary.')).toBeNull()
    expect(useClipMediaUrlMock).toHaveBeenCalledWith(undefined)
  })

  it('keeps event metadata visible when playback is unavailable', () => {
    // Given: Event metadata loads but the media URL cannot be prepared
    renderDetail({
      mediaQuery: {
        mediaUrl: null,
        error: new Error('Media file is missing'),
      },
    })

    // When: The detail page renders the unavailable playback state
    const summary = screen.getByLabelText('Event summary')

    // Then: Review metadata remains visible alongside the playback problem
    expect(screen.getByText('Event video is not available for playback.')).toBeTruthy()
    expect(within(summary).getByRole('heading', { name: 'Package Drop' })).toBeTruthy()
    expect(within(summary).getByText('Package left near the front door.')).toBeTruthy()
    expect(within(summary).getByText('front_door')).toBeTruthy()
    expect(within(summary).getByText('High')).toBeTruthy()
    expect(within(summary).getByText('Media file is missing')).toBeTruthy()
  })
})
