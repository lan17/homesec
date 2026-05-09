// @vitest-environment happy-dom

import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen, within } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter, Route, Routes } from 'react-router-dom'

import type { ClipListSnapshot } from '../../api/client'
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
}: {
  route?: string
  clip?: ClipResponse
  cachedClips?: ClipResponse[]
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
  } as unknown as ReturnType<typeof useClipQuery>)

  useClipMediaUrlMock.mockReturnValue({
    mediaUrl: '/api/v1/clips/clip-2/media',
    expiresAt: null,
    usesToken: false,
    isPending: false,
    error: null,
    refresh: vi.fn().mockResolvedValue('/api/v1/clips/clip-2/media'),
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
    expect(screen.getByRole('link', { name: 'Back to events' }).getAttribute('href')).toBe(
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
})
