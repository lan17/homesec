// @vitest-environment happy-dom

import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'

import { CameraCard } from './CameraCard'
import { EmptyState } from './EmptyState'
import { EventCard } from './EventCard'
import { FilterChips } from './FilterChips'
import { MediaPanel } from './MediaPanel'
import { MobileBottomNav } from './MobileBottomNav'
import { ResponsivePageShell } from './ResponsivePageShell'
import { RiskBadge } from './RiskBadge'
import { StatusBadge } from './StatusBadge'
import { TechnicalDetailsDisclosure } from './TechnicalDetailsDisclosure'
import { riskToneForLevel } from './riskTone'

describe('shared UI primitives', () => {
  afterEach(() => {
    cleanup()
  })

  it('renders responsive page shell actions without coupling to page data', () => {
    // Given: A page using the shared responsive shell
    render(
      <ResponsivePageShell title="Live" lead="Camera-first view" actions={<button>Refresh</button>}>
        <p>Page body</p>
      </ResponsivePageShell>,
    )

    // When: The shell is rendered
    const heading = screen.getByRole('heading', { name: 'Live' })

    // Then: The page hierarchy and action region are available to concrete pages
    expect(heading).toBeTruthy()
    expect(screen.getByText('Camera-first view')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Refresh' })).toBeTruthy()
    expect(screen.getByText('Page body')).toBeTruthy()
  })

  it('renders mobile bottom navigation from caller-provided links', () => {
    // Given: Mobile nav is configured with homeowner destinations only
    render(
      <MemoryRouter initialEntries={['/events']}>
        <MobileBottomNav
          links={[
            { to: '/live', label: 'Live' },
            { to: '/events', label: 'Events' },
            { to: '/settings', label: 'Settings' },
          ]}
        />
      </MemoryRouter>,
    )

    // When: The mobile navigation is inspected
    const mobileNav = screen.getByRole('navigation', { name: 'Mobile primary' })

    // Then: It exposes the configured links without adding System implicitly
    expect(within(mobileNav).getByRole('link', { name: 'Live' }).getAttribute('href')).toBe('/live')
    expect(within(mobileNav).getByRole('link', { name: 'Events' }).getAttribute('href')).toBe('/events')
    expect(within(mobileNav).getByRole('link', { name: 'Settings' }).getAttribute('href')).toBe('/settings')
    expect(within(mobileNav).queryByRole('link', { name: 'System' })).toBeNull()
  })

  it('renders status and risk badges from current API values', () => {
    // Given: Status and risk values from existing frontend data
    render(
      <div>
        <StatusBadge tone="healthy">Online</StatusBadge>
        <RiskBadge level="high" />
        <RiskBadge level={null} />
      </div>,
    )

    // When: Badges are rendered
    const badges = screen.getAllByRole('status')

    // Then: Existing values are displayed without requiring extra backend reason codes
    expect(badges.map((badge) => badge.textContent)).toEqual([
      'Online',
      'High',
      'Risk unavailable',
    ])
    expect(riskToneForLevel('moderate')).toBe('medium')
    expect(riskToneForLevel('unknown backend value')).toBe('unknown')
  })

  it('hides technical details behind a disclosure by default', () => {
    // Given: Technical fields need a reusable collapsed container
    render(
      <TechnicalDetailsDisclosure>
        <pre>{'{"clip_id":"abc"}'}</pre>
      </TechnicalDetailsDisclosure>,
    )

    // When: The disclosure is rendered
    const summary = screen.getByText('Advanced details')
    const details = summary.closest('details')

    // Then: The details are collapsed until the user opens them
    expect(details?.hasAttribute('open')).toBe(false)
    expect(screen.getByText('{"clip_id":"abc"}')).toBeTruthy()
  })

  it('invokes quick filter chip selections with active state exposed', async () => {
    // Given: Quick filters are rendered from caller-owned state
    const user = userEvent.setup()
    const onSelect = vi.fn()
    render(
      <FilterChips
        chips={[
          { id: 'today', label: 'Today', active: true },
          { id: 'alerts', label: 'Alerts' },
        ]}
        onSelect={onSelect}
      />,
    )

    // When: A chip is selected
    await user.click(screen.getByRole('button', { name: 'Alerts' }))

    // Then: The caller receives the selected id and active state remains accessible
    expect(onSelect).toHaveBeenCalledWith('alerts')
    expect(screen.getByRole('button', { name: 'Today' }).getAttribute('aria-pressed')).toBe('true')
  })

  it('renders empty and media states without page-specific data', () => {
    // Given: Shared empty and media states are used by future M1 pages
    render(
      <div>
        <EmptyState title="No events" description="Try a different filter." action={<a href="/events">Events</a>} />
        <MediaPanel title="Event video" status={<StatusBadge tone="unknown">Idle</StatusBadge>} />
      </div>,
    )

    // When: The primitives render their default surfaces
    const mediaPanel = screen.getByRole('heading', { name: 'Event video' }).closest('section')

    // Then: Empty copy, action, and media placeholder are available
    expect(screen.getByRole('heading', { name: 'No events' })).toBeTruthy()
    expect(screen.getByRole('link', { name: 'Events' }).getAttribute('href')).toBe('/events')
    expect(mediaPanel ? within(mediaPanel).getByText('Media unavailable') : null).toBeTruthy()
  })

  it('renders camera and event cards with caller-provided metadata and actions', () => {
    // Given: Card primitives receive lightweight display props, not API models
    render(
      <div>
        <CameraCard
          title="Front door"
          subtitle="rtsp"
          status={<StatusBadge tone="healthy">Online</StatusBadge>}
          meta={[{ label: 'Last seen', value: 'Today' }]}
          actions={<button>View Events</button>}
        />
        <EventCard
          camera="Driveway"
          time="Today, 8:05 AM"
          summary="Person near the driveway"
          risk={<RiskBadge level="medium" />}
          meta={[{ label: 'Objects', value: 'person' }]}
          actions={<a href="/events/clip-1">Open</a>}
        />
      </div>,
    )

    // When: The cards are rendered
    const cameraCard = screen.getByRole('heading', { name: 'Front door' }).closest('article')
    const eventCard = screen.getByText('Driveway').closest('article')

    // Then: Common card structure can support Live and Events without data fetching
    expect(cameraCard ? within(cameraCard).getByText('Last seen') : null).toBeTruthy()
    expect(cameraCard ? within(cameraCard).getByRole('button', { name: 'View Events' }) : null).toBeTruthy()
    expect(eventCard ? within(eventCard).getByText('Person near the driveway') : null).toBeTruthy()
    expect(eventCard ? within(eventCard).getByRole('link', { name: 'Open' }).getAttribute('href') : null).toBe('/events/clip-1')
  })
})
