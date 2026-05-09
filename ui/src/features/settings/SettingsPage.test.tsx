// @vitest-environment happy-dom

import { afterEach, describe, expect, it } from 'vitest'
import { cleanup, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'

import { SettingsPage } from './SettingsPage'

function renderSettingsPage(): void {
  render(
    <MemoryRouter initialEntries={['/settings']}>
      <SettingsPage />
    </MemoryRouter>,
  )
}

describe('SettingsPage', () => {
  afterEach(() => {
    cleanup()
  })

  it('separates homeowner settings from advanced system diagnostics', () => {
    // Given: The homeowner opens Settings from the mobile or desktop nav
    renderSettingsPage()

    // When: The settings cards render
    const settings = screen.getByRole('heading', { name: 'Settings' })
    const camerasCard = screen.getByRole('heading', { name: 'Cameras' }).closest('.card')
    const advancedCard = screen.getByRole('heading', { name: 'Advanced' }).closest('.card')

    // Then: User-facing configuration is visible and System is nested under Advanced
    expect(settings).toBeTruthy()
    expect(screen.getByRole('heading', { name: 'Cameras' })).toBeTruthy()
    expect(
      camerasCard
        ? within(camerasCard as HTMLElement).getByRole('link', { name: 'Camera setup' }).getAttribute('href')
        : null,
    ).toBe('/settings/cameras')
    expect(screen.getByRole('heading', { name: 'Notifications' })).toBeTruthy()
    expect(screen.getByRole('heading', { name: 'Detection' })).toBeTruthy()
    expect(screen.getByRole('heading', { name: 'Storage' })).toBeTruthy()
    expect(
      advancedCard
        ? within(advancedCard as HTMLElement).getByRole('link', { name: 'Open System' }).getAttribute('href')
        : null,
    ).toBe('/system')
  })
})
