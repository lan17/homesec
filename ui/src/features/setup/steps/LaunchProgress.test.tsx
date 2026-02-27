// @vitest-environment happy-dom

import { afterEach, describe, expect, it } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'

import { LaunchProgress } from './LaunchProgress'

describe('LaunchProgress', () => {
  afterEach(() => {
    cleanup()
  })

  it('renders launching state details', () => {
    // Given: A launch progress view in launching state
    render(<LaunchProgress status="launching" />)

    // When: The component is rendered
    const badge = screen.getByText('Launching')
    const message = screen.getByText(
      'Writing config and waiting for HomeSec runtime startup...',
    )

    // Then: Launching status and guidance are visible
    expect(badge).toBeTruthy()
    expect(message).toBeTruthy()
  })

  it('renders failure details with alert text', () => {
    // Given: A failed launch with an actionable error message
    render(<LaunchProgress status="failed" error="Timed out waiting for startup." />)

    // When: The component renders failed progress state
    const badge = screen.getByText('Failed')
    const error = screen.getByRole('alert')

    // Then: Failure badge and error details are presented
    expect(badge).toBeTruthy()
    expect(error.textContent).toContain('Timed out waiting for startup.')
  })

  it('renders started state details', () => {
    // Given: A launch progress view after runtime startup succeeds
    render(<LaunchProgress status="started" />)

    // When: The component renders started state
    const badge = screen.getByText('Started')
    const message = screen.getByText('Setup complete. HomeSec runtime is healthy.')

    // Then: Success status and completion guidance are visible
    expect(badge).toBeTruthy()
    expect(message).toBeTruthy()
  })
})
