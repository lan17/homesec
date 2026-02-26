// @vitest-environment happy-dom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { APIError } from '../../../api/errors'
import { WelcomeStep } from './WelcomeStep'

const usePreflightMutationMock = vi.fn()

vi.mock('../../../api/hooks/usePreflightMutation', () => ({
  usePreflightMutation: () => usePreflightMutationMock(),
}))

interface MockPreflightMutation {
  data: {
    all_passed: boolean
    checks: Array<{ name: string; passed: boolean; message: string; latency_ms: number | null }>
  } | undefined
  mutate: () => void
  isPending: boolean
  isError: boolean
  error: unknown
}

function mockPreflightMutation(
  overrides: Partial<MockPreflightMutation> = {},
): MockPreflightMutation {
  const mutation: MockPreflightMutation = {
    data: undefined,
    mutate: vi.fn(),
    isPending: false,
    isError: false,
    error: null,
    ...overrides,
  }
  usePreflightMutationMock.mockReturnValue(mutation)
  return mutation
}

describe('WelcomeStep', () => {
  beforeEach(() => {
    usePreflightMutationMock.mockReset()
  })

  afterEach(() => {
    cleanup()
  })

  it('renders welcome copy and run-checks action', () => {
    // Given: Step mounted before any preflight execution
    mockPreflightMutation()

    // When: Rendering welcome step
    render(<WelcomeStep isComplete={false} onComplete={vi.fn()} />)

    // Then: Intro copy and preflight CTA are visible
    expect(
      screen.getByText(
        'HomeSec helps you capture, analyze, and review security clips from your cameras.',
      ),
    ).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Run checks' })).toBeTruthy()
  })

  it('runs preflight on click and renders pass/fail badges', async () => {
    // Given: Mutation already has check results available from backend
    const mutation = mockPreflightMutation({
      data: {
        all_passed: false,
        checks: [
          { name: 'postgres', passed: true, message: 'Connected', latency_ms: 5.8 },
          { name: 'ffmpeg', passed: false, message: 'ffmpeg not found', latency_ms: null },
        ],
      },
    })
    const user = userEvent.setup()

    // When: Operator triggers run checks and views results
    render(<WelcomeStep isComplete={false} onComplete={vi.fn()} />)
    await user.click(screen.getByRole('button', { name: 'Run checks again' }))

    // Then: Mutation is invoked and per-check status badges/messages are rendered
    expect(mutation.mutate).toHaveBeenCalledTimes(1)
    expect(screen.getByText('Postgres')).toBeTruthy()
    expect(screen.getByText('Ffmpeg')).toBeTruthy()
    expect(screen.getByText('Pass')).toBeTruthy()
    expect(screen.getByText('Fail')).toBeTruthy()
    expect(
      screen.getByText('Some checks failed. You can continue and return to fix these settings later.'),
    ).toBeTruthy()
  })

  it('shows actionable unauthorized error copy', () => {
    // Given: Preflight request failed with unauthorized API error
    mockPreflightMutation({
      isError: true,
      error: new APIError(
        'Unauthorized',
        401,
        { detail: 'Unauthorized', error_code: 'UNAUTHORIZED' },
        'UNAUTHORIZED',
      ),
    })

    // When: Rendering the step after failed preflight attempt
    render(<WelcomeStep isComplete={false} onComplete={vi.fn()} />)

    // Then: Operator gets explicit auth guidance rather than generic crash
    expect(
      screen.getByText(
        'Authentication required to run setup checks. Apply API key from Dashboard and retry.',
      ),
    ).toBeTruthy()
  })

  it('calls onComplete when mark-step-complete is clicked', async () => {
    // Given: Welcome step not yet completed
    mockPreflightMutation()
    const onComplete = vi.fn()
    const user = userEvent.setup()
    render(<WelcomeStep isComplete={false} onComplete={onComplete} />)

    // When: Operator marks welcome step complete
    await user.click(screen.getByRole('button', { name: 'Mark step complete' }))

    // Then: Step completion callback is invoked once
    expect(onComplete).toHaveBeenCalledTimes(1)
  })
})
