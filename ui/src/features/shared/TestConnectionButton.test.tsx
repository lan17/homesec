// @vitest-environment happy-dom

import { afterEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

import { TestConnectionButton } from './TestConnectionButton'

function response(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), { status, headers: { 'content-type': 'application/json' } })
}

describe('TestConnectionButton', () => {
  afterEach(() => { cleanup(); vi.restoreAllMocks() })

  it('accepts the current result across equivalent request-object rerenders', async () => {
    // Given: The shared button uses the real client and a deferred HTTP probe
    let finish: ((value: Response) => void) | undefined
    const fetch = vi.spyOn(globalThis, 'fetch').mockImplementation(async () =>
      new Promise<Response>((resolve) => { finish = resolve }))
    const onResult = vi.fn()
    const client = new QueryClient()
    const button = () => <QueryClientProvider client={client}>
      <TestConnectionButton request={{ type: 'storage', backend: 'local', config: { root: '/clips' } }}
        result={null} onResult={onResult} />
    </QueryClientProvider>
    const { rerender } = render(button())
    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: 'Run connection test' }))
    await waitFor(() => expect(finish).toBeTruthy())

    // When: A parent renders equivalent inputs while the current probe completes
    rerender(button())
    await act(async () => { finish?.(response({ success: true, message: 'Current inputs work', latency_ms: 1 })) })

    // Then: Object identity changes do not cancel a valid test or alter its provider request
    await waitFor(() => expect(onResult).toHaveBeenCalledWith(expect.objectContaining({ success: true })))
    expect(fetch.mock.calls[0]?.[1]?.signal?.aborted).toBe(false)
    expect(JSON.parse(String(fetch.mock.calls[0]?.[1]?.body))).toEqual({
      type: 'storage', backend: 'local', config: { root: '/clips' },
    })
  })

  it.each(['success', 'error'])('cancels and drops a late %s after unmount', async (outcome) => {
    // Given: A probe can complete even if a transport ignores cancellation
    let finish: ((value: Response) => void) | undefined
    const fetch = vi.spyOn(globalThis, 'fetch').mockImplementation(async () =>
      new Promise<Response>((resolve) => { finish = resolve }))
    const onResult = vi.fn()
    const client = new QueryClient()
    const { unmount } = render(<QueryClientProvider client={client}>
      <TestConnectionButton request={{ type: 'storage', backend: 'local', config: { root: '/clips' } }}
        result={null} onResult={onResult} />
    </QueryClientProvider>)
    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: 'Run connection test' }))
    await waitFor(() => expect(finish).toBeTruthy())

    // When: Navigation removes the shared button before the HTTP result arrives
    unmount()
    await act(async () => {
      finish?.(outcome === 'success' ? response({ success: true, message: 'Old success', latency_ms: 1 })
        : response({ detail: 'Old failure' }, 500))
    })

    // Then: The client signal is cancelled and no parent receives a stale result
    expect(fetch.mock.calls[0]?.[1]?.signal?.aborted).toBe(true)
    expect(onResult).not.toHaveBeenCalled()
  })

  it('does not restore a completed error after inputs change A to B to A', async () => {
    // Given: The currently tested inputs have returned a connection error
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(response({ detail: 'Earlier inputs failed' }, 500))
    const client = new QueryClient()
    const button = (root: string) => <QueryClientProvider client={client}>
      <TestConnectionButton request={{ type: 'storage', backend: 'local', config: { root } }}
        result={null} onResult={vi.fn()} />
    </QueryClientProvider>
    const { rerender } = render(button('/a'))
    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: 'Run connection test' }))
    await screen.findByText('Earlier inputs failed')

    // When: The inputs change and later return to the original value without a new test
    rerender(button('/b'))
    rerender(button('/a'))

    // Then: The earlier attempt stays invalidated after the intervening edit
    expect(screen.queryByText('Earlier inputs failed')).toBeNull()
  })
})
