// @vitest-environment happy-dom

import { describe, expect, it, vi } from 'vitest'

import { bootstrapHomeSecApp } from './bootstrap'

describe('bootstrapHomeSecApp', () => {
  it('waits for runtime API configuration before rendering', async () => {
    // Given: Runtime configuration that resolves asynchronously
    const events: string[] = []
    const rootElement = document.createElement('div')
    let finishInitialization: (() => void) | undefined
    const initializeRuntimeConfig = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          finishInitialization = () => {
            events.push('initialized')
            resolve()
          }
        }),
    )
    const render = vi.fn(() => {
      events.push('rendered')
    })

    // When: Bootstrapping the app before runtime config has finished loading
    const bootstrapPromise = bootstrapHomeSecApp({
      rootElement,
      initializeRuntimeConfig,
      render,
    })

    // Then: Rendering is held until initialization completes
    expect(initializeRuntimeConfig).toHaveBeenCalledTimes(1)
    expect(render).not.toHaveBeenCalled()

    finishInitialization?.()
    await bootstrapPromise

    expect(events).toEqual(['initialized', 'rendered'])
    expect(render).toHaveBeenCalledWith(rootElement, expect.anything())
  })
})
