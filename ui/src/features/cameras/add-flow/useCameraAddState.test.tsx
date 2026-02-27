// @vitest-environment happy-dom

import { act, renderHook } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { useCameraAddState } from './useCameraAddState'

describe('useCameraAddState', () => {
  it('moves through configure steps, test, and confirm in order', () => {
    // Given: A selected backend with two configure steps
    const { result } = renderHook(() => useCameraAddState())
    act(() => {
      result.current.selectBackend({
        backend: 'rtsp',
        defaultConfig: { rtsp_url: 'rtsp://camera.local/live' },
        suggestedName: 'rtsp_1',
      })
    })

    // When: Advancing through the add-flow stages
    act(() => {
      result.current.goNext(2)
    })
    act(() => {
      result.current.goNext(2)
    })
    act(() => {
      result.current.goNext(2)
    })

    // Then: Hook reaches confirm stage and stays there on extra next
    expect(result.current.state.step).toBe('confirm')
    expect(result.current.state.backendStepIndex).toBe(1)
    act(() => {
      result.current.goNext(2)
    })
    expect(result.current.state.step).toBe('confirm')
  })

  it('moves backward from confirm to test to configure and picker', () => {
    // Given: Flow already advanced to confirm with three configure steps
    const { result } = renderHook(() => useCameraAddState())
    act(() => {
      result.current.selectBackend({
        backend: 'ftp',
        defaultConfig: { host: '0.0.0.0', port: 2121 },
        suggestedName: 'ftp_1',
      })
      result.current.goNext(3)
      result.current.goNext(3)
      result.current.goNext(3)
      result.current.goNext(3)
    })
    expect(result.current.state.step).toBe('confirm')

    // When: Navigating backward across stages
    act(() => {
      result.current.goBack(3)
    })
    act(() => {
      result.current.goBack(3)
    })
    act(() => {
      result.current.goBack(3)
    })
    act(() => {
      result.current.goBack(3)
    })
    act(() => {
      result.current.goBack(3)
    })

    // Then: Flow returns to backend picker stage
    expect(result.current.state.step).toBe('pick-backend')
    expect(result.current.state.backendStepIndex).toBe(0)
  })

  it('handles zero backend-step count without invalid indices', () => {
    // Given: A backend selection where configure step count is zero
    const { result } = renderHook(() => useCameraAddState())
    act(() => {
      result.current.selectBackend({
        backend: 'local_folder',
        defaultConfig: { watch_dir: './recordings' },
        suggestedName: 'local_folder_1',
      })
    })
    expect(result.current.state.step).toBe('configure')

    // When: Advancing and reversing with backendStepCount=0
    act(() => {
      result.current.goNext(0)
    })
    expect(result.current.state.step).toBe('test')
    act(() => {
      result.current.goBack(0)
    })

    // Then: Hook safely returns to configure index 0
    expect(result.current.state.step).toBe('configure')
    expect(result.current.state.backendStepIndex).toBe(0)
  })

  it('does not move backward from backend picker stage', () => {
    // Given: Initial picker state
    const { result } = renderHook(() => useCameraAddState())
    expect(result.current.state.step).toBe('pick-backend')

    // When: Back navigation is requested from picker stage
    act(() => {
      result.current.goBack(1)
    })

    // Then: Step remains unchanged
    expect(result.current.state.step).toBe('pick-backend')
    expect(result.current.state.backendStepIndex).toBe(0)
  })
})
