import { useState } from 'react'

import type { TestConnectionResponse } from '../../../api/generated/types'
import type { CameraAddBackend, CameraAddStage, CameraAddState } from './types'

const INITIAL_STATE: CameraAddState = {
  step: 'pick-backend',
  backend: null,
  backendStepIndex: 0,
  config: {},
  cameraName: '',
  testResult: null,
}

interface SelectBackendOptions {
  backend: CameraAddBackend
  defaultConfig: Record<string, unknown>
  suggestedName: string
}

export interface UseCameraAddStateResult {
  state: CameraAddState
  selectBackend: (options: SelectBackendOptions) => void
  updateConfig: (nextConfig: Record<string, unknown>) => void
  setCameraName: (cameraName: string) => void
  setTestResult: (result: TestConnectionResponse | null) => void
  goNext: (backendStepCount: number) => void
  goBack: (backendStepCount: number) => void
  reset: () => void
}

function nextStage(
  stage: CameraAddStage,
  backendStepIndex: number,
  backendStepCount: number,
): { stage: CameraAddStage; backendStepIndex: number } {
  if (stage === 'pick-backend') {
    return { stage: 'configure', backendStepIndex: 0 }
  }
  if (stage === 'configure') {
    if (backendStepIndex < backendStepCount - 1) {
      return {
        stage: 'configure',
        backendStepIndex: backendStepIndex + 1,
      }
    }
    return { stage: 'test', backendStepIndex }
  }
  if (stage === 'test') {
    return { stage: 'confirm', backendStepIndex }
  }
  return { stage, backendStepIndex }
}

function previousStage(
  stage: CameraAddStage,
  backendStepIndex: number,
  backendStepCount: number,
): { stage: CameraAddStage; backendStepIndex: number } {
  if (stage === 'configure') {
    if (backendStepIndex > 0) {
      return {
        stage: 'configure',
        backendStepIndex: backendStepIndex - 1,
      }
    }
    return {
      stage: 'pick-backend',
      backendStepIndex: 0,
    }
  }
  if (stage === 'test') {
    return {
      stage: 'configure',
      backendStepIndex: Math.max(backendStepCount - 1, 0),
    }
  }
  if (stage === 'confirm') {
    return {
      stage: 'test',
      backendStepIndex,
    }
  }
  return { stage, backendStepIndex }
}

export function useCameraAddState(): UseCameraAddStateResult {
  const [state, setState] = useState<CameraAddState>(INITIAL_STATE)

  function selectBackend({ backend, defaultConfig, suggestedName }: SelectBackendOptions): void {
    setState({
      step: 'configure',
      backend,
      backendStepIndex: 0,
      config: defaultConfig,
      cameraName: suggestedName,
      testResult: null,
    })
  }

  function updateConfig(nextConfig: Record<string, unknown>): void {
    setState((current) => ({
      ...current,
      config: nextConfig,
      testResult: null,
    }))
  }

  function setCameraName(cameraName: string): void {
    setState((current) => ({ ...current, cameraName }))
  }

  function setTestResult(result: TestConnectionResponse | null): void {
    setState((current) => ({ ...current, testResult: result }))
  }

  function goNext(backendStepCount: number): void {
    setState((current) => {
      const transition = nextStage(current.step, current.backendStepIndex, backendStepCount)
      return {
        ...current,
        step: transition.stage,
        backendStepIndex: transition.backendStepIndex,
      }
    })
  }

  function goBack(backendStepCount: number): void {
    setState((current) => {
      const transition = previousStage(current.step, current.backendStepIndex, backendStepCount)
      return {
        ...current,
        step: transition.stage,
        backendStepIndex: transition.backendStepIndex,
      }
    })
  }

  function reset(): void {
    setState(INITIAL_STATE)
  }

  return {
    state,
    selectBackend,
    updateConfig,
    setCameraName,
    setTestResult,
    goNext,
    goBack,
    reset,
  }
}

