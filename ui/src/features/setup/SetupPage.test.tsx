// @vitest-environment happy-dom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'

import { WIZARD_STATE_STORAGE_KEY } from './useWizardState'
import { SetupPage } from './SetupPage'

vi.mock('./steps/CameraStep', () => ({
  CameraStep: (props: {
    existingCameraNames: readonly string[]
    onComplete: () => void
    onUpdateData: (data: {
      name: string
      enabled: boolean
      source_backend: 'rtsp'
      source_config: Record<string, unknown>
    }) => void
    onSkip: () => void
  }) => (
    <section>
      <p>Mock camera step</p>
      <button
        type="button"
        onClick={() => {
          props.onUpdateData({
            name: 'wizard_camera',
            enabled: true,
            source_backend: 'rtsp',
            source_config: {
              rtsp_url: 'rtsp://camera.local/stream',
            },
          })
          props.onComplete()
        }}
      >
        Complete camera step
      </button>
      <button type="button" onClick={props.onSkip}>
        Skip camera step
      </button>
    </section>
  ),
}))

vi.mock('./steps/StorageStep', () => ({
  StorageStep: () => <section>Mock storage step</section>,
}))

vi.mock('./steps/DetectionStep', () => ({
  DetectionStep: () => <section>Mock detection step</section>,
}))

function renderSetupPage(initialEntry: string) {
  render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <Routes>
        <Route path="/setup" element={<SetupPage />} />
      </Routes>
    </MemoryRouter>,
  )
}

describe('SetupPage', () => {
  beforeEach(() => {
    window.localStorage.clear()
  })

  afterEach(() => {
    cleanup()
  })

  it('renders camera step component on wizard camera stage', () => {
    // Given: Wizard URL points to camera step
    renderSetupPage('/setup?step=1')

    // When: Setup page renders active step content
    const marker = screen.getByText('Mock camera step')

    // Then: Camera step wrapper is mounted for step 2
    expect(marker).toBeTruthy()
    expect(screen.getByRole('heading', { name: 'Camera' })).toBeTruthy()
  })

  it('advances to storage and keeps camera draft out of localStorage on completion', async () => {
    // Given: Wizard starts on camera step
    const user = userEvent.setup()
    renderSetupPage('/setup?step=1')

    // When: Camera step reports completion with a draft payload
    await user.click(screen.getByRole('button', { name: 'Complete camera step' }))

    // Then: Wizard advances and persisted state excludes non-persistent camera draft
    await waitFor(() => {
      expect(screen.getByRole('heading', { name: 'Storage' })).toBeTruthy()
    })
    expect(screen.getByText('Mock storage step')).toBeTruthy()
    const raw = window.localStorage.getItem(WIZARD_STATE_STORAGE_KEY)
    expect(raw).toBeTruthy()
    const parsed = JSON.parse(raw ?? '{}') as { stepData?: Record<string, unknown> }
    expect(parsed.stepData?.camera).toBeUndefined()
  })

  it('skips camera step and marks progress as skipped when camera flow is canceled', async () => {
    // Given: Wizard starts on camera step
    const user = userEvent.setup()
    renderSetupPage('/setup?step=1')

    // When: Operator cancels camera flow
    await user.click(screen.getByRole('button', { name: 'Skip camera step' }))

    // Then: Wizard advances and camera progress indicator is marked skipped
    await waitFor(() => {
      expect(screen.getByRole('heading', { name: 'Storage' })).toBeTruthy()
    })
    const progress = screen.getByRole('list', { name: 'Setup steps' })
    const cameraStep = within(progress).getByText('Camera').closest('li')
    expect(cameraStep?.className).toContain('wizard__progress-step--skipped')
  })

  it('renders detection step component on wizard detection stage', () => {
    // Given: Wizard URL points to detection step
    renderSetupPage('/setup?step=3')

    // When: Setup page renders active step content
    const marker = screen.getByText('Mock detection step')

    // Then: Detection step wrapper is mounted for step 4
    expect(marker).toBeTruthy()
    expect(screen.getByRole('heading', { name: 'Detection' })).toBeTruthy()
  })
})
