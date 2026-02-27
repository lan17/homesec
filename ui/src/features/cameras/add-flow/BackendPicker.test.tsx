// @vitest-environment happy-dom

import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { BackendPicker } from './BackendPicker'

describe('BackendPicker', () => {
  afterEach(() => {
    cleanup()
  })

  it('renders all configured backend cards and dispatches selected backend id', async () => {
    // Given: Backend picker with a select callback
    const onSelect = vi.fn()
    const user = userEvent.setup()
    render(<BackendPicker isMutating={false} onSelect={onSelect} onCancel={vi.fn()} />)

    // When: Operator chooses Local Folder backend
    await user.click(screen.getByRole('button', { name: 'Local Folder' }))

    // Then: Callback receives local_folder backend id
    expect(screen.getByRole('button', { name: 'RTSP' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'FTP' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'ONVIF Discovery' })).toBeTruthy()
    expect(onSelect).toHaveBeenCalledTimes(1)
    expect(onSelect).toHaveBeenCalledWith('local_folder')
  })
})

