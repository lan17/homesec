// @vitest-environment happy-dom

import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'

import { StorageConfigForm } from './StorageConfigForm'
import type { StorageFormState } from './types'

describe('StorageConfigForm', () => {
  afterEach(() => {
    cleanup()
  })

  it('switches backend and resets config to backend defaults', async () => {
    // Given: Form starts on local backend with metadata-driven backend labels/defaults
    const onChange = vi.fn()
    const user = userEvent.setup()

    function Harness() {
      const [value, setValue] = useState<StorageFormState>({
        backend: 'local',
        config: { root: './storage' },
      })
      return (
        <StorageConfigForm
          value={value}
          backends={[
            {
              backend: 'local',
              label: 'Local FS',
              description: 'Store on local disk',
              config_schema: {},
              fields: [
                {
                  name: 'root',
                  type: 'string',
                  required: true,
                  description: 'Root path',
                  default: './storage',
                  secret: false,
                },
              ],
              secret_fields: [],
            },
            {
              backend: 'dropbox',
              label: 'Dropbox Cloud',
              description: 'Upload to Dropbox',
              config_schema: {},
              fields: [
                {
                  name: 'root',
                  type: 'string',
                  required: true,
                  description: 'Dropbox root',
                  default: null,
                  secret: false,
                },
                {
                  name: 'token_env',
                  type: 'string',
                  required: true,
                  description: 'Token env var',
                  default: 'DROPBOX_TOKEN',
                  secret: false,
                },
              ],
              secret_fields: [],
            },
          ]}
          onChange={(nextValue) => {
            onChange(nextValue)
            setValue(nextValue)
          }}
        />
      )
    }

    render(<Harness />)

    // When: Operator switches to Dropbox backend
    await user.click(screen.getByRole('button', { name: 'Dropbox Cloud' }))

    // Then: Form switches backend and applies metadata-aware defaults
    expect(onChange).toHaveBeenCalled()
    expect(onChange.mock.calls.at(-1)?.[0]).toMatchObject({
      backend: 'dropbox',
      config: { root: '/homesec', token_env: 'DROPBOX_TOKEN' },
    })
    expect(screen.getByLabelText('Dropbox root path')).toBeTruthy()
  })

  it('emits config changes from active backend form', async () => {
    // Given: Form is on local backend
    const onChange = vi.fn()
    const user = userEvent.setup()

    function Harness() {
      const [value, setValue] = useState<StorageFormState>({
        backend: 'local',
        config: { root: './storage' },
      })
      return (
        <StorageConfigForm
          value={value}
          backends={null}
          onChange={(nextValue) => {
            onChange(nextValue)
            setValue(nextValue)
          }}
        />
      )
    }

    render(<Harness />)

    // When: Operator updates local storage root path
    await user.clear(screen.getByLabelText('Storage root directory'))
    await user.type(screen.getByLabelText('Storage root directory'), '/var/lib/homesec')

    // Then: Form emits updated local config in storage state
    expect(onChange).toHaveBeenCalled()
    expect(onChange.mock.calls.at(-1)?.[0]).toMatchObject({
      backend: 'local',
      config: { root: '/var/lib/homesec' },
    })
  })
})
