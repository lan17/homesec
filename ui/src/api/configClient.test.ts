import { afterEach, describe, expect, it, vi } from 'vitest'

import { APIError, HomeSecApiClient } from './client'

const configPayload = {
  config: { storage: { backend: 'local', config: { root: './storage' } } },
  saved_config_version: 'saved', active_config_version: 'active', apply_required: 'restart',
}

function response(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), { status, headers: { 'content-type': 'application/json' } })
}

describe('configuration client', () => {
  afterEach(() => vi.restoreAllMocks())

  it('parses saved configuration and preserves server application status', async () => {
    // Given: Saved storage settings differ from active settings
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValue(response(configPayload))
    const client = new HomeSecApiClient('http://localhost:8081')
    // When: Reading configuration with explicit authentication
    const config = await client.getConfig({ apiKey: 'test-api-key' })
    // Then: Versions and application action stay distinct and request is authenticated
    expect(config).toEqual({ ...configPayload, httpStatus: 200 })
    expect(fetch).toHaveBeenCalledWith('http://localhost:8081/api/v1/config', expect.objectContaining({
      headers: expect.objectContaining({ Authorization: 'Bearer test-api-key' }),
    }))
  })

  it('sends a save-only patch with the captured version precondition', async () => {
    // Given: A patch changes only the configured local root
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValue(response(configPayload))
    const client = new HomeSecApiClient()
    const patch = { expected_config_version: 'draft-version', storage: { config: { root: '/clips' } } }
    // When: Saving the patch
    await client.patchConfig(patch)
    // Then: PATCH does not request application or rewrite the full configuration
    expect(fetch).toHaveBeenCalledWith('/api/v1/config', expect.objectContaining({
      method: 'PATCH', body: JSON.stringify(patch),
    }))
  })

  it('preserves canonical optimistic-concurrency errors', async () => {
    // Given: Another client already changed the configuration
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(response({
      detail: 'Saved settings changed', error_code: 'CONFIG_VERSION_CONFLICT',
    }, 409))
    // When: An outdated draft is saved
    const save = new HomeSecApiClient().patchConfig({ expected_config_version: 'outdated' })
    // Then: The caller can retain the draft and offer a refresh
    await expect(save).rejects.toMatchObject({ status: 409, errorCode: 'CONFIG_VERSION_CONFLICT' })
  })

  it.each([
    { ...configPayload, config: [] },
    { ...configPayload, saved_config_version: '' },
    { ...configPayload, active_config_version: undefined },
    { ...configPayload, apply_required: 'deploy' },
  ])('rejects invalid configuration payload %#', async (payload) => {
    // Given: An untrusted configuration endpoint has an invalid shape
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(response(payload))
    // When: The client reads the response
    const read = new HomeSecApiClient().getConfig()
    // Then: Invalid data never reaches feature editors
    await expect(read).rejects.toBeInstanceOf(APIError)
  })

  it('parses restart acceptance without claiming runtime activation', async () => {
    // Given: Server accepts a separate application request
    const payload = { accepted: true, message: 'Restart scheduled', action: 'restart',
      target_config_version: 'saved', target_generation: null }
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValue(response(payload, 202))
    // When: Applying the saved version
    const result = await new HomeSecApiClient().applyConfig({ expected_config_version: 'saved' })
    // Then: Acceptance metadata remains available for activation polling
    expect(result).toEqual({ ...payload, httpStatus: 202 })
    expect(fetch).toHaveBeenCalledWith('/api/v1/config/apply', expect.objectContaining({
      method: 'POST', body: JSON.stringify({ expected_config_version: 'saved' }),
    }))
  })

  it.each([-1, 1.5, '2', undefined])('rejects invalid application generation %s', async (generation) => {
    // Given: Server sends invalid application target metadata
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(response({
      accepted: true, message: 'Reload', action: 'reload', target_config_version: 'saved',
      target_generation: generation,
    }, 202))
    // When: Requesting application
    const apply = new HomeSecApiClient().applyConfig({ expected_config_version: 'saved' })
    // Then: Invalid metadata cannot be treated as successful activation
    await expect(apply).rejects.toBeInstanceOf(APIError)
  })
})
