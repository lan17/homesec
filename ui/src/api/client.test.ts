import { afterEach, describe, expect, it, vi } from 'vitest'

import { APIError, HomeSecApiClient } from './client'

describe('HomeSecApiClient.getHealth', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('returns structured health data for healthy responses', async () => {
    // Given: A health endpoint that returns 200 with a valid payload
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          status: 'healthy',
          pipeline: 'running',
          postgres: 'connected',
          cameras_online: 2,
        }),
        {
          status: 200,
          headers: { 'content-type': 'application/json' },
        },
      ),
    )
    const client = new HomeSecApiClient('http://localhost:8081')

    // When: Requesting health
    const result = await client.getHealth()

    // Then: The response should include typed payload and HTTP metadata
    expect(result).toEqual({
      status: 'healthy',
      pipeline: 'running',
      postgres: 'connected',
      cameras_online: 2,
      httpStatus: 200,
    })
  })

  it('returns payload for accepted non-2xx health states', async () => {
    // Given: A health endpoint that returns 503 with a valid degraded payload
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          status: 'unhealthy',
          pipeline: 'stopped',
          postgres: 'connected',
          cameras_online: 0,
        }),
        {
          status: 503,
          headers: { 'content-type': 'application/json' },
        },
      ),
    )
    const client = new HomeSecApiClient('http://localhost:8081')

    // When: Requesting health
    const result = await client.getHealth()

    // Then: The client should still return typed data instead of throwing
    expect(result.httpStatus).toBe(503)
    expect(result.status).toBe('unhealthy')
  })

  it('throws APIError when payload shape is invalid', async () => {
    // Given: A malformed health payload
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          status: 'healthy',
          pipeline: 42,
        }),
        {
          status: 200,
          headers: { 'content-type': 'application/json' },
        },
      ),
    )
    const client = new HomeSecApiClient('http://localhost:8081')

    // When / Then: Client rejects malformed payloads with APIError
    await expect(client.getHealth()).rejects.toBeInstanceOf(APIError)
  })
})
