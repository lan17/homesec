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

describe('HomeSecApiClient.getStats', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('returns structured stats data for successful responses', async () => {
    // Given: A stats endpoint that returns 200 with a valid payload
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          clips_today: 14,
          alerts_today: 3,
          cameras_total: 4,
          cameras_online: 3,
          uptime_seconds: 123.45,
        }),
        {
          status: 200,
          headers: { 'content-type': 'application/json' },
        },
      ),
    )
    const client = new HomeSecApiClient('http://localhost:8081')

    // When: Requesting stats
    const result = await client.getStats()

    // Then: The response should include typed payload and HTTP metadata
    expect(result).toEqual({
      clips_today: 14,
      alerts_today: 3,
      cameras_total: 4,
      cameras_online: 3,
      uptime_seconds: 123.45,
      httpStatus: 200,
    })
  })

  it('maps canonical unauthorized envelope to APIError metadata', async () => {
    // Given: A stats endpoint that returns canonical 401 envelope
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          detail: 'Unauthorized',
          error_code: 'UNAUTHORIZED',
        }),
        {
          status: 401,
          headers: { 'content-type': 'application/json' },
        },
      ),
    )
    const client = new HomeSecApiClient('http://localhost:8081')

    // When / Then: The thrown APIError should include status and error_code
    await expect(client.getStats()).rejects.toMatchObject({
      name: 'APIError',
      status: 401,
      errorCode: 'UNAUTHORIZED',
      message: 'Unauthorized',
    })
  })
})

describe('HomeSecApiClient.getDiagnostics', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('parses nested diagnostics payloads', async () => {
    // Given: A diagnostics endpoint that returns nested health components
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          status: 'healthy',
          uptime_seconds: 900.5,
          postgres: {
            status: 'ok',
            error: null,
            latency_ms: 12.3,
          },
          storage: {
            status: 'ok',
            error: null,
            latency_ms: 21.1,
          },
          cameras: {
            front_door: {
              healthy: true,
              enabled: true,
              last_heartbeat: 1739590400.0,
            },
          },
        }),
        {
          status: 200,
          headers: { 'content-type': 'application/json' },
        },
      ),
    )
    const client = new HomeSecApiClient('http://localhost:8081')

    // When: Requesting diagnostics
    const result = await client.getDiagnostics()

    // Then: Nested component and camera fields should be preserved
    expect(result.httpStatus).toBe(200)
    expect(result.postgres.status).toBe('ok')
    expect(result.storage.latency_ms).toBe(21.1)
    expect(result.cameras.front_door.healthy).toBe(true)
  })
})
