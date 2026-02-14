import { afterEach, describe, expect, it, vi } from 'vitest'

import { APIError, HomeSecApiClient } from './client'

describe('HomeSecApiClient.getCameras', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('returns structured camera list data for successful responses', async () => {
    // Given: A cameras endpoint with one camera payload
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify([
          {
            name: 'front_door',
            enabled: true,
            source_backend: 'rtsp',
            healthy: true,
            last_heartbeat: 1739590400.0,
            source_config: {
              stream_url: 'rtsp://example/stream',
            },
          },
        ]),
        {
          status: 200,
          headers: { 'content-type': 'application/json' },
        },
      ),
    )
    const client = new HomeSecApiClient('http://localhost:8081')

    // When: Requesting cameras
    const result = await client.getCameras()

    // Then: Response should include typed payload and HTTP metadata
    expect(result.length).toBe(1)
    expect(result[0]?.name).toBe('front_door')
    expect(result[0]?.source_backend).toBe('rtsp')
  })
})

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

describe('HomeSecApiClient.getClips', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('serializes filters to query string and parses clip list payload', async () => {
    // Given: A clips endpoint returning a paginated list payload
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          clips: [
            {
              id: 'clip-1',
              camera: 'front_door',
              status: 'done',
              created_at: '2026-02-14T12:00:00.000Z',
              activity_type: 'package',
              risk_level: 'low',
              summary: 'Package dropped',
              detected_objects: ['person'],
              storage_uri: 'dropbox:/clips/clip-1.mp4',
              view_url: null,
              alerted: true,
            },
          ],
          limit: 25,
          next_cursor: 'cursor-2',
          has_more: true,
        }),
        {
          status: 200,
          headers: { 'content-type': 'application/json' },
        },
      ),
    )
    const client = new HomeSecApiClient('http://localhost:8081')

    // When: Requesting clips with filters
    const result = await client.getClips({
      camera: 'front_door',
      status: 'done',
      detected: true,
      activity_type: 'package',
      limit: 25,
      cursor: 'cursor-1',
    })

    // Then: Query should be serialized and payload parsed
    expect(fetchSpy).toHaveBeenCalledTimes(1)
    expect(fetchSpy.mock.calls[0]?.[0]).toBe(
      'http://localhost:8081/api/v1/clips?camera=front_door&status=done&detected=true&activity_type=package&limit=25&cursor=cursor-1',
    )
    expect(result.httpStatus).toBe(200)
    expect(result.has_more).toBe(true)
    expect(result.next_cursor).toBe('cursor-2')
    expect(result.clips[0]?.id).toBe('clip-1')
  })

  it('throws APIError when clip list payload is malformed', async () => {
    // Given: A malformed list payload
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          clips: {},
        }),
        {
          status: 200,
          headers: { 'content-type': 'application/json' },
        },
      ),
    )
    const client = new HomeSecApiClient('http://localhost:8081')

    // When / Then: The malformed payload should raise APIError
    await expect(client.getClips()).rejects.toBeInstanceOf(APIError)
  })
})

describe('HomeSecApiClient.getClip', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('fetches a clip by id and parses response payload', async () => {
    // Given: A clip endpoint returning a single clip payload
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          id: 'clip-42',
          camera: 'garage',
          status: 'uploaded',
          created_at: '2026-02-14T12:10:00.000Z',
          activity_type: null,
          risk_level: null,
          summary: null,
          detected_objects: [],
          storage_uri: 'dropbox:/clips/clip-42.mp4',
          view_url: 'https://example.com/view/clip-42',
          alerted: false,
        }),
        {
          status: 200,
          headers: { 'content-type': 'application/json' },
        },
      ),
    )
    const client = new HomeSecApiClient('http://localhost:8081')

    // When: Requesting clip detail by ID
    const result = await client.getClip('clip-42')

    // Then: Parsed clip should include HTTP metadata
    expect(result.httpStatus).toBe(200)
    expect(result.id).toBe('clip-42')
    expect(result.view_url).toBe('https://example.com/view/clip-42')
  })
})
