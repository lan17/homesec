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

describe('HomeSecApiClient.createClipMediaToken', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('posts media-token request and parses token payload', async () => {
    // Given: A media-token endpoint returning tokenized media URL
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          media_url: '/api/v1/clips/clip-42/media?token=abc',
          tokenized: true,
          expires_at: '2026-02-15T20:00:00.000Z',
        }),
        {
          status: 200,
          headers: { 'content-type': 'application/json' },
        },
      ),
    )
    const client = new HomeSecApiClient('http://localhost:8081')

    // When: Requesting media-token for a clip
    const result = await client.createClipMediaToken('clip-42')

    // Then: Client should POST and return parsed token metadata
    expect(fetchSpy).toHaveBeenCalledTimes(1)
    expect(fetchSpy.mock.calls[0]?.[0]).toBe('http://localhost:8081/api/v1/clips/clip-42/media-token')
    expect(fetchSpy.mock.calls[0]?.[1]).toMatchObject({ method: 'POST' })
    expect(result.httpStatus).toBe(200)
    expect(result.media_url).toBe('/api/v1/clips/clip-42/media?token=abc')
    expect(result.tokenized).toBe(true)
    expect(result.expires_at).toBe('2026-02-15T20:00:00.000Z')
  })

  it('throws APIError when media-token payload is malformed', async () => {
    // Given: A malformed media-token payload
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          expires_at: null,
        }),
        {
          status: 200,
          headers: { 'content-type': 'application/json' },
        },
      ),
    )
    const client = new HomeSecApiClient('http://localhost:8081')

    // When / Then: Client rejects malformed response
    await expect(client.createClipMediaToken('clip-42')).rejects.toBeInstanceOf(APIError)
  })
})

describe('HomeSecApiClient camera mutation methods', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('gets a camera by name and parses response payload', async () => {
    // Given: A camera detail endpoint returning a single camera payload
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          name: 'garage',
          enabled: true,
          source_backend: 'local_folder',
          healthy: true,
          last_heartbeat: 1739590400.0,
          source_config: {
            watch_dir: './recordings',
          },
        }),
        {
          status: 200,
          headers: { 'content-type': 'application/json' },
        },
      ),
    )
    const client = new HomeSecApiClient('http://localhost:8081')

    // When: Fetching camera detail by name
    const result = await client.getCamera('garage')

    // Then: Client should call camera detail route and return typed camera fields
    expect(fetchSpy).toHaveBeenCalledTimes(1)
    expect(fetchSpy.mock.calls[0]?.[0]).toBe('http://localhost:8081/api/v1/cameras/garage')
    expect(result.name).toBe('garage')
    expect(result.source_backend).toBe('local_folder')
    expect(result.enabled).toBe(true)
  })

  it('posts create-camera payload with JSON headers and parses response', async () => {
    // Given: A camera create endpoint returning restart-required response payload
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          restart_required: true,
          camera: {
            name: 'front_door',
            enabled: true,
            source_backend: 'rtsp',
            healthy: false,
            last_heartbeat: null,
            source_config: {
              rtsp_url: '***',
            },
          },
          runtime_reload: null,
        }),
        {
          status: 201,
          headers: { 'content-type': 'application/json' },
        },
      ),
    )
    const client = new HomeSecApiClient('http://localhost:8081')

    // When: Creating a camera through the typed API client
    const result = await client.createCamera({
      name: 'front_door',
      enabled: true,
      source_backend: 'rtsp',
      source_config: { rtsp_url: 'rtsp://secret' },
    })

    // Then: Client should issue JSON POST and parse restart metadata
    expect(fetchSpy).toHaveBeenCalledTimes(1)
    expect(fetchSpy.mock.calls[0]?.[0]).toBe('http://localhost:8081/api/v1/cameras')
    expect(fetchSpy.mock.calls[0]?.[1]).toMatchObject({
      method: 'POST',
      headers: {
        Accept: 'application/json',
        'content-type': 'application/json',
      },
    })
    expect(fetchSpy.mock.calls[0]?.[1]?.body).toBe(
      JSON.stringify({
        name: 'front_door',
        enabled: true,
        source_backend: 'rtsp',
        source_config: { rtsp_url: 'rtsp://secret' },
      }),
    )
    expect(result.restart_required).toBe(true)
    expect(result.camera?.name).toBe('front_door')
    expect(result.httpStatus).toBe(201)
  })

  it('passes apply_changes query parameter for create camera when requested', async () => {
    // Given: A camera create endpoint with runtime-reload acceptance in payload
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          restart_required: false,
          camera: {
            name: 'driveway',
            enabled: true,
            source_backend: 'rtsp',
            healthy: false,
            last_heartbeat: null,
            source_config: {},
          },
          runtime_reload: {
            accepted: true,
            message: 'Runtime reload accepted',
            target_generation: 12,
          },
        }),
        {
          status: 201,
          headers: { 'content-type': 'application/json' },
        },
      ),
    )
    const client = new HomeSecApiClient('http://localhost:8081')

    // When: Creating a camera and requesting immediate apply
    const result = await client.createCamera(
      {
        name: 'driveway',
        enabled: true,
        source_backend: 'rtsp',
        source_config: { rtsp_url: 'rtsp://example/stream' },
      },
      { applyChanges: true },
    )

    // Then: Request URL contains apply_changes and runtime reload payload is parsed
    expect(fetchSpy).toHaveBeenCalledTimes(1)
    expect(fetchSpy.mock.calls[0]?.[0]).toBe('http://localhost:8081/api/v1/cameras?apply_changes=true')
    expect(result.restart_required).toBe(false)
    expect(result.runtime_reload?.accepted).toBe(true)
    expect(result.runtime_reload?.target_generation).toBe(12)
  })

  it('patches camera enabled toggle payload and parses config-change response', async () => {
    // Given: A camera update endpoint with restart-required response
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          restart_required: true,
          camera: {
            name: 'garage',
            enabled: false,
            source_backend: 'rtsp',
            healthy: false,
            last_heartbeat: null,
            source_config: {},
          },
          runtime_reload: null,
        }),
        {
          status: 200,
          headers: { 'content-type': 'application/json' },
        },
      ),
    )
    const client = new HomeSecApiClient('http://localhost:8081')

    // When: Updating camera enabled state
    const result = await client.updateCamera('garage', { enabled: false })

    // Then: Client should issue PATCH request and return parsed camera payload
    expect(fetchSpy).toHaveBeenCalledTimes(1)
    expect(fetchSpy.mock.calls[0]?.[0]).toBe('http://localhost:8081/api/v1/cameras/garage')
    expect(fetchSpy.mock.calls[0]?.[1]).toMatchObject({ method: 'PATCH' })
    expect(result.restart_required).toBe(true)
    expect(result.camera?.enabled).toBe(false)
  })

  it('passes apply_changes query parameter for update camera when requested', async () => {
    // Given: A camera update endpoint that accepts apply changes
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          restart_required: false,
          camera: {
            name: 'garage',
            enabled: true,
            source_backend: 'rtsp',
            healthy: false,
            last_heartbeat: null,
            source_config: {},
          },
          runtime_reload: {
            accepted: true,
            message: 'Runtime reload accepted',
            target_generation: 6,
          },
        }),
        {
          status: 200,
          headers: { 'content-type': 'application/json' },
        },
      ),
    )
    const client = new HomeSecApiClient('http://localhost:8081')

    // When: Updating a camera with applyChanges enabled
    const result = await client.updateCamera(
      'garage',
      { enabled: true },
      { applyChanges: true },
    )

    // Then: Request URL contains apply_changes and parsed payload includes runtime reload metadata
    expect(fetchSpy).toHaveBeenCalledTimes(1)
    expect(fetchSpy.mock.calls[0]?.[0]).toBe(
      'http://localhost:8081/api/v1/cameras/garage?apply_changes=true',
    )
    expect(result.restart_required).toBe(false)
    expect(result.runtime_reload?.target_generation).toBe(6)
  })

  it('deletes camera and handles null camera payloads', async () => {
    // Given: A camera delete endpoint returning null camera in response body
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          restart_required: true,
          camera: null,
          runtime_reload: null,
        }),
        {
          status: 200,
          headers: { 'content-type': 'application/json' },
        },
      ),
    )
    const client = new HomeSecApiClient('http://localhost:8081')

    // When: Deleting a camera by name
    const result = await client.deleteCamera('garage')

    // Then: Client should issue DELETE and preserve null camera semantics
    expect(fetchSpy).toHaveBeenCalledTimes(1)
    expect(fetchSpy.mock.calls[0]?.[1]).toMatchObject({ method: 'DELETE' })
    expect(result.restart_required).toBe(true)
    expect(result.camera).toBeNull()
  })

  it('passes apply_changes query parameter for delete camera when requested', async () => {
    // Given: A camera delete endpoint with reload acceptance payload
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          restart_required: false,
          camera: null,
          runtime_reload: {
            accepted: true,
            message: 'Runtime reload accepted',
            target_generation: 7,
          },
        }),
        {
          status: 200,
          headers: { 'content-type': 'application/json' },
        },
      ),
    )
    const client = new HomeSecApiClient('http://localhost:8081')

    // When: Deleting a camera with applyChanges enabled
    const result = await client.deleteCamera('garage', { applyChanges: true })

    // Then: DELETE request includes apply_changes and parsed payload includes runtime reload metadata
    expect(fetchSpy).toHaveBeenCalledTimes(1)
    expect(fetchSpy.mock.calls[0]?.[0]).toBe(
      'http://localhost:8081/api/v1/cameras/garage?apply_changes=true',
    )
    expect(result.restart_required).toBe(false)
    expect(result.runtime_reload?.accepted).toBe(true)
  })
})

describe('HomeSecApiClient runtime methods', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('requests runtime reload and parses accepted response', async () => {
    // Given: Runtime reload endpoint returning async acceptance metadata
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          accepted: true,
          message: 'Runtime reload accepted',
          target_generation: 4,
        }),
        {
          status: 202,
          headers: { 'content-type': 'application/json' },
        },
      ),
    )
    const client = new HomeSecApiClient('http://localhost:8081')

    // When: Triggering runtime reload
    const result = await client.reloadRuntime()

    // Then: Client should POST and return typed acceptance payload
    expect(fetchSpy).toHaveBeenCalledTimes(1)
    expect(fetchSpy.mock.calls[0]?.[0]).toBe('http://localhost:8081/api/v1/runtime/reload')
    expect(fetchSpy.mock.calls[0]?.[1]).toMatchObject({ method: 'POST' })
    expect(result.httpStatus).toBe(202)
    expect(result.accepted).toBe(true)
    expect(result.target_generation).toBe(4)
  })

  it('parses runtime status payload for control-plane rendering', async () => {
    // Given: Runtime status endpoint with a reloading state payload
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          state: 'reloading',
          generation: 4,
          reload_in_progress: true,
          active_config_version: 'abc123',
          last_reload_at: '2026-02-16T01:00:00.000Z',
          last_reload_error: null,
        }),
        {
          status: 200,
          headers: { 'content-type': 'application/json' },
        },
      ),
    )
    const client = new HomeSecApiClient('http://localhost:8081')

    // When: Fetching runtime status snapshot
    const result = await client.getRuntimeStatus()

    // Then: Runtime fields should be parsed into a typed snapshot
    expect(result.httpStatus).toBe(200)
    expect(result.state).toBe('reloading')
    expect(result.reload_in_progress).toBe(true)
    expect(result.active_config_version).toBe('abc123')
  })

  it('rejects malformed runtime status payloads', async () => {
    // Given: Runtime status payload with unsupported state value
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          state: 'booting',
          generation: 4,
          reload_in_progress: false,
          active_config_version: null,
          last_reload_at: null,
          last_reload_error: null,
        }),
        {
          status: 200,
          headers: { 'content-type': 'application/json' },
        },
      ),
    )
    const client = new HomeSecApiClient('http://localhost:8081')

    // When / Then: Client should fail fast on contract mismatch
    await expect(client.getRuntimeStatus()).rejects.toBeInstanceOf(APIError)
  })
})
