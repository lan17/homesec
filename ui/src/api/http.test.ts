import { afterEach, describe, expect, it, vi } from 'vitest'

import { JsonHttpClient } from './http'

function installWindowSessionStorageMock(): void {
  const store = new Map<string, string>()
  vi.stubGlobal('window', {
    sessionStorage: {
      getItem: (key: string): string | null => store.get(key) ?? null,
      setItem: (key: string, value: string): void => {
        store.set(key, value)
      },
      removeItem: (key: string): void => {
        store.delete(key)
      },
      clear: (): void => {
        store.clear()
      },
    },
  })
}

describe('JsonHttpClient.requestJson', () => {
  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it('serializes query params, auth header, and JSON body', async () => {
    // Given: A fetch mock and an authenticated POST request with query/body
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      }),
    )
    const client = new JsonHttpClient('http://localhost:8081')

    // When: Sending a request with query params and JSON payload
    const response = await client.requestJson('/api/v1/cameras', {
      method: 'POST',
      query: { camera: 'front', enabled: true },
      apiKey: 'secret',
      body: { name: 'front' },
    })

    // Then: URL, headers, and payload shape are preserved
    expect(fetchSpy).toHaveBeenCalledTimes(1)
    expect(fetchSpy.mock.calls[0]?.[0]).toBe(
      'http://localhost:8081/api/v1/cameras?camera=front&enabled=true',
    )
    expect(fetchSpy.mock.calls[0]?.[1]).toMatchObject({
      method: 'POST',
      headers: {
        Accept: 'application/json',
        Authorization: 'Bearer secret',
        'content-type': 'application/json',
      },
      body: JSON.stringify({ name: 'front' }),
    })
    expect(response.status).toBe(200)
    expect(response.payload).toEqual({ ok: true })
  })

  it('loads API key from session storage when apiKey option is omitted', async () => {
    // Given: A stored API key and a request without explicit apiKey option
    installWindowSessionStorageMock()
    window.sessionStorage.setItem('homesec.apiKey', 'stored-secret')
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      }),
    )
    const client = new JsonHttpClient('http://localhost:8081')

    // When: Sending the request
    await client.requestJson('/api/v1/health', {})

    // Then: Authorization header is derived from storage
    expect(fetchSpy.mock.calls[0]?.[1]).toMatchObject({
      headers: {
        Accept: 'application/json',
        Authorization: 'Bearer stored-secret',
      },
    })
  })

  it('throws APIError with canonical metadata for non-allowed non-2xx responses', async () => {
    // Given: A failing endpoint with canonical error envelope
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ detail: 'Unauthorized', error_code: 'UNAUTHORIZED' }), {
        status: 401,
        headers: { 'content-type': 'application/json' },
      }),
    )
    const client = new JsonHttpClient('http://localhost:8081')

    // When / Then: Request rejects with typed APIError
    await expect(client.requestJson('/api/v1/stats', {})).rejects.toMatchObject({
      name: 'APIError',
      status: 401,
      errorCode: 'UNAUTHORIZED',
      message: 'Unauthorized',
    })
  })

  it('returns payload for explicitly allowed non-2xx statuses', async () => {
    // Given: A health endpoint returning a degraded 503 response
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ status: 'unhealthy' }), {
        status: 503,
        headers: { 'content-type': 'application/json' },
      }),
    )
    const client = new JsonHttpClient('http://localhost:8081')

    // When: Allowing 503 in request options
    const response = await client.requestJson('/api/v1/health', { allowStatuses: [503] })

    // Then: Client returns payload instead of throwing
    expect(response.status).toBe(503)
    expect(response.payload).toEqual({ status: 'unhealthy' })
  })
})
