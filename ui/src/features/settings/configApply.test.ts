import { afterEach, describe, expect, it, vi } from 'vitest'

import { apiClient, type ConfigApplySnapshot, type ConfigSnapshot } from '../../api/client'
import { waitForConfigApply } from './configApply'

const saved: ConfigSnapshot = { config: {}, saved_config_version: 'target', active_config_version: 'target',
  apply_required: 'none', credentials: {}, credentials_editable: true, httpStatus: 200 }
const response: ConfigApplySnapshot = { accepted: true, action: 'restart', message: 'Accepted',
  target_config_version: 'target', target_generation: null, httpStatus: 202 }

describe('configuration activation monitoring', () => {
  afterEach(() => { vi.restoreAllMocks(); vi.useRealTimers() })

  it('waits for a restart to activate the saved version despite temporary disconnects', async () => {
    // Given: API is initially reachable with the old active version, then restarts
    vi.useFakeTimers()
    const get = vi.spyOn(apiClient, 'getConfig')
      .mockResolvedValueOnce({ ...saved, active_config_version: 'old', apply_required: 'restart' })
      .mockRejectedValueOnce(new TypeError('fetch failed'))
      .mockResolvedValue(saved)
    // When: Monitoring the accepted restart
    const result = waitForConfigApply(response, new AbortController().signal)
    await vi.advanceTimersByTimeAsync(2_000)
    // Then: Only observed active target is reported as successful
    await expect(result).resolves.toEqual(saved)
    expect(get).toHaveBeenCalledTimes(3)
  })

  it('requires a completed reload generation and a successful configuration status', async () => {
    // Given: Target hashes match but the runtime still requires recovery
    vi.useFakeTimers()
    vi.spyOn(apiClient, 'getRuntimeStatus').mockResolvedValue({
      state: 'idle', generation: 4, reload_in_progress: false, active_config_version: 'target',
      last_reload_at: null, last_reload_error: null, httpStatus: 200,
    })
    const get = vi.spyOn(apiClient, 'getConfig')
      .mockResolvedValueOnce({ ...saved, apply_required: 'reload' }).mockResolvedValue(saved)
    // When: Monitoring a reload for generation four
    const result = waitForConfigApply({ ...response, action: 'reload', target_generation: 4 }, new AbortController().signal)
    await vi.advanceTimersByTimeAsync(1_000)
    // Then: Matching hashes alone are insufficient
    await expect(result).resolves.toEqual(saved)
    expect(get).toHaveBeenCalledTimes(2)
  })

  it('reports reload failure while settings remain saved', async () => {
    // Given: Worker reports failure at the requested generation
    vi.spyOn(apiClient, 'getRuntimeStatus').mockResolvedValue({
      state: 'failed', generation: 4, reload_in_progress: false, active_config_version: 'target',
      last_reload_at: null, last_reload_error: 'Plugin initialization failed', httpStatus: 200,
    })
    // When: Monitoring that reload
    const result = waitForConfigApply({ ...response, action: 'reload', target_generation: 4 }, new AbortController().signal)
    // Then: Failure is observable instead of a false activation success
    await expect(result).rejects.toThrow('Runtime reload failed')
  })

  it('reports failed candidate rollback while the previous runtime remains idle', async () => {
    // Given: A candidate reload failed and restored the previous runtime generation
    vi.spyOn(apiClient, 'getRuntimeStatus').mockResolvedValue({
      state: 'idle', generation: 3, reload_in_progress: false, active_config_version: 'old',
      last_reload_at: null, last_reload_error: 'Candidate initialization failed', httpStatus: 200,
    })
    // When: Monitoring the requested candidate generation four
    const result = waitForConfigApply({ ...response, action: 'reload', target_generation: 4 }, new AbortController().signal)
    // Then: Rollback is an immediate activation failure even though recording continues
    await expect(result).rejects.toThrow('Runtime reload failed: Candidate initialization failed')
  })

  it('bounds monitoring when activation never occurs', async () => {
    // Given: Saved target remains pending
    vi.useFakeTimers()
    vi.spyOn(apiClient, 'getConfig').mockResolvedValue({ ...saved, active_config_version: 'old', apply_required: 'restart' })
    // When: Waiting through the bounded polling interval
    const result = waitForConfigApply(response, new AbortController().signal)
    const assertion = expect(result).rejects.toThrow('activation could not be confirmed')
    await vi.advanceTimersByTimeAsync(45_000)
    // Then: User receives an explicit saved-but-unconfirmed outcome
    await assertion
  })

  it.each(['reload', 'restart', 'none'] as const)('confirms healthy %s target A while the latest saved B still needs Apply', async (action) => {
    // Given: Another writer saves B before accepted target A finishes activation
    const latest = { ...saved, saved_config_version: 'later', active_config_version: 'target', apply_required: 'reload' }
    const fetch = vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) => new Response(JSON.stringify(
      String(url).endsWith('/runtime/status') ? {
        state: 'idle', generation: 4, reload_in_progress: false, active_config_version: 'target',
        last_reload_at: null, last_reload_error: null,
      } : latest,
    ), { headers: { 'content-type': 'application/json' } }))

    // When: Monitoring A using the actual API client and fake HTTP boundary
    const result = await waitForConfigApply({ ...response, action,
      target_generation: action === 'restart' ? null : 4 }, new AbortController().signal)

    // Then: A is confirmed once while the returned saved B remains pending instead of timing out
    expect(result.saved_config_version).toBe('later')
    expect(result.active_config_version).toBe('target')
    expect(result.apply_required).toBe('reload')
    expect(fetch.mock.calls.filter(([url]) => String(url).endsWith('/runtime/status'))).toHaveLength(1)
    expect(fetch.mock.calls.filter(([url]) => String(url).endsWith('/config'))).toHaveLength(1)
  })

  it.each(['reload', 'restart', 'none'] as const)('rejects failed %s target A even though saved B differs and active hashes match', async (action) => {
    // Given: A matching active version belongs to a failed runtime and a newer saved revision
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) => new Response(JSON.stringify(
      String(url).endsWith('/runtime/status') ? {
        state: 'failed', generation: 4, reload_in_progress: false, active_config_version: 'target',
        last_reload_at: null, last_reload_error: 'Worker initialization failed',
      } : { ...saved, saved_config_version: 'later', apply_required: 'reload' },
    ), { headers: { 'content-type': 'application/json' } }))

    // When: Monitoring a matching target with a pending later save
    const result = waitForConfigApply({ ...response, action,
      target_generation: action === 'restart' ? null : 4 }, new AbortController().signal)

    // Then: A later saved hash cannot bypass the runtime-health requirement
    await expect(result).rejects.toThrow('Runtime reload failed: Worker initialization failed')
  })

  it('requires the requested reload generation even with a later pending saved revision', async () => {
    // Given: Runtime has the matching target hash but has not reached the accepted generation
    vi.useFakeTimers()
    const runtime = vi.spyOn(apiClient, 'getRuntimeStatus')
      .mockResolvedValueOnce({ state: 'idle', generation: 3, reload_in_progress: false,
        active_config_version: 'target', last_reload_at: null, last_reload_error: null, httpStatus: 200 })
      .mockResolvedValue({ state: 'idle', generation: 4, reload_in_progress: false,
        active_config_version: 'target', last_reload_at: null, last_reload_error: null, httpStatus: 200 })
    const get = vi.spyOn(apiClient, 'getConfig').mockResolvedValue({ ...saved,
      saved_config_version: 'later', apply_required: 'reload' })

    // When: The runtime reaches the requested generation on the second poll
    const result = waitForConfigApply({ ...response, action: 'reload', target_generation: 4 }, new AbortController().signal)
    await vi.advanceTimersByTimeAsync(1_000)

    // Then: Only the matching healthy generation confirms A and returns B as pending
    await expect(result).resolves.toMatchObject({ saved_config_version: 'later', apply_required: 'reload' })
    expect(runtime).toHaveBeenCalledTimes(2)
    expect(get).toHaveBeenCalledTimes(1)
  })

  it.each([401, 403])('propagates authorization failure %s without waiting through the activation timeout', async (status) => {
    // Given: Restart monitoring cannot read authenticated configuration status
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({ detail: 'Unauthorized' }),
      { status, headers: { 'content-type': 'application/json' } }))

    // When: The actual API client receives an auth refusal during polling
    const result = waitForConfigApply(response, new AbortController().signal)

    // Then: Monitoring rejects promptly rather than treating auth failure as a restart disconnect
    await expect(result).rejects.toMatchObject({ status })
    expect(fetch).toHaveBeenCalledTimes(1)
  })

  it('drops a matching success returned by a transport after monitoring is cancelled', async () => {
    // Given: A restart confirmation request ignores cancellation at the HTTP boundary
    let finish: ((value: Response) => void) | undefined
    vi.spyOn(globalThis, 'fetch').mockImplementation(async () => new Promise<Response>((resolve) => { finish = resolve }))
    const controller = new AbortController()
    const result = waitForConfigApply(response, controller.signal)
    const assertion = expect(result).rejects.toMatchObject({ name: 'AbortError' })

    // When: The monitor is cancelled before the old matching config response arrives
    controller.abort()
    finish?.(new Response(JSON.stringify(saved), { headers: { 'content-type': 'application/json' } }))

    // Then: A completed transport cannot convert cancellation into confirmed activation
    await assertion
  })


  it('reports finished candidate failure after rollback to the same hash at an older generation', async () => {
    // Given: A same-config reload failed and restored its matching hash at the earlier generation
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({
      state: 'idle', generation: 3, reload_in_progress: false, active_config_version: 'target',
      last_reload_at: null, last_reload_error: 'Same-config worker initialization failed',
    }), { headers: { 'content-type': 'application/json' } }))

    // When: Monitoring accepted generation four after the reload task has finished
    const result = waitForConfigApply({ ...response, action: 'reload', target_generation: 4 }, new AbortController().signal)

    // Then: The matching hash cannot hide a completed generation failure behind a polling timeout
    await expect(result).rejects.toThrow('Runtime reload failed: Same-config worker initialization failed')
    expect(fetch).toHaveBeenCalledTimes(1)
  })

})
