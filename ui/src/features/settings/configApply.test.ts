import { afterEach, describe, expect, it, vi } from 'vitest'

import { apiClient, type ConfigApplySnapshot, type ConfigSnapshot } from '../../api/client'
import { waitForConfigApply } from './configApply'

const saved: ConfigSnapshot = { config: {}, saved_config_version: 'target', active_config_version: 'target',
  apply_required: 'none', httpStatus: 200 }
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
})
