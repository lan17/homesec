import { apiClient, isAPIError, type ConfigApplySnapshot, type ConfigSnapshot } from '../../api/client'

export const CONFIG_APPLY_TIMEOUT_MS = 45_000
const APPLY_POLL_INTERVAL_MS = 1_000

function abortError(): Error {
  return new DOMException('Apply monitoring cancelled', 'AbortError')
}

function sleep(signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(abortError())
      return
    }
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', abort)
      resolve()
    }, APPLY_POLL_INTERVAL_MS)
    function abort(): void {
      clearTimeout(timer)
      reject(abortError())
    }
    signal.addEventListener('abort', abort, { once: true })
  })
}

export async function waitForConfigApply(
  response: ConfigApplySnapshot,
  signal: AbortSignal,
): Promise<ConfigSnapshot> {
  const deadline = Date.now() + CONFIG_APPLY_TIMEOUT_MS
  while (Date.now() < deadline) {
    if (signal.aborted) {
      throw abortError()
    }
    let ready = response.action !== 'reload'
    try {
      if (!ready) {
        const runtime = await apiClient.getRuntimeStatus({ signal })
        if (!runtime.reload_in_progress && runtime.last_reload_error
            && (runtime.state === 'failed'
              || runtime.active_config_version !== response.target_config_version)) {
          throw new Error(`Runtime reload failed: ${runtime.last_reload_error}`)
        }
        ready = runtime.state === 'idle'
          && !runtime.reload_in_progress
          && runtime.active_config_version === response.target_config_version
          && response.target_generation !== null
          && runtime.generation >= response.target_generation
      }
      if (ready) {
        const config = await apiClient.getConfig({ signal })
        if (config.saved_config_version === response.target_config_version
            && config.active_config_version === response.target_config_version
            && config.apply_required === 'none') {
          return config
        }
      }
    } catch (error) {
      if (signal.aborted || (error instanceof Error && error.name === 'AbortError')) {
        throw abortError()
      }
      if (error instanceof Error && error.message.startsWith('Runtime reload failed:')) {
        throw error
      }
      if (isAPIError(error) && (error.status === 401 || error.status === 403)) {
        throw error
      }
      // A server restart temporarily interrupts authenticated requests. Retry until bounded timeout.
    }
    await sleep(signal)
  }
  throw new Error('Settings are saved, but activation could not be confirmed. Refresh status before retrying Apply.')
}
