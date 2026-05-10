import { normalizeServerBaseUrl } from '../../api/serverBaseUrlProvider'

export interface NativeSetupServerUrl {
  serverBaseUrl: string
  isPlainHttp: boolean
}

export type NativeSetupServerUrlValidation =
  | {
      ok: true
      value: NativeSetupServerUrl
    }
  | {
      ok: false
      message: string
    }

export function validateNativeSetupServerUrl(input: string): NativeSetupServerUrlValidation {
  const normalized = normalizeServerBaseUrl(input)
  if (!normalized) {
    return {
      ok: false,
      message: 'Enter the HomeSec server URL.',
    }
  }

  let parsed: URL
  try {
    parsed = new URL(normalized)
  } catch {
    return {
      ok: false,
      message: 'Enter a valid URL that starts with http:// or https://.',
    }
  }

  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    return {
      ok: false,
      message: 'Only http:// and https:// server URLs are supported.',
    }
  }

  if (!parsed.hostname) {
    return {
      ok: false,
      message: 'Enter a server URL with a host name or IP address.',
    }
  }

  parsed.hash = ''
  parsed.search = ''
  parsed.pathname = parsed.pathname.replace(/\/+$/, '')

  return {
    ok: true,
    value: {
      serverBaseUrl: parsed.toString().replace(/\/+$/, ''),
      isPlainHttp: parsed.protocol === 'http:',
    },
  }
}
