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

function isPrivateIPv4Address(hostname: string): boolean {
  const octets = hostname.split('.').map((part) => Number(part))
  if (
    octets.length !== 4 ||
    octets.some((octet) => !Number.isInteger(octet) || octet < 0 || octet > 255)
  ) {
    return false
  }

  const [first = 0, second = 0] = octets
  return (
    first === 10 ||
    first === 127 ||
    (first === 100 && second >= 64 && second <= 127) ||
    (first === 169 && second === 254) ||
    (first === 172 && second >= 16 && second <= 31) ||
    (first === 192 && second === 168)
  )
}

function isPrivateIPv6Address(hostname: string): boolean {
  const normalized = hostname.replace(/^\[|\]$/g, '').toLowerCase()
  if (normalized === '::1') {
    return true
  }

  const [firstHextetRaw] = normalized.split(':')
  const firstHextet = Number.parseInt(firstHextetRaw ?? '', 16)
  if (Number.isNaN(firstHextet)) {
    return false
  }

  return (firstHextet & 0xfe00) === 0xfc00 || (firstHextet & 0xffc0) === 0xfe80
}

function isLocalPlainHttpHost(hostname: string): boolean {
  const normalized = hostname.toLowerCase()
  if (
    normalized === 'localhost' ||
    normalized.endsWith('.localhost') ||
    normalized.endsWith('.local')
  ) {
    return true
  }

  if (normalized.includes(':')) {
    return isPrivateIPv6Address(normalized)
  }

  if (isPrivateIPv4Address(normalized)) {
    return true
  }

  return !normalized.includes('.')
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

  if (parsed.protocol === 'http:' && !isLocalPlainHttpHost(parsed.hostname)) {
    return {
      ok: false,
      message: 'Plain HTTP is only supported for local network HomeSec servers. Use HTTPS for public hosts.',
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
