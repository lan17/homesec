import type { DeviceInfoResponse, DiscoveredCameraResponse } from '../../api/generated/types'

function slugPart(value: string): string {
  const normalized = value.trim().toLowerCase().replace(/[^a-z0-9]+/g, '-')
  return normalized.replace(/^-+|-+$/g, '')
}

function normalizeIpForSlug(ip: string): string {
  return slugPart(ip.replaceAll(':', '-'))
}

export function summarizeOnvifScopes(scopes: string[], maxItems = 2): string {
  if (scopes.length === 0) {
    return 'No scopes reported'
  }
  const preview = scopes.slice(0, maxItems).join(', ')
  if (scopes.length <= maxItems) {
    return preview
  }
  return `${preview}, +${scopes.length - maxItems} more`
}

export function deriveOnvifCameraName({
  discoveredCamera,
  deviceInfo,
}: {
  discoveredCamera: DiscoveredCameraResponse
  deviceInfo: DeviceInfoResponse
}): string {
  const parts = [
    slugPart(deviceInfo.manufacturer),
    slugPart(deviceInfo.model),
    normalizeIpForSlug(discoveredCamera.ip),
  ].filter((part) => part.length > 0)

  if (parts.length > 0) {
    return parts.join('-')
  }
  return `camera-${normalizeIpForSlug(discoveredCamera.ip)}`
}

export function injectCredentialsIntoRtspUri({
  streamUri,
  username,
  password,
}: {
  streamUri: string
  username: string
  password: string
}): string {
  const trimmedUsername = username.trim()
  const trimmedPassword = password.trim()
  if (trimmedUsername.length === 0 || trimmedPassword.length === 0) {
    return streamUri
  }

  try {
    const parsed = new URL(streamUri)
    if (parsed.protocol !== 'rtsp:') {
      return streamUri
    }

    if (!parsed.username) {
      parsed.username = trimmedUsername
    }
    if (!parsed.password) {
      parsed.password = trimmedPassword
    }
    return parsed.toString()
  } catch {
    return streamUri
  }
}

export function deriveOnvifProbePortFromXaddr(xaddr: string): number {
  const fallbackPort = 80

  const parsed = parseXaddrUrl(xaddr)
  if (parsed === null) {
    return fallbackPort
  }

  if (parsed.port) {
    const parsedPort = Number.parseInt(parsed.port, 10)
    if (!Number.isNaN(parsedPort) && parsedPort >= 1 && parsedPort <= 65535) {
      return parsedPort
    }
  }

  if (parsed.protocol === 'https:') {
    return 443
  }
  if (parsed.protocol === 'http:') {
    return 80
  }
  return fallbackPort
}

function parseXaddrUrl(xaddr: string): URL | null {
  try {
    return new URL(xaddr)
  } catch {
    if (xaddr.includes('://')) {
      return null
    }
  }

  try {
    return new URL(`http://${xaddr}`)
  } catch {
    return null
  }
}
