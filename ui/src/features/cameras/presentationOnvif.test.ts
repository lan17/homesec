import { describe, expect, it } from 'vitest'

import { deriveOnvifProbePortFromXaddr } from './presentationOnvif'

describe('deriveOnvifProbePortFromXaddr', () => {
  it('returns explicit port from fully-qualified xaddr', () => {
    // Given: Discovery xaddr contains an explicit non-default ONVIF port
    const xaddr = 'http://192.168.1.50:8899/onvif/device_service'

    // When: Deriving probe port from discovery metadata
    const port = deriveOnvifProbePortFromXaddr(xaddr)

    // Then: Derived port should preserve the discovered explicit port
    expect(port).toBe(8899)
  })

  it('returns explicit port from xaddr even when scheme is omitted', () => {
    // Given: Discovery xaddr omits scheme but still provides host:port
    const xaddr = '192.168.1.50:5000/onvif/device_service'

    // When: Deriving probe port from discovery metadata
    const port = deriveOnvifProbePortFromXaddr(xaddr)

    // Then: Derived port should still use the discovered explicit port
    expect(port).toBe(5000)
  })

  it('falls back to protocol default when xaddr has no explicit port', () => {
    // Given: Discovery xaddr has https scheme and no explicit port
    const xaddr = 'https://192.168.1.50/onvif/device_service'

    // When: Deriving probe port from discovery metadata
    const port = deriveOnvifProbePortFromXaddr(xaddr)

    // Then: Derived port should use HTTPS default
    expect(port).toBe(443)
  })
})
