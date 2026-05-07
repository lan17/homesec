import { describe, expect, it } from 'vitest'

import { expectedPcmFrameBytes, frameSampleCount, PcmFrameEncoder } from './pcm'

describe('PcmFrameEncoder', () => {
  it('emits exact PCM frame sizes without resampling', () => {
    // Given: A PCM encoder whose browser and target sample rates already match
    const encoder = new PcmFrameEncoder(16_000, 16_000, frameSampleCount(16_000, 20))

    // When: One complete 20 ms frame of microphone samples is encoded
    const frames = encoder.push(new Float32Array(320).fill(0.5))

    // Then: It emits exactly one 16-bit PCM frame with the expected byte size
    expect(frames).toHaveLength(1)
    expect(frames[0]).toHaveProperty('byteLength', expectedPcmFrameBytes(16_000, 20, 1))
  })

  it('resamples 48 kHz browser input to 16 kHz talk frames', () => {
    // Given: A PCM encoder receiving common 48 kHz browser audio for a 16 kHz camera stream
    const encoder = new PcmFrameEncoder(48_000, 16_000, frameSampleCount(16_000, 20))

    // When: Two partial browser chunks are pushed through the resampler
    const firstFrames = encoder.push(new Float32Array(480).fill(0.25))
    const secondFrames = encoder.push(new Float32Array(480).fill(0.25))

    // Then: The encoder buffers until it can emit one exact 20 ms 16 kHz PCM frame
    expect(firstFrames).toHaveLength(0)
    expect(secondFrames).toHaveLength(1)
    expect(secondFrames[0]).toHaveProperty('byteLength', 320 * Int16Array.BYTES_PER_ELEMENT)
  })

  it('does not drift across consecutive resampled frames', () => {
    // Given: One second of 48 kHz browser audio and a 16 kHz talk frame target
    const encoder = new PcmFrameEncoder(48_000, 16_000, frameSampleCount(16_000, 20))

    // When: The full second of audio is resampled into outbound talk frames
    const frames = encoder.push(new Float32Array(48_000).fill(0.1))

    // Then: It emits exactly 50 fixed-size frames without cumulative resampling drift
    expect(frames).toHaveLength(50)
    expect(frames.every((frame) => frame.byteLength === 640)).toBe(true)
  })
})
