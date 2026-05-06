export function frameSampleCount(sampleRate: number, frameMs: number): number {
  return Math.max(1, Math.round((sampleRate * frameMs) / 1000))
}

// Keep this encoder behavior aligned with talkPcmProcessor.js.
// The AudioWorklet is loaded as a standalone browser module, so the worklet path
// cannot import this TypeScript helper directly.
export function floatToPcm16(frame: Float32Array): ArrayBuffer {
  const pcm = new Int16Array(frame.length)
  for (let index = 0; index < frame.length; index += 1) {
    const clamped = Math.max(-1, Math.min(1, frame[index] ?? 0))
    pcm[index] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff
  }
  return pcm.buffer
}

export function expectedPcmFrameBytes(sampleRate: number, frameMs: number, channels: number): number {
  return frameSampleCount(sampleRate, frameMs) * Math.max(1, channels) * Int16Array.BYTES_PER_ELEMENT
}

export class PcmFrameEncoder {
  private readonly sourceSampleRate: number
  private readonly targetSampleRate: number
  private readonly samplesPerFrame: number
  private readonly sourceToTargetRatio: number
  private readonly pendingSourceSamples: number[] = []
  private readonly pendingTargetSamples: number[] = []
  private sourcePosition = 0

  constructor(sourceSampleRate: number, targetSampleRate: number, samplesPerFrame: number) {
    this.sourceSampleRate = Math.max(1, sourceSampleRate)
    this.targetSampleRate = Math.max(1, targetSampleRate)
    this.samplesPerFrame = Math.max(1, samplesPerFrame)
    this.sourceToTargetRatio = this.sourceSampleRate / this.targetSampleRate
  }

  push(input: Float32Array): ArrayBuffer[] {
    const frames: ArrayBuffer[] = []
    if (input.length === 0) {
      return frames
    }

    if (this.sourceSampleRate === this.targetSampleRate) {
      for (const sample of input) {
        this.pendingTargetSamples.push(sample)
        this.flushReadyFrames(frames)
      }
      return frames
    }

    this.pendingSourceSamples.push(...input)
    while (this.sourcePosition + 1 < this.pendingSourceSamples.length) {
      const sampleIndex = Math.floor(this.sourcePosition)
      const fraction = this.sourcePosition - sampleIndex
      const current = this.pendingSourceSamples[sampleIndex] ?? 0
      const next = this.pendingSourceSamples[sampleIndex + 1] ?? current
      this.pendingTargetSamples.push(current + (next - current) * fraction)
      this.sourcePosition += this.sourceToTargetRatio
      this.flushReadyFrames(frames)
    }

    const discardCount = Math.max(0, Math.floor(this.sourcePosition) - 1)
    if (discardCount > 0) {
      this.pendingSourceSamples.splice(0, discardCount)
      this.sourcePosition -= discardCount
    }

    return frames
  }

  private flushReadyFrames(frames: ArrayBuffer[]): void {
    while (this.pendingTargetSamples.length >= this.samplesPerFrame) {
      const frame = new Float32Array(this.pendingTargetSamples.splice(0, this.samplesPerFrame))
      frames.push(floatToPcm16(frame))
    }
  }
}
