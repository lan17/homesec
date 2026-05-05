function floatToPcm16(frame) {
  const pcm = new Int16Array(frame.length)
  for (let index = 0; index < frame.length; index += 1) {
    const clamped = Math.max(-1, Math.min(1, frame[index] || 0))
    pcm[index] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff
  }
  return pcm.buffer
}

class LinearPcmFrameEncoder {
  constructor(sourceSampleRate, targetSampleRate, frameSamples) {
    this.sourceSampleRate = Math.max(1, Number(sourceSampleRate) || targetSampleRate || 16000)
    this.targetSampleRate = Math.max(1, Number(targetSampleRate) || 16000)
    this.frameSamples = Math.max(1, Number(frameSamples) || 320)
    this.sourceToTargetRatio = this.sourceSampleRate / this.targetSampleRate
    this.pendingSourceSamples = []
    this.pendingTargetSamples = []
    this.sourcePosition = 0
  }

  push(input) {
    const frames = []
    if (!input || input.length === 0) {
      return frames
    }

    if (this.sourceSampleRate === this.targetSampleRate) {
      for (let index = 0; index < input.length; index += 1) {
        this.pendingTargetSamples.push(input[index])
        this.flushReadyFrames(frames)
      }
      return frames
    }

    for (let index = 0; index < input.length; index += 1) {
      this.pendingSourceSamples.push(input[index])
    }

    while (this.sourcePosition + 1 < this.pendingSourceSamples.length) {
      const sampleIndex = Math.floor(this.sourcePosition)
      const fraction = this.sourcePosition - sampleIndex
      const current = this.pendingSourceSamples[sampleIndex] || 0
      const next = this.pendingSourceSamples[sampleIndex + 1] || current
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

  flushReadyFrames(frames) {
    while (this.pendingTargetSamples.length >= this.frameSamples) {
      const frame = new Float32Array(this.pendingTargetSamples.splice(0, this.frameSamples))
      frames.push(floatToPcm16(frame))
    }
  }
}

class TalkPcmProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super()
    const processorOptions = options.processorOptions || {}
    this.encoder = new LinearPcmFrameEncoder(
      processorOptions.sourceSampleRate || sampleRate,
      processorOptions.targetSampleRate,
      processorOptions.frameSamples,
    )
  }

  process(inputs) {
    const input = inputs[0]
    const channel = input && input[0]
    if (!channel) {
      return true
    }

    for (const frame of this.encoder.push(channel)) {
      this.port.postMessage(frame, [frame])
    }

    return true
  }
}

registerProcessor('talk-pcm-processor', TalkPcmProcessor)
