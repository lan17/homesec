import type { BackendFormStepProps } from './types'
import { readString } from './configReaders'

export function RtspForm({ config, onChange }: BackendFormStepProps) {
  const rtspUrl = readString(config, 'rtsp_url', '')
  const outputDir = readString(config, 'output_dir', './recordings')
  const previewStream = readString(config, 'preview_stream', 'main')

  return (
    <div className="inline-form">
      <label className="field-label" htmlFor="camera-rtsp-url">
        RTSP URL
        <input
          id="camera-rtsp-url"
          className="input"
          type="text"
          value={rtspUrl}
          placeholder="rtsp://username:password@camera.local/stream"
          onChange={(event) => {
            onChange({
              ...config,
              rtsp_url: event.target.value,
            })
          }}
        />
      </label>

      <label className="field-label" htmlFor="camera-rtsp-output-dir">
        Output directory
        <input
          id="camera-rtsp-output-dir"
          className="input"
          type="text"
          value={outputDir}
          onChange={(event) => {
            onChange({
              ...config,
              output_dir: event.target.value,
            })
          }}
        />
      </label>

      <label className="field-label" htmlFor="camera-rtsp-preview-stream">
        Preview stream
        <select
          id="camera-rtsp-preview-stream"
          className="input"
          value={previewStream === 'detect' ? 'detect' : 'main'}
          onChange={(event) => {
            onChange({
              ...config,
              preview_stream: event.target.value,
            })
          }}
        >
          <option value="main">Main</option>
          <option value="detect">Detect</option>
        </select>
      </label>
    </div>
  )
}
