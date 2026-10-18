# Spec Delta

## ADDED Requirements

### Requirement: Isolated clip preparation

Built-in clip analysis SHALL perform general media preparation inside isolated analysis workers using HomeSec's packaged, pinned media implementation. It SHALL NOT invoke a media CLI or silently use another installed decoder. A preparation failure SHALL follow the affected stage's failure rules; pipeline policy, model inference and provider requests SHALL retain their existing ownership.

#### Scenario: Installed media tools do not change analysis preparation

- **GIVEN** a supported local clip and an installation with unrelated system media tools and Python decoder packages
- **WHEN** built-in filtering or eligible image analysis prepares that clip
- **THEN** preparation uses HomeSec's packaged media implementation inside an analysis worker
- **AND** system CLI tools and an independent Python-package decoder are not used for that preparation

#### Scenario: Preparation failure is contained to analysis work

- **GIVEN** an accepted clip whose preparation worker fails
- **WHEN** processing handles that stage's failure
- **THEN** filtering or analysis follows its existing failure and retry rules
- **AND** upload, notification fallback and repository writes retain their existing independent behavior

### Requirement: Consumer-compatible prepared frames

General preparation SHALL preserve each consumer's selected-frame identity, timestamp labels, orientation, dimensions and settings. YOLO SHALL receive original-size uncompressed BGR images and retain its result and early-exit rules. VLM images SHALL preserve sampling, no-upscale resizing and configured JPEG quality semantics. Equivalent pixels and encodings SHALL pass explicit parity checks; byte-identical JPEG output is not required.

#### Scenario: YOLO sampling and early exit remain compatible

- **GIVEN** a supported clip and effective detector settings
- **WHEN** the existing Python model evaluates prepared images
- **THEN** images follow the existing `sample_fps` frame-index stride and retain oriented frame height for box filtering
- **AND** `min_hits` continues to count distinct accepted classes, including existing confidence and box-height boundary behavior
- **AND** reaching early exit prevents requesting later samples or eagerly preparing the remaining clip

#### Scenario: VLM sampling and image settings remain compatible

- **GIVEN** a readable clip with reported frame count N, positive frame limit M, and supported image settings
- **WHEN** VLM images are prepared
- **THEN** selected indices are all indices when N is at most M, otherwise floor(i times N divided by M) for i from zero through M minus one
- **AND** successful samples retain the baseline timestamp labels and orientation
- **AND** resizing preserves aspect ratio with existing dimension rounding, never upscales, and respects the configured maximum size
- **AND** equivalent JPEG images use the configured quality semantics and pass the documented image-parity gate

#### Scenario: Decoder metadata and read failures retain existing meaning

- **GIVEN** a supported clip with count estimates, rotation, unusual timestamps or a failed selected read
- **WHEN** native preparation is compared with the frozen Python baseline
- **THEN** metadata and seeking produce the same selected-frame identities, dimensions and formatted timestamps
- **AND** YOLO stops on a failed read and retains its existing empty or accumulated result semantics
- **AND** VLM preparation skips a failed selected read and fails before a provider request when no images survive
- **AND** an empty or partial detector result does not certify complete inspection or safe deletion

### Requirement: Conditional and bounded preparation

Preparation SHALL occur only when its consuming stage is eligible. Its admitted jobs and retained media SHALL remain bounded by worker capacity and effective preprocessing settings, without accumulating entire decoded clips. Raw frames SHALL NOT enter JSON control payloads, workflow persistence or logs. Sharing prepared output SHALL NOT change consumer eligibility or force unnecessary preparation.

#### Scenario: Skipped analysis performs no VLM preparation

- **GIVEN** filtering fails, analysis is disabled, or trigger-only analysis has no matching trigger
- **WHEN** processing determines stage eligibility
- **THEN** no VLM preparation job or provider request is started for that processing run
- **AND** existing skip reasons and upload overlap remain applicable

#### Scenario: Slow or cancelled work does not grow a hidden queue

- **GIVEN** preparation capacity is occupied and further analysis arrives
- **WHEN** callers wait or cancel while accepted work is still running
- **THEN** additional work waits for actual capacity rather than accumulating an unbounded process queue
- **AND** retained YOLO media is limited to the current preparation step and VLM output to its effective frame and image settings
- **AND** cancellation does not release occupied capacity until that work actually settles

### Requirement: Cooperative analysis-worker cancellation and recovery

Caller cancellation SHALL affect only that job. Plugin shutdown SHALL stop admission and cancel queued work. While their owner remains active, workers SHALL stop preparation cooperatively; model inference can finish but its cancelled result SHALL NOT be accepted. Crashes SHALL fail attempts and permit a fresh pool under Python retry policy. Whole-runtime supervision can interrupt cleanup; cooperation SHALL NOT imply forced termination of a blocked call.

#### Scenario: Cancellation releases preparation resources

- **GIVEN** accepted preparation is running when its caller is cancelled
- **WHEN** the worker reaches a cancellation check between preparation steps or after inference
- **THEN** it stops further samples, closes its decoder and releases job-owned media and cancellation resources
- **AND** a cancelled job sends no later provider request or successful stage result
- **AND** its capacity is reusable after the underlying work settles
- **AND** unrelated jobs remain eligible to complete or be admitted

#### Scenario: Pool failure is retried through the pipeline

- **GIVEN** a worker exits abruptly and the pool becomes unusable
- **WHEN** affected attempts fail and a later attempt is admitted under existing retry policy
- **THEN** affected attempts do not manufacture successful empty results
- **AND** the later attempt can use a fresh pool without restarting camera capture
- **AND** pool replacement does not independently repeat inference, provider requests or workflow writes

#### Scenario: Cooperative shutdown preserves unfinished ownership

- **GIVEN** plugin shutdown starts while a native or model call has not returned and its owning runtime remains active
- **WHEN** the configured shutdown wait budget expires
- **THEN** the event loop is not blocked indefinitely by a synchronous pool join
- **AND** the runtime retains responsibility for eventual worker and job-resource cleanup
- **AND** shutdown reports unfinished work without claiming that its process was forcibly terminated

#### Scenario: Whole-runtime termination interrupts plugin cleanup

- **GIVEN** whole-runtime process supervision terminates analysis workers during shutdown
- **WHEN** a worker cannot complete its cooperative cleanup
- **THEN** that termination is not reported as successful cooperative job completion
- **AND** no stage accepts a successful result from that interrupted attempt
- **AND** plugin cleanup does not claim to guarantee descendant cleanup after an abrupt runtime-parent exit

### Requirement: Reproducible analysis-media installation

Supported Linux and macOS HomeSec installations SHALL include the native analysis component built from pinned, verified media sources. Wheel, source and editable development installs SHALL use the same source identity and SHALL NOT select system FFmpeg libraries. Installation failures SHALL explain missing build prerequisites; runtime preparation SHALL NOT silently fall back to a separately versioned decoder.

#### Scenario: Clean installation includes native preparation

- **GIVEN** a supported Linux or macOS machine with documented prerequisites
- **WHEN** HomeSec is installed from its wheel, source distribution or editable checkout
- **THEN** the same HomeSec distribution provides an importable native preparation component for the declared Python and platform target
- **AND** a known local-clip smoke test works without system FFmpeg libraries or executable tools

#### Scenario: Missing source-build prerequisites are actionable

- **GIVEN** a source installation lacks a required compiler or build tool
- **WHEN** its native build starts
- **THEN** installation fails with an actionable prerequisite error
- **AND** it does not use an unrelated system decoder to complete the installation
