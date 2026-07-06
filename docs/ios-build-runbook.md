# HomeSec iOS Build And Runbook

Last reviewed: 2026-06-14

This runbook covers personal HomeSec iPhone and iPad builds from this repo.
The current app is a Capacitor iOS shell around the React UI in `ui/`.

HomeSec intentionally supports the latest iOS major only. The native project
currently builds with the installed iOS 26.5 SDK and has
`IPHONEOS_DEPLOYMENT_TARGET = 26.0`. Older iOS 17/18 simulator runtimes may be
installed locally, but they are not supported targets for this app stream.

## Prerequisites

- macOS with Xcode installed and selected by `xcode-select`.
- Xcode command line tools available: `xcodebuild -version` should succeed.
- Node compatible with `ui/package.json` (`>=22.12.0`).
- pnpm compatible with the repo lockfile.
- Python/uv dependencies installed for backend validation.
- An Apple Developer account/team for real-device signing and APNs.
- A reachable HomeSec server over HTTPS, VPN, or local LAN.

Recommended preflight:

```bash
xcodebuild -showsdks
xcrun devicectl list devices
uv sync
pnpm --dir ui install
```

`xcodebuild -showsdks` should show an iOS SDK matching the latest supported
major version. On 2026-06-14 this repo was validated with iOS SDK 26.5 and iOS
Simulator SDK 26.5.

## Local Development Build

Use this path for simulator work and web/native asset sync checks.

```bash
pnpm --dir ui ios:sync
xcodebuild \
  -project ui/ios/App/App.xcodeproj \
  -scheme App \
  -destination 'platform=iOS Simulator,name=iPhone 17,OS=26.5' \
  -configuration Debug \
  -derivedDataPath /tmp/homesec-ios-qa \
  build
```

The `ios:sync` script builds the React app, copies web assets into
`ui/ios/App/App/public`, and regenerates the local Capacitor SPM package.
`ui/capacitor.config.ts` pins `experimental.ios.spm.swiftToolsVersion` to
`6.2`; keep that setting while the package platform is `.iOS(.v26)`.
Without it, Capacitor can regenerate `Package.swift` with Swift tools 5.9,
which Xcode cannot resolve for the iOS 26 package platform enum.

To install and launch a simulator build manually:

```bash
xcrun simctl bootstatus booted -b
xcrun simctl install booted /tmp/homesec-ios-qa/Build/Products/Debug-iphonesimulator/App.app
xcrun simctl launch booted com.levneiman.homesec
```

The expected first-launch screen is `Connect to HomeSec` with server URL and
API token controls.

## Personal Device Build

Use this path for installing the app on your own iPhone or iPad.

1. Connect the iPhone/iPad over USB or enable wireless debugging in Xcode.
2. Unlock the device and trust the Mac if prompted.
3. Confirm Xcode can see it:

   ```bash
   xcrun devicectl list devices
   ```

4. Sync the native assets:

   ```bash
   pnpm --dir ui ios:sync
   ```

5. Open the native project:

   ```bash
   open ui/ios/App/App.xcodeproj
   ```

6. In Xcode, select the `App` target and set Signing & Capabilities:
   - Team: your Apple Developer team.
   - Bundle Identifier: keep `com.levneiman.homesec` for the default personal
     build, or change it consistently in Xcode and backend APNs config if your
     Apple account requires a unique identifier.
   - Signing: automatic signing is expected.
   - Push Notifications capability must be present when testing APNs.

7. Select the connected device as the run destination and run the `App` scheme.

The Debug target uses `APS_ENVIRONMENT = development`; the Release target uses
`APS_ENVIRONMENT = production`. For personal device QA and sandbox pushes, run
Debug unless you are intentionally validating a production APNs profile.

## HomeSec Server Setup

The iOS shell stores the server URL and API token in the native Keychain bridge,
not WebView storage.

On first launch:

1. Enter the HomeSec server base URL.
2. Tap `Check server`.
3. Paste the HomeSec API token.
4. Tap `Save and continue`.

Use HTTPS or VPN whenever possible. Plain HTTP is only acceptable for a trusted
LAN/VPN development setup; the app allows local networking for LAN bootstrap but
should not be treated as secure over untrusted networks.

If server auth is disabled, the app can proceed for first LAN/VPN iteration, but
that is a personal-use convenience only. Do not expose auth-disabled HomeSec to
the public internet.

## APNs Sandbox Setup

APNs is optional for basic app browsing but required for notification QA.

Apple-side setup:

1. In the Apple Developer portal, make sure the bundle id has Push
   Notifications enabled.
2. Create or reuse an APNs Auth Key.
3. Record the key id and team id.
4. Download the `.p8` private key once and store it outside the repo.

HomeSec server environment variables:

```bash
export HOMESEC_APNS_KEY_ID='ABC123DEFG'
export HOMESEC_APNS_TEAM_ID='TEAM123456'
export HOMESEC_APNS_PRIVATE_KEY="$(cat /secure/path/AuthKey_ABC123DEFG.p8)"
```

Example notifier config:

```yaml
notifiers:
  - backend: apns_mobile
    config:
      bundle_id: com.levneiman.homesec
      environment: sandbox
      key_id_env: HOMESEC_APNS_KEY_ID
      team_id_env: HOMESEC_APNS_TEAM_ID
      private_key_env: HOMESEC_APNS_PRIVATE_KEY
```

Use `environment: sandbox` for Debug builds and `environment: production` only
for Release/TestFlight/App Store builds signed with the production APNs
environment. The bundle id and APNs environment must match the registered mobile
device record, otherwise HomeSec will not find an enabled APNs target.

Never commit APNs keys, HomeSec API tokens, RTSP credentials, or `.env` files.

## QA Checklist

Run the real-device QA matrix before treating a personal build as ready:

- First launch setup renders.
- VPN/LAN server URL check succeeds.
- API token paste auth succeeds.
- API token persists after app restart.
- Live page loads.
- Events page loads.
- Event detail playback works.
- Live HLS preview works.
- Push-to-talk path works if enabled for the configured camera.
- Backgrounding stops active preview and talk sessions.
- Plain APNs push is received.
- Tapping a push opens the event detail route.
- iPad layout is usable.

File bugs for failures and keep iOS-19 updated with pass/fail notes.

## Future TestFlight Build

TestFlight is not required for the first personal release. When it is needed:

1. Switch to a unique production bundle id if `com.levneiman.homesec` is not
   owned by the target Apple Developer team.
2. Keep `APS_ENVIRONMENT = production` for Release.
3. Use `environment: production` in the `apns_mobile` notifier config.
4. Archive from Xcode with the `App` scheme.
5. Upload through Xcode Organizer or `xcrun altool`/Transporter.
6. Re-test APNs because sandbox device tokens do not work against production
   APNs, and production tokens do not work against sandbox APNs.

## Troubleshooting

- `No devices found.` from `xcrun devicectl list devices`: unlock the device,
  trust the Mac, reconnect USB, or enable wireless debugging from Xcode.
- `PackageDescription.SupportedPlatform.IOSVersion.v26 is unavailable`: rerun
  `pnpm --dir ui ios:sync` and confirm `ui/ios/App/CapApp-SPM/Package.swift`
  starts with `// swift-tools-version: 6.2`.
- App installs but cannot connect to HomeSec: confirm the iPhone can reach the
  server URL in Safari over the same VPN/LAN, and confirm auth is enabled with
  the expected bearer token.
- No APNs devices receive alerts: confirm the iOS app registered after setup,
  the device is enabled in the mobile device list, the APNs environment matches
  the build configuration, and `bundle_id` matches the app bundle identifier.
- Push tap opens the app but not the event: confirm payload `data.route` is an
  app-relative route such as `/events/<clip_id>?from=notification`.
