import Capacitor
import Foundation
import UIKit

enum HomeSecDevicePluginError: LocalizedError {
    case missingBundleIdentifier

    var errorDescription: String? {
        switch self {
        case .missingBundleIdentifier:
            return "App bundle identifier is unavailable."
        }
    }
}

@objc(HomeSecDevicePlugin)
public class HomeSecDevicePlugin: CAPPlugin, CAPBridgedPlugin {
    public let identifier = "HomeSecDevicePlugin"
    public let jsName = "HomeSecDevice"
    public let pluginMethods: [CAPPluginMethod] = [
        CAPPluginMethod(name: "getRegistrationInfo", returnType: CAPPluginReturnPromise),
    ]

    @objc func getRegistrationInfo(_ call: CAPPluginCall) {
        do {
            guard let bundleIdentifier = Bundle.main.bundleIdentifier else {
                throw HomeSecDevicePluginError.missingBundleIdentifier
            }

            call.resolve([
                "apnsEnvironment": apnsEnvironment(),
                "appVersion": bundleValue("CFBundleShortVersionString") ?? NSNull(),
                "bundleId": bundleIdentifier,
                "deviceName": nullableDeviceName(),
            ])
        } catch {
            call.reject(error.localizedDescription, "HOMESEC_DEVICE_INFO_ERROR", error)
        }
    }

    private func apnsEnvironment() -> String {
        if let configured = bundleValue("HomeSecAPNSEnvironment")?.lowercased() {
            switch configured {
            case "development", "sandbox":
                return "sandbox"
            case "production":
                return "production"
            default:
                break
            }
        }

        #if DEBUG
        return "sandbox"
        #else
        return "production"
        #endif
    }

    private func bundleValue(_ key: String) -> String? {
        guard let value = Bundle.main.object(forInfoDictionaryKey: key) as? String else {
            return nil
        }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private func nullableDeviceName() -> Any {
        let trimmed = UIDevice.current.name.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? NSNull() : trimmed
    }
}
