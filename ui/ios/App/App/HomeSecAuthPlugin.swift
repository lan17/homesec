import Capacitor
import Foundation

enum HomeSecAuthPluginError: LocalizedError {
    case invalidServerBaseUrl(String)
    case missingValue(String)
    case serverBaseUrlRequired

    var errorDescription: String? {
        switch self {
        case .invalidServerBaseUrl(let value):
            return "Invalid HomeSec server URL: \(value)"
        case .missingValue(let field):
            return "\(field) is required."
        case .serverBaseUrlRequired:
            return "Set the HomeSec server URL before storing an API token."
        }
    }
}

@objc(HomeSecAuthPlugin)
public class HomeSecAuthPlugin: CAPPlugin, CAPBridgedPlugin {
    public let identifier = "HomeSecAuthPlugin"
    public let jsName = "HomeSecAuth"
    public let pluginMethods: [CAPPluginMethod] = [
        CAPPluginMethod(name: "getServerBaseUrl", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "setServerBaseUrl", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "clearServerBaseUrl", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "getApiToken", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "setApiToken", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "clearApiToken", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "getAuthDisabledReady", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "setAuthDisabledReady", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "clearAuthDisabledReady", returnType: CAPPluginReturnPromise),
    ]

    private let authDisabledReadyPrefix = "auth-disabled-ready:"
    private let keychain = HomeSecKeychainStore()
    private let serverBaseUrlAccount = "server-base-url"
    private let tokenAccountPrefix = "api-token:"

    @objc func getServerBaseUrl(_ call: CAPPluginCall) {
        resolveStoredValue(call, account: serverBaseUrlAccount)
    }

    @objc func setServerBaseUrl(_ call: CAPPluginCall) {
        do {
            let value = try requiredString(call, key: "value")
            let normalized = try normalizedServerBaseUrl(value)
            let previousBaseUrl = try currentServerBaseUrl()
            if let previousBaseUrl, previousBaseUrl != normalized {
                try keychain.delete(account: "\(tokenAccountPrefix)\(previousBaseUrl)")
                try keychain.delete(account: "\(authDisabledReadyPrefix)\(previousBaseUrl)")
            }
            try keychain.set(normalized, account: serverBaseUrlAccount)
            call.resolve()
        } catch {
            reject(call, error: error)
        }
    }

    @objc func clearServerBaseUrl(_ call: CAPPluginCall) {
        do {
            if let tokenAccount = try currentApiTokenAccount() {
                try keychain.delete(account: tokenAccount)
            }
            if let authDisabledAccount = try currentAuthDisabledReadyAccount() {
                try keychain.delete(account: authDisabledAccount)
            }
            try keychain.delete(account: serverBaseUrlAccount)
            call.resolve()
        } catch {
            reject(call, error: error)
        }
    }

    @objc func getApiToken(_ call: CAPPluginCall) {
        do {
            guard let account = try currentApiTokenAccount() else {
                call.resolve(["value": NSNull()])
                return
            }
            resolveStoredValue(call, account: account)
        } catch {
            reject(call, error: error)
        }
    }

    @objc func setApiToken(_ call: CAPPluginCall) {
        do {
            let value = try requiredString(call, key: "value")
            let account = try requiredApiTokenAccount()
            try keychain.set(value, account: account)
            call.resolve()
        } catch {
            reject(call, error: error)
        }
    }

    @objc func clearApiToken(_ call: CAPPluginCall) {
        do {
            if let account = try currentApiTokenAccount() {
                try keychain.delete(account: account)
            }
            call.resolve()
        } catch {
            reject(call, error: error)
        }
    }

    @objc func getAuthDisabledReady(_ call: CAPPluginCall) {
        do {
            guard let account = try currentAuthDisabledReadyAccount() else {
                call.resolve(["value": false])
                return
            }
            call.resolve(["value": try keychain.read(account: account) == "true"])
        } catch {
            reject(call, error: error)
        }
    }

    @objc func setAuthDisabledReady(_ call: CAPPluginCall) {
        do {
            let ready = call.getBool("value", false)
            let account = try requiredAuthDisabledReadyAccount()
            if ready {
                try keychain.set("true", account: account)
            } else {
                try keychain.delete(account: account)
            }
            call.resolve()
        } catch {
            reject(call, error: error)
        }
    }

    @objc func clearAuthDisabledReady(_ call: CAPPluginCall) {
        do {
            if let account = try currentAuthDisabledReadyAccount() {
                try keychain.delete(account: account)
            }
            call.resolve()
        } catch {
            reject(call, error: error)
        }
    }

    private func resolveStoredValue(_ call: CAPPluginCall, account: String) {
        do {
            if let value = try keychain.read(account: account) {
                call.resolve(["value": value])
            } else {
                call.resolve(["value": NSNull()])
            }
        } catch {
            reject(call, error: error)
        }
    }

    private func requiredString(_ call: CAPPluginCall, key: String) throws -> String {
        guard let value = call.getString(key), !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw HomeSecAuthPluginError.missingValue(key)
        }
        return value
    }

    private func normalizedServerBaseUrl(_ value: String) throws -> String {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard
            var components = URLComponents(string: trimmed),
            let scheme = components.scheme?.lowercased(),
            (scheme == "http" || scheme == "https"),
            components.host != nil
        else {
            throw HomeSecAuthPluginError.invalidServerBaseUrl(value)
        }
        components.scheme = scheme
        var path = components.percentEncodedPath
        while path.count > 1 && path.hasSuffix("/") {
            path.removeLast()
        }
        components.percentEncodedPath = path == "/" ? "" : path
        components.query = nil
        components.fragment = nil
        guard let normalized = components.string else {
            throw HomeSecAuthPluginError.invalidServerBaseUrl(value)
        }
        return normalized
    }

    private func currentServerBaseUrl() throws -> String? {
        guard let serverBaseUrl = try keychain.read(account: serverBaseUrlAccount) else {
            return nil
        }
        return try normalizedServerBaseUrl(serverBaseUrl)
    }

    private func currentApiTokenAccount() throws -> String? {
        guard let serverBaseUrl = try currentServerBaseUrl() else {
            return nil
        }
        return "\(tokenAccountPrefix)\(serverBaseUrl)"
    }

    private func requiredApiTokenAccount() throws -> String {
        guard let account = try currentApiTokenAccount() else {
            throw HomeSecAuthPluginError.serverBaseUrlRequired
        }
        return account
    }

    private func currentAuthDisabledReadyAccount() throws -> String? {
        guard let serverBaseUrl = try currentServerBaseUrl() else {
            return nil
        }
        return "\(authDisabledReadyPrefix)\(serverBaseUrl)"
    }

    private func requiredAuthDisabledReadyAccount() throws -> String {
        guard let account = try currentAuthDisabledReadyAccount() else {
            throw HomeSecAuthPluginError.serverBaseUrlRequired
        }
        return account
    }

    private func reject(_ call: CAPPluginCall, error: Error) {
        call.reject(error.localizedDescription, "HOMESEC_AUTH_STORAGE_ERROR", error)
    }
}
