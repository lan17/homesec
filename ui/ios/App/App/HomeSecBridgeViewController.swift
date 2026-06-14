import Capacitor
import UIKit

@objc(HomeSecBridgeViewController)
class HomeSecBridgeViewController: CAPBridgeViewController {
    override func capacitorDidLoad() {
        super.capacitorDidLoad()
        bridge?.registerPluginInstance(HomeSecAuthPlugin())
    }
}
