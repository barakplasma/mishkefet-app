import SwiftData
import SwiftUI
import UIKit

@main
struct MishkefetApp: App {
    @State private var model: AppModel
    @Environment(\.scenePhase) private var scenePhase
    private let container: ModelContainer

    init() {
        let launch = LaunchOptions.current
        let container: ModelContainer
        do {
            // Creates Application Support before the store opens there.
            _ = Storage.scansDirectory
            let config = ModelConfiguration(isStoredInMemoryOnly: launch.inMemoryStore)
            container = try ModelContainer(for: Scan.self, Sample.self, configurations: config)
        } catch {
            fatalError("Could not open the scan store: \(error)")
        }
        self.container = container

        var engine: ReadingEngine = CoreReadingEngine()
        var defaults = UserDefaults.standard
        #if DEBUG
        if launch.fakeEngine { engine = FakeReadingEngine() }
        if launch.seedDemo, let d = UserDefaults(suiteName: "demo") {
            d.removePersistentDomain(forName: "demo")
            defaults = d
        }
        #endif
        let model = AppModel(container: container, engine: engine, defaults: defaults)
        _model = State(initialValue: model)
        #if DEBUG
        DemoContent.apply(launch, to: model)
        #endif
    }

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(model)
                .modelContainer(container)
                .onReceive(NotificationCenter.default.publisher(for: UIApplication.didReceiveMemoryWarningNotification)) { _ in
                    model.memoryWarning()
                }
        }
        .onChange(of: scenePhase) { _, phase in
            if phase == .background { model.enteredBackground() }
        }
    }
}

/// Launch arguments. Everything except the defaults is honoured in DEBUG builds only.
struct LaunchOptions {
    var fakeEngine = false
    var seedDemo = false
    var screen: String?
    var sheet: String?
    var readFile: String?

    var inMemoryStore: Bool { seedDemo }

    static let current: LaunchOptions = {
        var o = LaunchOptions()
        #if DEBUG
        let args = ProcessInfo.processInfo.arguments
        func value(_ key: String) -> String? {
            guard let i = args.firstIndex(of: key), i + 1 < args.count else { return nil }
            return args[i + 1]
        }
        o.fakeEngine = args.contains("-fakeEngine")
        o.seedDemo = args.contains("-seedDemo")
        o.screen = value("-screen")
        o.sheet = value("-sheet")
        o.readFile = value("-readFile")
        #endif
        return o
    }()
}
