import PhotosUI
import SwiftUI

struct RootView: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        @Bindable var app = app
        GeometryReader { geo in
            let width = geo.size.width + geo.safeAreaInsets.leading + geo.safeAreaInsets.trailing
            ZStack {
                Mk.bg.ignoresSafeArea()

                // Screens slide horizontally in physical coordinates (LTR context);
                // each screen's content is RTL.
                ZStack {
                    screen(.home) { HomeView() }
                    screen(.list) { ListScreen() }
                    screen(.camera) { CameraScreen() }
                    screen(.intro) { TrainIntroScreen() }
                    screen(.processing) { ProcessingScreen() }
                    screen(.train) { TrainScreen() }
                    screen(.result) { ResultScreen() }
                    screen(.refit) { RefitScreen() }
                    screen(.compare) { CompareScreen() }
                }
                .environment(\.layoutDirection, .leftToRight)
                .environment(\.screenWidth, width)

                if let toast = app.toast {
                    VStack {
                        Spacer()
                        ToastView(toast: toast)
                            .id(toast.id)
                            .padding(.bottom, 110)
                            .transition(.asymmetric(insertion: .scale(scale: 0.92).combined(with: .offset(y: 14)).combined(with: .opacity),
                                                    removal: .opacity))
                    }
                    .ignoresSafeArea(.container, edges: .bottom)
                    .allowsHitTesting(false)
                    .zIndex(45)
                }

                SheetHost()
                    .zIndex(50)
            }
        }
        .environment(\.layoutDirection, .rightToLeft)
        .environment(\.locale, Locale(identifier: "he_IL"))
        .font(Mk.rubik(16))
        .foregroundStyle(Mk.ink)
        .tint(Mk.orange)
        .photosPicker(isPresented: $app.showPhotoPicker, selection: $app.photoSelection, maxSelectionCount: 20,
                      selectionBehavior: .ordered, matching: .images, preferredItemEncoding: .current)
        .onChange(of: app.photoSelection) { _, items in
            if !items.isEmpty { app.importPhotos(items) }
        }
        .fileImporter(isPresented: $app.showFileImporter, allowedContentTypes: ImportTypes.all,
                      allowsMultipleSelection: true) { result in
            if case let .success(urls) = result, !urls.isEmpty { app.importFiles(urls) }
        }
    }

    @ViewBuilder
    private func screen<Content: View>(_ s: Screen, @ViewBuilder content: () -> Content) -> some View {
        let current = app.screen
        let isCurrent = s == current
        content()
            .environment(\.layoutDirection, .rightToLeft)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background { background(s) }
            .modifier(ScreenPlacement(screen: s, current: current))
            .allowsHitTesting(isCurrent)
            .accessibilityHidden(!isCurrent)
            .zIndex(isCurrent ? 2 : (s.depth > current.depth ? 1 : 0))
    }

    @ViewBuilder
    private func background(_ s: Screen) -> some View {
        switch s {
        case .home: Color.clear
        case .camera: Mk.cameraBg.ignoresSafeArea()
        case .refit: Mk.night.ignoresSafeArea()
        default: Mk.bg.ignoresSafeArea()
        }
    }
}

private struct ScreenWidthKey: EnvironmentKey {
    static let defaultValue: CGFloat = 390
}

extension EnvironmentValues {
    var screenWidth: CGFloat {
        get { self[ScreenWidthKey.self] }
        set { self[ScreenWidthKey.self] = newValue }
    }
}

/// The design's per-screen transform: current at rest; shallower screens drift right 30% and fade;
/// deeper screens wait off-screen to the left; processing and refit scale in and out.
private struct ScreenPlacement: ViewModifier {
    var screen: Screen
    var current: Screen
    @Environment(\.screenWidth) private var width

    func body(content: Content) -> some View {
        let (x, opacity, scale): (CGFloat, Double, CGFloat) = {
            if screen == current { return (0, 1, 1) }
            if screen.isModal { return (0, 0, 0.94) }
            if screen.depth < current.depth { return (width * 0.3, 0, 1) }
            return (-width, 0, 1)
        }()
        content
            .scaleEffect(scale)
            .offset(x: x)
            .opacity(opacity)
    }
}

/// The design's bottom pill toast.
struct ToastView: View {
    var toast: Toast

    var body: some View {
        Text(toast.text)
            .font(Mk.rubik(15))
            .foregroundStyle(Mk.cream)
            .lineLimit(1)
            .padding(.horizontal, 20)
            .frame(height: 46)
            .background(Capsule().fill(Mk.charcoal))
            .shadow(color: .black.opacity(0.4), radius: 15, y: 12)
    }
}
