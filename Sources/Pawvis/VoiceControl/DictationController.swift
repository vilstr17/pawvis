import Foundation
import PawvisCore

/// Continuous dictation, toggled by a gesture action: speech streams in and
/// whatever the recognizer finalizes is typed into the focused app — no wake
/// word, no command parsing. Toggle the gesture again to stop. This is NOT
/// voice control (no commands, no agent hand-off, nothing is executed) — it
/// types exactly what it hears, and nothing else.
///
/// Runs its own `SpeechEngine` so it is fully independent of the voice-control
/// session (mic, permission and lifecycle separate). Speech events arrive on
/// the main queue; everything here is @MainActor, matching `VoiceController`.
@MainActor
final class DictationController: ObservableObject {
    enum State: Equatable {
        case off
        case connecting
        case dictating
        case error(String)

        var isActive: Bool {
            switch self {
            case .off, .error: return false
            default: return true
            }
        }
    }

    @Published private(set) var state: State = .off
    /// Set while the engine is live and speech is streaming in — the gesture
    /// pill and the menu read this.
    @Published private(set) var typedCharacterCount = 0

    private let typer = TextTyper()
    private var engine: SpeechEngine?
    private var sessionGeneration = 0

    /// True when the user has granted microphone access (Pawvis's own voice
    /// control grants the same permission, so this usually sticks).
    var microphoneGranted: Bool { Permissions.microphone() == .granted }

    /// Menu bar entry point (also wired to the gesture action): start when
    /// off, stop when dictating.
    func toggle() {
        if state.isActive {
            stop()
        } else {
            start()
        }
    }

    func start() {
        guard !state.isActive else { return }
        switch Permissions.microphone() {
        case .denied:
            state = .error("Microphone access denied — enable in System Settings")
        case .notDetermined:
            state = .connecting
            let session = sessionGeneration
            Task { [weak self] in
                let granted = await Permissions.requestMicrophone()
                guard let self else { return }
                // The dialog can outlive the toggle: a grant landing after a
                // stop() must not launch a hot mic under state == .off.
                guard self.sessionGeneration == session else { return }
                if granted {
                    self.launchEngine()
                } else {
                    self.state = .error("Microphone access denied")
                }
            }
        case .granted:
            state = .connecting
            launchEngine()
        }
    }

    func stop() {
        guard state.isActive else { return }
        sessionGeneration += 1
        engine?.stop()
        engine = nil
        state = .off
    }

    private func launchEngine() {
        // Defensive: nothing should reach here with an engine live (start()
        // refuses while active; stale grants are dropped) — overwriting one
        // would orphan it with the mic still hot.
        engine?.stop()
        let config = VoiceControlConfig()
        let engine = SpeechEngine(config: config)
        self.engine = engine
        engine.onEvent = { [weak self, weak engine] event in
            // Identity-gated like VoiceController: a stopped engine can still
            // have events queued on the main run loop; an event from any
            // engine but the current one is a dead session talking.
            guard let self, let engine, self.engine === engine else { return }
            self.handle(event)
        }
        engine.start()
    }

    private func handle(_ event: SpeechEvent) {
        switch event {
        case .ready:
            if state == .connecting { state = .dictating }

        case .hypothesis:
            // No capsule work of our own — voice control's transcript overlay
            // is woken only by wake-worded speech, and dictation types
            // silently while the recognizer revises its hypothesis.
            break

        case .completed(_, let transcript):
            let text = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty, state.isActive else { return }
            typer.type(text + " ")
            typedCharacterCount += text.count + 1

        case .failed(let message):
            sessionGeneration += 1
            engine?.stop()
            engine = nil
            state = .error("Dictation failed: \(message)")
        }
    }
}