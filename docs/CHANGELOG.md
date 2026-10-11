# AOS Changelog

## Phase 3 — HumanIntent Layer + TTS (October 2026)

### What was built
- HumanIntentSignal — modality-independent intent abstraction
- MediaPipe Tasks API migration
- ContextManager semantic migration from gesture tokens to intent
- CentralOrchestrator accepts modality-independent HumanIntentSignal inputs; voice input/STT integration remains future work
- KittenTTS local TTS — non-blocking daemon thread
- 70 tests passing across L2-L5 and TTS integration
- prototype_token_id deprecation path established

### Architecture decisions made
- TTS always local to the machine with speakers — never remote
- AI proposes, kernel decides — ActionValidator invariant preserved
- KernelLoop classified as legacy, CentralOrchestrator is active path
- Remote RTX 4060 parked — not validated on current Arch setup

### Deferred to Phase 4
- ResourceWarnings from X11/PyScreeze — cause documented
- KDE Wayland capability backend
- scrot replacement with spectacle
