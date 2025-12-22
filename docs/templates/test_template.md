# Golden-Stream-Testtemplate

- **Input Frames:** Sequenz aus SignalFrames/ControlFrames mit fixierten Timestamps und deterministischen RC-Sets.
- **Setup:** Definierte Profile/Overlays, Fenstergrößen und Guardrails aktivieren.
- **Execution:** Pipeline in deterministischem Modus (keine Randomisierung, sortierte RCs) durchlaufen lassen.
- **Expected Outputs:** Erwarten von ControlFrames/Overlays pro Fenster; Snapshots für ISV/RSV/PVGS prüfen.
- **Determinism:** Anti-Flapping-Checks (minimale Umschaltfrequenz) und tighten-only Validierung.
