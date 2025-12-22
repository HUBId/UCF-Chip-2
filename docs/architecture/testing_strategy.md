# Teststrategie

- **Unit-Tests:** Abdeckung der Kernpfade mit deterministischen Inputs und klaren RC-Bounds.
- **Integrationstests:** End-to-End-Flows über SignalFrame → ISV/RSV → Profile/Overlays → ControlFrame inklusive CBV/PEV/HBV.
- **Golden-Stream-Tests:** Fixierte Input-Streams mit erwarteten Outputs; vergleichen von Snapshots (ISV/RSV/PVGS) unter tighten-only.
- **Anti-Flapping:** Sicherstellen, dass Profile/Overlays nur innerhalb definierter Fenster wechseln; minimale Umschaltabstände beachten.
- **Determinism:** Wiederholbare Ausführungen mit sortierten RCs, stabilen Fenstergrenzen und identischen Merge-Sequenzen.
