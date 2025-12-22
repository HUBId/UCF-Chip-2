# Chip-2 Interfaces

Chip-2 konsumiert und emittiert Protobuf-Objekte sowie interne Snapshots. Wichtige Artefakte sind:

- **SignalFrame / ControlFrame:** Primäre Ein- und Ausgabeframes der Pipeline, inklusive deterministischer RC-Sets (sortiert).
- **ISV/RSV Snapshots:** Zwischenstände für regulatorische Bewertungen; deterministisch erzeugt und versioniert.
- **PVGS Snapshots:** Capture der Proportional-/Verstärkungszustände mit quantisierten Werten.
- **Profile/Overlays:** Konfigurierte Reglerzustände und temporäre Anpassungen; Merge-Regeln bleiben deterministisch.
- **CBV/PEV/HBV:** Einbindung zusätzlicher Vektoren in den ControlFlow bei stabiler Sortierung und Windowing-Ausrichtung.

Deterministische Regeln umfassen sortierte RC-Listen, stabile Merge-Reihenfolgen sowie fixierte Fenstergrenzen, damit Replays identische Artefakte liefern.
