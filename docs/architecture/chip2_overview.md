# Chip-2 Architekturüberblick

Chip-2 verarbeitet eingehende **SignalFrames** und transformiert sie deterministisch weiter: SignalFrame → ISV/RSV → Profile und Overlays → ControlFrame. CBV-, PEV- und HBV-Signale werden entlang desselben Pfads eingeordnet, bevor sie in den ControlFlow gelangen. Windowing sorgt für klar definierte Auswerteintervalle, während tighten-only-Regeln sicherstellen, dass Bounds ausschließlich verschärft werden und keine Rücknahmen stattfinden.

## BrainBus Tick-Reihenfolge

Der BrainBus tickt alle Module in einer festen, deterministischen Reihenfolge. Zwischenstationen wie Baseline-Bestimmung oder Reason-Code-Merging folgen derselben Ordnung, damit Wiederholungen identische Ausgaben liefern:

1. **HPA** (Allostatic load & HBV Offsets)
2. **Baseline Resolver** (CBV/PEV/HBV Kombination)
3. **Cerebellum** (Tool-Divergenz + Suspend-Empfehlungen)
4. **LC** (Arousal/Hysterese)
5. **Serotonin** (Stabilität/Cooldown)
6. **Amygdala** (Threat + Vektoren)
7. **PAG** (Defense-Pattern)
8. **STN** (Policy-Pressure + Divergenz-Hinweise)
9. **PMRF** (Hold/Checkpoint)
10. **Dopamin/NAcc** (Progress/Replay/Reward-Block)
11. **Insula** (ISV-Synthese)
12. **Substantia Nigra** (DWM-Wahl)
13. **Superior Colliculus** (Orientierung/Salienz)
14. **PPRF** (Orient-Latch mit M3-Respekt)
15. **Hypothalamus** (Profile/Overlays/Cooldown)
16. **Emotion Field** (Snapshot der Emotionslage)

Die tighten-only-Regeln wirken auf Profil- und Overlay-Änderungen: strengere Profile setzen sich durch, Overlays werden nur gesetzt (nie innerhalb eines Ticks deaktiviert) und Reason Codes werden sortiert, dedupliziert und auf die ReasonSet-Grenzen gekappt.

## Completeness Pass

Der „Completeness Pass" stellt sicher, dass jedes DBM-Modul einen einheitlichen Input/Output besitzt, deterministisch normalisiert und durch Tests abgedeckt ist. Kriterien:

- Einheitliche Signaturen (`new`, `tick(&mut self, &Input)`), ReasonSet-basierte Outputs und gebundene Reason-Code-Längen.
- Für jedes Modul mindestens Basis-Unit-Tests, die HIGH/LOW-Übergänge und deterministisches Verhalten abdecken.
- Ein zentraler goldener Integrationstest im BrainBus, der eine Sequenz synthetischer BrainInputs durchläuft und identische Outputs (inkl. Reason-Code-Ordnung) erzwingt.
