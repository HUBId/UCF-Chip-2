# Chip-2 Architekturüberblick

Chip-2 verarbeitet eingehende **SignalFrames** und transformiert sie deterministisch weiter: SignalFrame → ISV/RSV → Profile und Overlays → ControlFrame. CBV-, PEV- und HBV-Signale werden entlang desselben Pfads eingeordnet, bevor sie in den ControlFlow gelangen. Windowing sorgt für klar definierte Auswerteintervalle, während tighten-only-Regeln sicherstellen, dass Bounds ausschließlich verschärft werden und keine Rücknahmen stattfinden.
