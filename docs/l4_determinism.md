# L4 determinism constraints

The L4 compartmental solver is designed to be deterministic when run with the same inputs.
The following assumptions must hold to preserve repeatable results:

- Use a fixed time step (`dt_ms`) and the same iteration count for each run.
- Compartments are iterated in ascending compartment ID order, and no parallelism is used.
- Floating-point determinism assumes identical CPU architecture and compiler flags.
  - Use the same Rust version, target triple, and floating-point settings across runs.
  - Avoid enabling fast-math or non-default FMA behavior that can reorder operations.
- The solver clamps membrane voltages to a fixed range to prevent divergent numerical
  behavior.

If these conditions are met, repeated runs should produce identical voltages, gating
states, and digests.
