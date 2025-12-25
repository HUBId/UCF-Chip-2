# Asset Conventions

## Hypothalamus L4 Asset Convention v1

Hypothalamus L4 assets do not carry neuron labels yet, so pool assignment is
deterministic by neuron id. Each neuron is a 3-compartment cell. Total neurons
must be exactly 16; otherwise the bundle is rejected.

### Profile pools

| Pool | Neuron IDs |
| --- | --- |
| P0 M0 | 0–1 |
| P1 M1 | 2–3 |
| P2 M2 | 4–5 |
| P3 M3 | 6–7 |

### Overlay pools

| Pool | Neuron IDs |
| --- | --- |
| O_sim | 8–9 |
| O_exp | 10–11 |
| O_nov | 12–13 |

### Inhibitory control

| Pool | Neuron IDs |
| --- | --- |
| Inhibitory | 14–15 |
