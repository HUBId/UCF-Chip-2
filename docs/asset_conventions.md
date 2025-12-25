## Asset conventions

### Morphology labels (v2)

Each neuron may include up to 8 label key/value pairs. Keys and values are bounded
strings, and labels are ordered deterministically by `(k, v)` when encoded.

#### Standard label keys

* `pool`
  * **SN**: `EXEC`, `SIM`, `STAB`, `REPORT`, `INH`
  * **Amygdala**: `INTEGRITY`, `EXFIL`, `PROBING`, `TOOLSE`, `INH`
  * **Hypothalamus**: `P0`, `P1`, `P2`, `P3`, `O_SIM`, `O_EXP`, `O_NOV`, `INH`
* `role`
  * `E` (excitatory)
  * `I` (inhibitory)
