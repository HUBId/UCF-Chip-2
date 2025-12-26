# Determinism Runbook

This runbook captures the deterministic build profiles used by Chip 2, the feature flags that
select each profile, and the evidence emitted for reproducibility.

## Supported build profiles

### default (rules + minimal)

**Required features**
- None (default feature set).

**Determinism guarantees**
- Rule-only pipeline; outputs are deterministic for the same inputs and configuration.
- No microcircuit state or floating-point integration is involved.

**Boundedness caps**
- `dbm_core::limits::MAX_REASON_CODES`
- `dbm_core::limits::MAX_SALIENCE_ITEMS`
- `dbm_core::limits::MAX_SUSPEND_RECS`
- `dbm_core::limits::MAX_EVIDENCE_REFS`
- `dbm_core::limits::MAX_LOCK_MS`

**Evidence emitted**
- `mc_cfg`: no
- `mc_snap`: no
- `plasticity`: no
- `asset bundle`: no
- `trace run`: no

---

### microcircuit (integer attractors)

**Required features**
- `microcircuit-sn-attractor`
- `microcircuit-lc-spike`
- `microcircuit-serotonin-attractor`
- `microcircuit-dopamin-attractor`
- `microcircuit-amygdala-pop`
- `microcircuit-pag-attractor`
- `microcircuit-stn-hold`
- `microcircuit-pmrf-rhythm`
- `microcircuit-sc-attractor`
- `microcircuit-cerebellum-pop`
- `microcircuit-insula-fusion`
- `microcircuit-hypothalamus-setpoint`

**Determinism guarantees**
- Integer attractor dynamics are deterministic for the same inputs.
- Evidence ordering is deterministic (e.g., microcircuit snapshots are stable).

**Boundedness caps**
- Same DBM caps as default (`MAX_REASON_CODES`, `MAX_SALIENCE_ITEMS`, `MAX_SUSPEND_RECS`,
  `MAX_EVIDENCE_REFS`, `MAX_LOCK_MS`).

**Evidence emitted**
- `mc_cfg`: yes (microcircuit config evidence)
- `mc_snap`: yes (e.g., LC/SN micro snapshots)
- `plasticity`: no
- `asset bundle`: no
- `trace run`: no

---

### biophys-l3 (LIF + STP)

**Required features**
- `biophys`
- `biophys-lc`
- `biophys-sn`
- `biophys-stn`
- `biophys-pmrf`
- `biophys-amygdala`
- `biophys-pag`

**Determinism guarantees**
- Deterministic floating-point integration for LIF + STP given the same target/inputs.
- Stable event ordering within a single target (no parallelism).

**Boundedness caps**
- DBM caps (`MAX_REASON_CODES`, `MAX_SALIENCE_ITEMS`, `MAX_SUSPEND_RECS`,
  `MAX_EVIDENCE_REFS`, `MAX_LOCK_MS`).

**Evidence emitted**
- `mc_cfg`: yes
- `mc_snap`: yes
- `plasticity`: no
- `asset bundle`: no
- `trace run`: no

---

### biophys-l4-core (compartments + synapses)

**Required features**
- `biophys`
- `biophys-l4`
- `biophys-l4-synapses`

**Determinism guarantees**
- Deterministic floating-point integration for compartmental solvers and synapse queues
  on the same target.
- Deterministic ordering of synapse events and accumulators within configured queue limits.

**Boundedness caps**
- `biophys_morphology::MAX_COMPARTMENTS`
- `biophys_synapses_l4` clamps (e.g., max synapse conductance)
- `biophys_event_queue_l4::QueueLimits` (max events per bucket/total)

**Evidence emitted**
- `mc_cfg`: yes
- `mc_snap`: yes
- `plasticity`: no
- `asset bundle`: no
- `trace run`: no

---

### biophys-l4-full (synapses + NMDA + STP + modulation + homeostasis + assets)

**Required features**
- `biophys`
- `biophys-l4`
- `biophys-l4-synapses`
- `biophys-l4-nmda`
- `biophys-l4-stp`
- `biophys-l4-modulation`
- `biophys-l4-homeostasis`
- `biophys-l4-plasticity`
- `biophys-l4-targeting`
- `biophys-l4-morphology-multi`
- `biophys-l4-ca`
- `biophys-l4-ca-feedback`
- `biophys-l4-sn`
- `biophys-l4-hypothalamus`
- `biophys-l4-insula`
- `biophys-l4-amygdala`
- `biophys-l4-pag`
- Asset-backed circuits (when enabled):
  - `biophys-l4-sn-assets`
  - `biophys-l4-amygdala-assets`
  - `biophys-l4-hypothalamus-assets`

**Determinism guarantees**
- Deterministic floating-point integration for L4 compartmental circuits on the same target.
- Plasticity snapshots are deterministic for a fixed sequence of inputs.
- Asset-backed circuits are deterministic when asset bundles are identical and validated.

**Boundedness caps**
- `biophys_assets::MAX_NEURONS`
- `biophys_assets::MAX_EDGES`
- `biophys_assets::MAX_COMPARTMENTS_PER_NEURON`
- `biophys_morphology::MAX_COMPARTMENTS`
- `biophys_synapses_l4` conductance caps
- `biophys_event_queue_l4::QueueLimits` (max events per bucket/total)

**Evidence emitted**
- `mc_cfg`: yes
- `mc_snap`: yes
- `plasticity`: yes
- `asset bundle`: yes (asset manifest/pool digests)
- `trace run`: optional (only when trace features are enabled)

---

### biophys-l4-full-parallel (optional)

**Required features**
- All `biophys-l4-full` features
- `biophys-parallel`

**Determinism guarantees**
- Deterministic within the same target when parallel scheduling order is stable.
- Use only for stable builds where parallel determinism is validated.

**Boundedness caps**
- Same as biophys-l4-full.

**Evidence emitted**
- `mc_cfg`: yes
- `mc_snap`: yes
- `plasticity`: yes
- `asset bundle`: yes
- `trace run`: optional
