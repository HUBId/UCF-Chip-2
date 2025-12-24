#![forbid(unsafe_code)]

use biophys_core::{
    clamp_i32, LifParams, LifState, ModChannel, NeuronId, PopCode, StpParams, StpState, STP_SCALE,
};
use biophys_runtime::BiophysRuntime;
use dbm_core::{
    DbmModule, DwmMode, IntegrityState, LevelClass, ReasonSet, SalienceItem, SalienceSource,
};
use microcircuit_core::{digest_meta, CircuitConfig, MicrocircuitBackend};
use microcircuit_sn_stub::{SnInput, SnOutput};

const POOL_COUNT: usize = 4;
const POOL_SIZE: usize = 5;
const EXCITATORY_COUNT: usize = POOL_COUNT * POOL_SIZE;
const INHIBITORY_COUNT: usize = 5;
const NEURON_COUNT: usize = EXCITATORY_COUNT + INHIBITORY_COUNT;

const GLOBAL_INHIBITORY_IDX: usize = EXCITATORY_COUNT + 4;

const DT_MS: u16 = 1;
const MAX_SPIKES_PER_STEP: usize = 20;
const MAX_EVENTS_PER_STEP: usize = 512;

const BASELINE_CURRENT: i32 = 2;
const DRIVE_EXEC_PLAN: i32 = 8;
const DRIVE_HIGH: i32 = 20;
const DRIVE_REPLAY: i32 = 6;

const EXCITATORY_WEIGHT: i32 = 6;
const INHIBIT_GLOBAL_WEIGHT: i32 = -9;
const INHIBIT_LOCAL_WEIGHT: i32 = -6;
const EXC_TO_GLOBAL_WEIGHT: i32 = 8;
const EXC_TO_LOCAL_WEIGHT: i32 = 6;

const ACCUMULATOR_MAX: i32 = 100;
const ACCUMULATOR_DECAY: i32 = 3;
const ACCUMULATOR_GAIN: i32 = 6;

const HYSTERESIS_TICKS: u8 = 3;

const STP_EXCITATORY: StpParams = StpParams {
    u: 350,
    tau_rec_steps: 5,
    tau_fac_steps: 3,
    mod_channel: None,
};

const STP_NONE: StpParams = StpParams {
    u: STP_SCALE,
    tau_rec_steps: 0,
    tau_fac_steps: 0,
    mod_channel: None,
};

const IDX_EXEC_PLAN: usize = 0;
const IDX_SIMULATE: usize = 1;
const IDX_STABILIZE: usize = 2;
const IDX_REPORT: usize = 3;

#[derive(Debug, Clone)]
struct SnBiophysState {
    pool_acc: [i32; POOL_COUNT],
    winner: usize,
    hysteresis_count: u8,
    pending_winner: Option<usize>,
    step_count: u64,
    last_spike_count_exc: usize,
}

impl Default for SnBiophysState {
    fn default() -> Self {
        Self {
            pool_acc: [0; POOL_COUNT],
            winner: IDX_EXEC_PLAN,
            hysteresis_count: 0,
            pending_winner: None,
            step_count: 0,
            last_spike_count_exc: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SnBiophysMicrocircuit {
    _config: CircuitConfig,
    runtime: BiophysRuntime,
    state: SnBiophysState,
}

impl SnBiophysMicrocircuit {
    pub fn new(config: CircuitConfig) -> Self {
        let params = vec![
            LifParams {
                tau_ms: 10,
                v_rest: 0,
                v_reset: 0,
                v_threshold: 20,
            };
            NEURON_COUNT
        ];
        let states = vec![
            LifState {
                v: 0,
                refractory_steps: 0,
            };
            NEURON_COUNT
        ];
        let (edges, stp_params) = build_edges();
        let runtime = BiophysRuntime::new_with_synapses(
            params,
            states,
            DT_MS,
            MAX_SPIKES_PER_STEP,
            edges,
            stp_params,
            MAX_EVENTS_PER_STEP,
        );
        Self {
            _config: config,
            runtime,
            state: SnBiophysState::default(),
        }
    }

    fn pool_bounds(pool: usize) -> (usize, usize) {
        let start = pool * POOL_SIZE;
        (start, start + POOL_SIZE)
    }

    fn pool_index_for_exc(neuron_idx: usize) -> Option<usize> {
        if neuron_idx >= EXCITATORY_COUNT {
            return None;
        }
        Some(neuron_idx / POOL_SIZE)
    }

    fn severity_index(mode: DwmMode) -> u8 {
        match mode {
            DwmMode::ExecPlan => 0,
            DwmMode::Simulate => 1,
            DwmMode::Stabilize => 2,
            DwmMode::Report => 3,
        }
    }

    fn rules_floor(input: &SnInput) -> DwmMode {
        if input.isv.integrity == IntegrityState::Fail {
            DwmMode::Report
        } else if input.isv.threat == LevelClass::High {
            DwmMode::Stabilize
        } else if input.isv.policy_pressure == LevelClass::High
            || input.isv.arousal == LevelClass::High
            || input.replay_hint
        {
            DwmMode::Simulate
        } else {
            DwmMode::ExecPlan
        }
    }

    fn mode_from_pool(pool: usize) -> DwmMode {
        match pool {
            IDX_REPORT => DwmMode::Report,
            IDX_STABILIZE => DwmMode::Stabilize,
            IDX_SIMULATE => DwmMode::Simulate,
            IDX_EXEC_PLAN => DwmMode::ExecPlan,
            _ => DwmMode::ExecPlan,
        }
    }

    fn rules_drives(input: &SnInput) -> ([i32; POOL_COUNT], Vec<SalienceSource>) {
        let mut currents = [BASELINE_CURRENT; POOL_COUNT];
        let mut salience = Vec::new();

        if input.isv.integrity == IntegrityState::Fail {
            currents[IDX_REPORT] = currents[IDX_REPORT].saturating_add(DRIVE_HIGH);
            salience.push(SalienceSource::Integrity);
        } else if input.isv.threat == LevelClass::High {
            currents[IDX_STABILIZE] = currents[IDX_STABILIZE].saturating_add(DRIVE_HIGH);
            salience.push(SalienceSource::Threat);
        } else if input.isv.policy_pressure == LevelClass::High
            || input.isv.arousal == LevelClass::High
        {
            currents[IDX_SIMULATE] = currents[IDX_SIMULATE].saturating_add(DRIVE_HIGH);
            if input.isv.policy_pressure == LevelClass::High {
                salience.push(SalienceSource::PolicyPressure);
            }
        } else {
            currents[IDX_EXEC_PLAN] = currents[IDX_EXEC_PLAN].saturating_add(DRIVE_EXEC_PLAN);
        }

        if input.replay_hint {
            currents[IDX_SIMULATE] = currents[IDX_SIMULATE].saturating_add(DRIVE_REPLAY);
            salience.push(SalienceSource::Replay);
        }

        (currents, salience)
    }

    fn build_inputs(input: &SnInput) -> ([i32; NEURON_COUNT], Vec<SalienceSource>) {
        let (drives, salience) = Self::rules_drives(input);
        let mut currents = [0i32; NEURON_COUNT];
        for (pool, drive) in drives.iter().enumerate().take(POOL_COUNT) {
            let (start, end) = Self::pool_bounds(pool);
            for current in currents[start..end].iter_mut() {
                *current = current.saturating_add(*drive);
            }
        }
        currents[GLOBAL_INHIBITORY_IDX] = 0;
        (currents, salience)
    }

    fn update_pool_accumulators(&mut self, spikes: &PopCode) -> [usize; POOL_COUNT] {
        let mut counts = [0usize; POOL_COUNT];
        for neuron in &spikes.spikes {
            if let Some(pool) = Self::pool_index_for_exc(neuron.0 as usize) {
                counts[pool] += 1;
            }
        }
        self.state.last_spike_count_exc = counts.iter().sum();
        for (acc, &count) in self.state.pool_acc.iter_mut().zip(counts.iter()) {
            let delta = (count as i32).saturating_mul(ACCUMULATOR_GAIN);
            *acc = clamp_i32(*acc + delta - ACCUMULATOR_DECAY, 0, ACCUMULATOR_MAX);
        }
        counts
    }

    fn choose_winner(&mut self) -> (usize, bool) {
        let max_value = self.state.pool_acc.iter().copied().max().unwrap_or(0);
        let mut best_pool = IDX_EXEC_PLAN;
        for pool in (0..POOL_COUNT).rev() {
            if self.state.pool_acc[pool] == max_value {
                best_pool = pool;
                break;
            }
        }

        let current = self.state.winner;
        let mut changed = false;
        if best_pool == current {
            self.state.hysteresis_count = 0;
            self.state.pending_winner = None;
        } else if Self::severity_index(Self::mode_from_pool(best_pool))
            > Self::severity_index(Self::mode_from_pool(current))
        {
            self.state.winner = best_pool;
            self.state.hysteresis_count = 0;
            self.state.pending_winner = None;
            changed = true;
        } else {
            if self.state.pending_winner == Some(best_pool) {
                self.state.hysteresis_count = self.state.hysteresis_count.saturating_add(1);
            } else {
                self.state.pending_winner = Some(best_pool);
                self.state.hysteresis_count = 1;
            }
            if self.state.hysteresis_count >= HYSTERESIS_TICKS {
                self.state.winner = best_pool;
                self.state.hysteresis_count = 0;
                self.state.pending_winner = None;
                changed = true;
            }
        }

        (self.state.winner, changed)
    }

    fn build_salience(input: &SnInput, sources: &[SalienceSource]) -> Vec<SalienceItem> {
        let mut items: Vec<SalienceItem> = sources
            .iter()
            .map(|source| {
                SalienceItem::new(
                    *source,
                    LevelClass::High,
                    input.isv.dominant_reason_codes.codes.clone(),
                )
            })
            .collect();

        items.sort_by(|a, b| (a.source as u8).cmp(&(b.source as u8)));
        if items.len() > 8 {
            items.truncate(8);
        }
        items
    }
}

impl MicrocircuitBackend<SnInput, SnOutput> for SnBiophysMicrocircuit {
    fn step(&mut self, input: &SnInput, _now_ms: u64) -> SnOutput {
        self.state.step_count = self.state.step_count.saturating_add(1);

        let (currents, salience_sources) = Self::build_inputs(input);
        self.runtime.set_modulators(input.modulators);
        let spikes = self.runtime.step(&currents);
        self.update_pool_accumulators(&spikes);
        let (winner_pool, mut switched) = self.choose_winner();
        let mut dwm = Self::mode_from_pool(winner_pool);
        let floor = Self::rules_floor(input);
        if Self::severity_index(dwm) < Self::severity_index(floor) {
            dwm = floor;
            let floor_pool = match floor {
                DwmMode::ExecPlan => IDX_EXEC_PLAN,
                DwmMode::Simulate => IDX_SIMULATE,
                DwmMode::Stabilize => IDX_STABILIZE,
                DwmMode::Report => IDX_REPORT,
            };
            if self.state.winner != floor_pool {
                self.state.winner = floor_pool;
                self.state.hysteresis_count = 0;
                self.state.pending_winner = None;
                switched = true;
            }
        }

        let mut reason_codes = ReasonSet::default();
        if switched {
            reason_codes.insert("RC.GV.DWM.SWITCHED");
        }
        reason_codes.insert(match dwm {
            DwmMode::Report => "RC.GV.DWM.REPORT",
            DwmMode::Stabilize => "RC.GV.DWM.STABILIZE",
            DwmMode::Simulate => "RC.GV.DWM.SIMULATE",
            DwmMode::ExecPlan => "RC.GV.DWM.EXEC_PLAN",
        });

        let salience_items = Self::build_salience(input, &salience_sources);

        SnOutput {
            dwm,
            salience_items,
            reason_codes,
        }
    }

    fn snapshot_digest(&self) -> [u8; 32] {
        digest_meta("UCF:BIO:SN:SNAP", &self.runtime.snapshot_digest())
    }

    fn config_digest(&self) -> [u8; 32] {
        digest_meta("UCF:BIO:SN:CFG", &self.runtime.config_digest())
    }
}

impl DbmModule for SnBiophysMicrocircuit {
    type Input = SnInput;
    type Output = SnOutput;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        self.step(input, 0)
    }
}

fn build_edges() -> (Vec<biophys_core::SynapseEdge>, Vec<StpParams>) {
    let mut edges = Vec::new();
    let mut params = Vec::new();

    for pool in 0..POOL_COUNT {
        let (start, end) = SnBiophysMicrocircuit::pool_bounds(pool);
        for pre in start..end {
            for post in start..end {
                if pre == post {
                    continue;
                }
                edges.push(biophys_core::SynapseEdge {
                    pre: NeuronId(pre as u32),
                    post: NeuronId(post as u32),
                    weight_base: EXCITATORY_WEIGHT,
                    weight_effective: EXCITATORY_WEIGHT,
                    delay_steps: 1,
                    mod_channel: ModChannel::None,
                    stp: StpState {
                        x: STP_SCALE,
                        u: STP_EXCITATORY.u,
                    },
                });
                params.push(STP_EXCITATORY);
            }
        }
    }

    for pre in 0..EXCITATORY_COUNT {
        edges.push(biophys_core::SynapseEdge {
            pre: NeuronId(pre as u32),
            post: NeuronId(GLOBAL_INHIBITORY_IDX as u32),
            weight_base: EXC_TO_GLOBAL_WEIGHT,
            weight_effective: EXC_TO_GLOBAL_WEIGHT,
            delay_steps: 1,
            mod_channel: ModChannel::None,
            stp: StpState {
                x: STP_SCALE,
                u: STP_NONE.u,
            },
        });
        params.push(STP_NONE);
    }

    for pool in 0..POOL_COUNT {
        let inhibitory_idx = EXCITATORY_COUNT + pool;
        let (start, end) = SnBiophysMicrocircuit::pool_bounds(pool);
        for pre in start..end {
            edges.push(biophys_core::SynapseEdge {
                pre: NeuronId(pre as u32),
                post: NeuronId(inhibitory_idx as u32),
                weight_base: EXC_TO_LOCAL_WEIGHT,
                weight_effective: EXC_TO_LOCAL_WEIGHT,
                delay_steps: 1,
                mod_channel: ModChannel::None,
                stp: StpState {
                    x: STP_SCALE,
                    u: STP_NONE.u,
                },
            });
            params.push(STP_NONE);
        }
    }

    for post in 0..EXCITATORY_COUNT {
        edges.push(biophys_core::SynapseEdge {
            pre: NeuronId(GLOBAL_INHIBITORY_IDX as u32),
            post: NeuronId(post as u32),
            weight_base: INHIBIT_GLOBAL_WEIGHT,
            weight_effective: INHIBIT_GLOBAL_WEIGHT,
            delay_steps: 1,
            mod_channel: ModChannel::None,
            stp: StpState {
                x: STP_SCALE,
                u: STP_NONE.u,
            },
        });
        params.push(STP_NONE);
    }

    for pool in 0..POOL_COUNT {
        let inhibitory_idx = EXCITATORY_COUNT + pool;
        let (start, end) = SnBiophysMicrocircuit::pool_bounds(pool);
        for post in start..end {
            edges.push(biophys_core::SynapseEdge {
                pre: NeuronId(inhibitory_idx as u32),
                post: NeuronId(post as u32),
                weight_base: INHIBIT_LOCAL_WEIGHT,
                weight_effective: INHIBIT_LOCAL_WEIGHT,
                delay_steps: 1,
                mod_channel: ModChannel::None,
                stp: StpState {
                    x: STP_SCALE,
                    u: STP_NONE.u,
                },
            });
            params.push(STP_NONE);
        }
    }

    (edges, params)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn severity(mode: DwmMode) -> u8 {
        match mode {
            DwmMode::ExecPlan => 0,
            DwmMode::Simulate => 1,
            DwmMode::Stabilize => 2,
            DwmMode::Report => 3,
        }
    }

    #[test]
    fn determinism_sequence_and_digests() {
        let inputs = (0..50)
            .map(|idx| SnInput {
                isv: dbm_core::IsvSnapshot {
                    integrity: if idx % 11 == 0 {
                        IntegrityState::Fail
                    } else {
                        IntegrityState::Ok
                    },
                    threat: if idx % 7 == 0 {
                        LevelClass::High
                    } else {
                        LevelClass::Low
                    },
                    policy_pressure: if idx % 5 == 0 {
                        LevelClass::High
                    } else {
                        LevelClass::Low
                    },
                    arousal: if idx % 9 == 0 {
                        LevelClass::High
                    } else {
                        LevelClass::Low
                    },
                    ..dbm_core::IsvSnapshot::default()
                },
                replay_hint: idx % 4 == 0,
                ..Default::default()
            })
            .collect::<Vec<_>>();

        let run_sequence = |inputs: &[SnInput]| -> Vec<(DwmMode, [u8; 32])> {
            let mut circuit = SnBiophysMicrocircuit::new(CircuitConfig::default());
            inputs
                .iter()
                .map(|input| {
                    let output = circuit.step(input, 0);
                    let digest = circuit.snapshot_digest();
                    (output.dwm, digest)
                })
                .collect()
        };

        let outputs_a = run_sequence(&inputs);
        let outputs_b = run_sequence(&inputs);

        assert_eq!(outputs_a, outputs_b);
        for (_, digest) in outputs_a {
            assert_ne!(digest, [0u8; 32]);
        }
    }

    #[test]
    fn critical_invariants_hold() {
        let mut circuit = SnBiophysMicrocircuit::new(CircuitConfig::default());
        let integrity = SnInput {
            isv: dbm_core::IsvSnapshot {
                integrity: IntegrityState::Fail,
                ..dbm_core::IsvSnapshot::default()
            },
            ..Default::default()
        };
        let threat = SnInput {
            isv: dbm_core::IsvSnapshot {
                threat: LevelClass::High,
                ..dbm_core::IsvSnapshot::default()
            },
            ..Default::default()
        };
        let policy = SnInput {
            isv: dbm_core::IsvSnapshot {
                policy_pressure: LevelClass::High,
                ..dbm_core::IsvSnapshot::default()
            },
            ..Default::default()
        };

        let integrity_out = circuit.step(&integrity, 0);
        assert_eq!(integrity_out.dwm, DwmMode::Report);

        let threat_out = circuit.step(&threat, 0);
        assert!(severity(threat_out.dwm) >= severity(DwmMode::Stabilize));

        let policy_out = circuit.step(&policy, 0);
        assert!(severity(policy_out.dwm) >= severity(DwmMode::Simulate));
    }

    #[test]
    fn hysteresis_requires_three_wins() {
        let mut circuit = SnBiophysMicrocircuit::new(CircuitConfig::default());
        circuit.state.winner = IDX_REPORT;
        circuit.state.pool_acc = [30, 20, 10, 5];

        let (first, _) = circuit.choose_winner();
        let (second, _) = circuit.choose_winner();
        assert_eq!(first, IDX_REPORT);
        assert_eq!(second, IDX_REPORT);

        let (third, _) = circuit.choose_winner();
        assert_eq!(third, IDX_EXEC_PLAN);
    }

    #[test]
    fn spike_counts_bounded() {
        let mut circuit = SnBiophysMicrocircuit::new(CircuitConfig::default());
        let input = SnInput {
            isv: dbm_core::IsvSnapshot {
                integrity: IntegrityState::Fail,
                threat: LevelClass::High,
                policy_pressure: LevelClass::High,
                arousal: LevelClass::High,
                ..dbm_core::IsvSnapshot::default()
            },
            replay_hint: true,
            ..Default::default()
        };

        circuit.step(&input, 0);
        assert!(circuit.state.last_spike_count_exc <= MAX_SPIKES_PER_STEP);
        assert!(circuit
            .state
            .pool_acc
            .iter()
            .all(|&acc| acc <= ACCUMULATOR_MAX));
    }
}
