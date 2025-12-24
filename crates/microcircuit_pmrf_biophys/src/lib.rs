#![forbid(unsafe_code)]

use biophys_core::{
    LifParams, LifState, ModChannel, NeuronId, PopCode, StpParams, StpState, STP_SCALE,
};
use biophys_runtime::BiophysRuntime;
use dbm_core::{LevelClass, ReasonSet};
use microcircuit_core::{digest_meta, CircuitConfig, MicrocircuitBackend};
use microcircuit_pmrf_stub::{PmrfInput, PmrfOutput, SequenceMode};

const POOL_COUNT: usize = 3;
const POOL_SIZE: usize = 4;
const EXCITATORY_COUNT: usize = POOL_COUNT * POOL_SIZE;
const INHIBITORY_COUNT: usize = 2;
const NEURON_COUNT: usize = EXCITATORY_COUNT + INHIBITORY_COUNT;

const NORMAL_POOL_START: usize = 0;
const SLOW_POOL_START: usize = 4;
const SPLIT_POOL_START: usize = 8;
const INHIBITORY_START: usize = 12;

const DT_MS: u16 = 1;
const MAX_SPIKES_PER_STEP: usize = 14;
const MAX_EVENTS_PER_STEP: usize = 256;

const POOL_ACC_MAX: i32 = 100;
const POOL_ACC_DECAY: i32 = 3;
const POOL_ACC_SPIKE_GAIN: i32 = 15;

const HYSTERESIS_TICKS: u8 = 3;

const CURRENT_BASELINE: i32 = 6;
const CURRENT_MEDIUM: i32 = 14;
const CURRENT_STRONG: i32 = 22;
const CURRENT_VERY_STRONG: i32 = 32;

const EXCITATORY_WEIGHT: i32 = 7;
const INHIBITORY_WEIGHT: i32 = -10;
const EXC_TO_INHIB_WEIGHT: i32 = 8;
const CROSS_INHIB_STRONG: i32 = -14;
const CROSS_INHIB_MODERATE: i32 = -9;

const STP_SPLIT: StpParams = StpParams {
    u: 320,
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

const IDX_NORMAL: usize = 0;
const IDX_SLOW: usize = 1;
const IDX_SPLIT: usize = 2;

#[derive(Debug, Clone)]
struct PmrfBiophysState {
    pool_acc: [i32; POOL_COUNT],
    winner: usize,
    pending_winner: Option<usize>,
    pending_count: u8,
    hold_ticks: u8,
    step_count: u64,
}

impl Default for PmrfBiophysState {
    fn default() -> Self {
        Self {
            pool_acc: [0; POOL_COUNT],
            winner: IDX_NORMAL,
            pending_winner: None,
            pending_count: 0,
            hold_ticks: 0,
            step_count: 0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct PmrfDrives {
    pool_currents: [i32; POOL_COUNT],
    divergence_high: bool,
}

#[derive(Debug, Clone)]
pub struct PmrfBiophysMicrocircuit {
    _config: CircuitConfig,
    runtime: BiophysRuntime,
    state: PmrfBiophysState,
}

impl PmrfBiophysMicrocircuit {
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
            state: PmrfBiophysState::default(),
        }
    }

    fn encode_drives(input: &PmrfInput) -> PmrfDrives {
        let mut split_drive: i32 = 0;
        let mut slow_drive: i32 = 0;
        let mut normal_drive: i32 = 0;

        let divergence_high = input.divergence == LevelClass::High;
        match input.divergence {
            LevelClass::High => split_drive = split_drive.saturating_add(CURRENT_VERY_STRONG),
            LevelClass::Med => split_drive = split_drive.saturating_add(CURRENT_MEDIUM),
            LevelClass::Low => {}
        }

        if input.hold_active {
            slow_drive = slow_drive.saturating_add(CURRENT_STRONG);
        }
        if input.policy_pressure == LevelClass::High {
            slow_drive = slow_drive.saturating_add(CURRENT_MEDIUM);
        }
        if input.budget_stress == LevelClass::High {
            slow_drive = slow_drive.saturating_add(CURRENT_MEDIUM);
        }

        if split_drive == 0 && slow_drive == 0 {
            normal_drive = normal_drive.saturating_add(CURRENT_BASELINE);
        }

        PmrfDrives {
            pool_currents: [normal_drive, slow_drive, split_drive],
            divergence_high,
        }
    }

    fn apply_pool_currents(currents: &mut [i32; NEURON_COUNT], start: usize, drive: i32) {
        for value in currents.iter_mut().skip(start).take(POOL_SIZE) {
            *value = value.saturating_add(drive);
        }
    }

    fn pool_for_neuron(idx: usize) -> Option<usize> {
        if idx < SLOW_POOL_START {
            Some(IDX_NORMAL)
        } else if idx < SPLIT_POOL_START {
            Some(IDX_SLOW)
        } else if idx < EXCITATORY_COUNT {
            Some(IDX_SPLIT)
        } else {
            None
        }
    }

    fn update_pool_acc(current: i32, spikes: usize) -> i32 {
        let delta = spikes as i32 * POOL_ACC_SPIKE_GAIN - POOL_ACC_DECAY;
        (current + delta).clamp(0, POOL_ACC_MAX)
    }

    fn select_winner(scores: &[i32; POOL_COUNT]) -> usize {
        let max_score = scores.iter().copied().max().unwrap_or(0);
        let order = [IDX_SPLIT, IDX_SLOW, IDX_NORMAL];
        for candidate in order {
            if scores[candidate] == max_score {
                return candidate;
            }
        }
        IDX_NORMAL
    }

    fn severity(pool: usize) -> u8 {
        match pool {
            IDX_SPLIT => 2,
            IDX_SLOW => 1,
            _ => 0,
        }
    }

    fn resolve_winner(&mut self, candidate: usize) -> usize {
        let current = self.state.winner;
        if candidate == current {
            self.state.pending_winner = None;
            self.state.pending_count = 0;
            return current;
        }

        let candidate_severity = Self::severity(candidate);
        let current_severity = Self::severity(current);

        if candidate_severity > current_severity {
            self.state.winner = candidate;
            self.state.pending_winner = None;
            self.state.pending_count = 0;
            return candidate;
        }

        if self.state.pending_winner == Some(candidate) {
            self.state.pending_count = self.state.pending_count.saturating_add(1);
        } else {
            self.state.pending_winner = Some(candidate);
            self.state.pending_count = 1;
        }

        if self.state.pending_count >= HYSTERESIS_TICKS {
            self.state.winner = candidate;
            self.state.pending_winner = None;
            self.state.pending_count = 0;
            return candidate;
        }

        current
    }
}

impl MicrocircuitBackend<PmrfInput, PmrfOutput> for PmrfBiophysMicrocircuit {
    fn step(&mut self, input: &PmrfInput, _now_ms: u64) -> PmrfOutput {
        self.state.step_count = self.state.step_count.saturating_add(1);

        let drives = Self::encode_drives(input);

        let mut currents = [0i32; NEURON_COUNT];
        Self::apply_pool_currents(
            &mut currents,
            NORMAL_POOL_START,
            drives.pool_currents[IDX_NORMAL],
        );
        Self::apply_pool_currents(
            &mut currents,
            SLOW_POOL_START,
            drives.pool_currents[IDX_SLOW],
        );
        Self::apply_pool_currents(
            &mut currents,
            SPLIT_POOL_START,
            drives.pool_currents[IDX_SPLIT],
        );

        self.runtime.set_modulators(input.modulators);
        let PopCode { spikes } = self.runtime.step(&currents);
        let mut pool_spikes = [0usize; POOL_COUNT];
        for spike in &spikes {
            if let Some(pool_idx) = Self::pool_for_neuron(spike.0 as usize) {
                pool_spikes[pool_idx] = pool_spikes[pool_idx].saturating_add(1);
            }
        }

        for (idx, acc) in self.state.pool_acc.iter_mut().enumerate() {
            *acc = Self::update_pool_acc(*acc, pool_spikes[idx]);
        }

        if drives.divergence_high {
            self.state.pool_acc[IDX_SPLIT] = POOL_ACC_MAX;
        }

        let candidate = Self::select_winner(&self.state.pool_acc);
        let winner = self.resolve_winner(candidate);

        if input.hold_active {
            self.state.hold_ticks = self.state.hold_ticks.saturating_add(1).min(4);
        } else {
            self.state.hold_ticks = 0;
        }

        let sequence_mode = match winner {
            IDX_SPLIT => SequenceMode::SplitRequired,
            IDX_SLOW => SequenceMode::Slow,
            _ => SequenceMode::Normal,
        };
        let chain_tightening = sequence_mode != SequenceMode::Normal;
        let checkpoint_required =
            sequence_mode == SequenceMode::SplitRequired || self.state.hold_ticks >= 2;

        let mut reason_codes = ReasonSet::default();
        match sequence_mode {
            SequenceMode::SplitRequired => {
                reason_codes.insert("RC.GV.SEQUENCE.SPLIT_REQUIRED");
            }
            SequenceMode::Slow => {
                reason_codes.insert("RC.GV.SEQUENCE.SLOW_DOWN");
            }
            SequenceMode::Normal => {}
        }

        PmrfOutput {
            sequence_mode,
            chain_tightening,
            checkpoint_required,
            reason_codes,
        }
    }

    fn snapshot_digest(&self) -> [u8; 32] {
        let digest = self.runtime.snapshot_digest();
        digest_meta("UCF:BIO:PMRF:SNAP", &digest)
    }

    fn config_digest(&self) -> [u8; 32] {
        let digest = self.runtime.config_digest();
        digest_meta("UCF:BIO:PMRF:CFG", &digest)
    }
}

fn build_edges() -> (Vec<biophys_core::SynapseEdge>, Vec<StpParams>) {
    let mut edges = Vec::new();
    let mut params = Vec::new();

    let pools = [NORMAL_POOL_START, SLOW_POOL_START, SPLIT_POOL_START];
    for (pool_idx, &start) in pools.iter().enumerate() {
        let stp = if pool_idx == IDX_SPLIT {
            STP_SPLIT
        } else {
            STP_NONE
        };
        for pre in start..start + POOL_SIZE {
            for post in start..start + POOL_SIZE {
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
                        u: stp.u,
                    },
                });
                params.push(stp);
            }
        }
    }

    for pre in 0..EXCITATORY_COUNT {
        for post in INHIBITORY_START..NEURON_COUNT {
            edges.push(biophys_core::SynapseEdge {
                pre: NeuronId(pre as u32),
                post: NeuronId(post as u32),
                weight_base: EXC_TO_INHIB_WEIGHT,
                weight_effective: EXC_TO_INHIB_WEIGHT,
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

    for pre in INHIBITORY_START..NEURON_COUNT {
        for post in 0..EXCITATORY_COUNT {
            edges.push(biophys_core::SynapseEdge {
                pre: NeuronId(pre as u32),
                post: NeuronId(post as u32),
                weight_base: INHIBITORY_WEIGHT,
                weight_effective: INHIBITORY_WEIGHT,
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

    for pre in SPLIT_POOL_START..SPLIT_POOL_START + POOL_SIZE {
        for post in NORMAL_POOL_START..NORMAL_POOL_START + POOL_SIZE {
            edges.push(biophys_core::SynapseEdge {
                pre: NeuronId(pre as u32),
                post: NeuronId(post as u32),
                weight_base: CROSS_INHIB_STRONG,
                weight_effective: CROSS_INHIB_STRONG,
                delay_steps: 1,
                mod_channel: ModChannel::None,
                stp: StpState {
                    x: STP_SCALE,
                    u: STP_NONE.u,
                },
            });
            params.push(STP_NONE);
        }
        for post in SLOW_POOL_START..SLOW_POOL_START + POOL_SIZE {
            edges.push(biophys_core::SynapseEdge {
                pre: NeuronId(pre as u32),
                post: NeuronId(post as u32),
                weight_base: CROSS_INHIB_STRONG,
                weight_effective: CROSS_INHIB_STRONG,
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

    for pre in SLOW_POOL_START..SLOW_POOL_START + POOL_SIZE {
        for post in NORMAL_POOL_START..NORMAL_POOL_START + POOL_SIZE {
            edges.push(biophys_core::SynapseEdge {
                pre: NeuronId(pre as u32),
                post: NeuronId(post as u32),
                weight_base: CROSS_INHIB_MODERATE,
                weight_effective: CROSS_INHIB_MODERATE,
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
    use biophys_core::ModulatorField;
    use dbm_core::DbmModule;
    use microcircuit_pmrf_stub::PmrfRules;

    fn base_input() -> PmrfInput {
        PmrfInput {
            divergence: LevelClass::Low,
            policy_pressure: LevelClass::Low,
            stability: LevelClass::Low,
            hold_active: false,
            budget_stress: LevelClass::Low,
            modulators: ModulatorField::default(),
        }
    }

    fn severity(mode: SequenceMode) -> u8 {
        match mode {
            SequenceMode::Normal => 0,
            SequenceMode::Slow => 1,
            SequenceMode::SplitRequired => 2,
        }
    }

    #[test]
    fn determinism_same_sequence_same_outputs() {
        let mut micro_a = PmrfBiophysMicrocircuit::new(CircuitConfig::default());
        let mut micro_b = PmrfBiophysMicrocircuit::new(CircuitConfig::default());

        let inputs = vec![
            base_input(),
            PmrfInput {
                divergence: LevelClass::Med,
                ..base_input()
            },
            PmrfInput {
                hold_active: true,
                ..base_input()
            },
        ];

        for input in inputs {
            let out_a = micro_a.step(&input, 0);
            let out_b = micro_b.step(&input, 0);
            assert_eq!(out_a, out_b);
        }
    }

    #[test]
    fn divergence_high_triggers_split_within_two_ticks() {
        let mut micro = PmrfBiophysMicrocircuit::new(CircuitConfig::default());
        let input = PmrfInput {
            divergence: LevelClass::High,
            ..base_input()
        };

        let first = micro.step(&input, 0);
        if first.sequence_mode == SequenceMode::SplitRequired {
            return;
        }

        let second = micro.step(&input, 0);
        assert_eq!(second.sequence_mode, SequenceMode::SplitRequired);
    }

    #[test]
    fn hold_active_drives_slow_or_split() {
        let mut micro = PmrfBiophysMicrocircuit::new(CircuitConfig::default());
        let output = micro.step(
            &PmrfInput {
                hold_active: true,
                ..base_input()
            },
            0,
        );

        assert!(output.sequence_mode != SequenceMode::Normal);
    }

    #[test]
    fn invariants_vs_rules_backend() {
        let cases = vec![
            PmrfInput {
                divergence: LevelClass::High,
                ..base_input()
            },
            PmrfInput {
                hold_active: true,
                ..base_input()
            },
            PmrfInput {
                policy_pressure: LevelClass::High,
                ..base_input()
            },
            PmrfInput {
                budget_stress: LevelClass::High,
                ..base_input()
            },
        ];

        for input in cases {
            let mut rules = PmrfRules::new();
            let mut micro = PmrfBiophysMicrocircuit::new(CircuitConfig::default());

            let rules_output = rules.tick(&input);
            let micro_output = micro.step(&input, 0);

            assert!(
                severity(micro_output.sequence_mode) >= severity(rules_output.sequence_mode),
                "micro {:?} < rules {:?}",
                micro_output.sequence_mode,
                rules_output.sequence_mode
            );
        }
    }
}
