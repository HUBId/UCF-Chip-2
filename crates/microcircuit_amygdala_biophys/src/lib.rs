#![forbid(unsafe_code)]

use biophys_core::{
    clamp_i32, LifParams, LifState, ModChannel, NeuronId, PopCode, StpParams, StpState, STP_SCALE,
};
use biophys_runtime::BiophysRuntime;
use dbm_core::{IntegrityState, LevelClass, ReasonSet, ThreatVector};
use microcircuit_amygdala_stub::{AmyInput, AmyOutput};
use microcircuit_core::{digest_meta, CircuitConfig, MicrocircuitBackend};

const POOL_COUNT: usize = 4;
const POOL_SIZE: usize = 6;
const EXCITATORY_COUNT: usize = POOL_COUNT * POOL_SIZE;
const INHIBITORY_COUNT: usize = 2;
const NEURON_COUNT: usize = EXCITATORY_COUNT + INHIBITORY_COUNT;

const IDX_INTEGRITY: usize = 0;
const IDX_EXFIL: usize = 1;
const IDX_PROBING: usize = 2;
const IDX_TOOL: usize = 3;

const INHIBITORY_START: usize = EXCITATORY_COUNT;

const DT_MS: u16 = 1;
const MAX_SPIKES_PER_STEP: usize = 24;
const MAX_EVENTS_PER_STEP: usize = 512;

const CURRENT_STRONG: i32 = 200;

const EXCITATORY_WEIGHT: i32 = 7;
const CROSS_EXCITATORY_WEIGHT: i32 = 3;
const EXC_TO_INHIB_WEIGHT: i32 = 8;
const INHIBIT_WEIGHT: i32 = -10;

const ACCUMULATOR_MAX: i32 = 100;
const ACCUMULATOR_DECAY: i32 = 3;
const ACCUMULATOR_GAIN: i32 = 10;
const LATCH_HIGH: i32 = 70;
const LATCH_LOW: i32 = 40;
const LATCH_STEPS_MAX: u8 = 10;

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

#[derive(Debug, Clone)]
struct AmyBiophysState {
    pool_acc: [i32; POOL_COUNT],
    latch_steps: [u8; POOL_COUNT],
    step_count: u64,
    last_spike_count_exc: usize,
}

impl Default for AmyBiophysState {
    fn default() -> Self {
        Self {
            pool_acc: [3; POOL_COUNT],
            latch_steps: [0; POOL_COUNT],
            step_count: 0,
            last_spike_count_exc: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AmygdalaBiophysMicrocircuit {
    _config: CircuitConfig,
    runtime: BiophysRuntime,
    state: AmyBiophysState,
}

impl AmygdalaBiophysMicrocircuit {
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
            state: AmyBiophysState::default(),
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

    fn tool_anomaly_present(input: &AmyInput) -> bool {
        input.tool_anomaly_present
            || input.cerebellum_tool_anomaly_present.unwrap_or(false)
            || input
                .tool_anomalies
                .iter()
                .any(|(_, level)| matches!(level, LevelClass::High))
    }

    fn build_inputs(input: &AmyInput) -> [i32; NEURON_COUNT] {
        let integrity_drive = matches!(
            input.integrity,
            IntegrityState::Fail | IntegrityState::Degraded
        ) || input.replay_mismatch_present
            || input.receipt_invalid_medium >= 1;
        let exfil_drive =
            input.dlp_secret_present || input.dlp_obfuscation_present || input.dlp_stegano_present;
        let probing_drive = input.policy_pressure == LevelClass::High || input.deny_storm_present;
        let tool_drive = Self::tool_anomaly_present(input);

        let mut currents = [0i32; NEURON_COUNT];
        let drives = [integrity_drive, exfil_drive, probing_drive, tool_drive];
        for (pool, drive) in drives.iter().enumerate() {
            if !drive {
                continue;
            }
            let (start, end) = Self::pool_bounds(pool);
            for current in currents[start..end].iter_mut() {
                *current = current.saturating_add(CURRENT_STRONG);
            }
        }
        currents
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

    fn update_latches(&mut self) {
        for (acc, latch) in self
            .state
            .pool_acc
            .iter()
            .copied()
            .zip(self.state.latch_steps.iter_mut())
        {
            if acc >= LATCH_HIGH {
                *latch = LATCH_STEPS_MAX;
            } else if acc < LATCH_LOW {
                *latch = latch.saturating_sub(1);
            }
        }
    }

    fn pool_active(&self, pool: usize) -> bool {
        self.state.pool_acc[pool] >= 60 || self.state.latch_steps[pool] > 0
    }

    fn build_vectors(
        integrity_active: bool,
        exfil_active: bool,
        probing_active: bool,
        tool_active: bool,
    ) -> Vec<ThreatVector> {
        let mut vectors = Vec::new();
        for vector in [
            ThreatVector::IntegrityCompromise,
            ThreatVector::Exfil,
            ThreatVector::Probing,
            ThreatVector::ToolSideEffects,
        ] {
            let active = match vector {
                ThreatVector::IntegrityCompromise => integrity_active,
                ThreatVector::Exfil => exfil_active,
                ThreatVector::Probing => probing_active,
                ThreatVector::ToolSideEffects => tool_active,
                _ => false,
            };

            if active {
                vectors.push(vector);
            }
        }

        if vectors.len() > 8 {
            vectors.truncate(8);
        }

        vectors
    }
}

impl MicrocircuitBackend<AmyInput, AmyOutput> for AmygdalaBiophysMicrocircuit {
    fn step(&mut self, input: &AmyInput, _now_ms: u64) -> AmyOutput {
        self.state.step_count = self.state.step_count.saturating_add(1);
        let currents = Self::build_inputs(input);
        self.runtime.set_modulators(input.modulators);
        let spikes = self.runtime.step(&currents);
        self.update_pool_accumulators(&spikes);
        self.update_latches();

        let integrity_active = self.pool_active(IDX_INTEGRITY);
        let exfil_active = self.pool_active(IDX_EXFIL);
        let probing_active = self.pool_active(IDX_PROBING);
        let tool_active = self.pool_active(IDX_TOOL);

        let threat = if integrity_active || exfil_active {
            LevelClass::High
        } else if tool_active || probing_active {
            LevelClass::Med
        } else {
            LevelClass::Low
        };

        let vectors =
            Self::build_vectors(integrity_active, exfil_active, probing_active, tool_active);

        let mut reason_codes = ReasonSet::default();
        if integrity_active {
            reason_codes.insert("RC.TH.INTEGRITY_COMPROMISE");
        }
        if exfil_active {
            reason_codes.insert("RC.TH.EXFIL.HIGH_CONFIDENCE");
        }
        if probing_active {
            reason_codes.insert("RC.TH.POLICY_PROBING");
        }
        if tool_active {
            reason_codes.insert("RC.TH.TOOL_SIDE_EFFECTS");
        }

        AmyOutput {
            threat,
            vectors,
            reason_codes,
        }
    }

    fn snapshot_digest(&self) -> [u8; 32] {
        digest_meta("UCF:BIO:AMY:SNAP", &self.runtime.snapshot_digest())
    }

    fn config_digest(&self) -> [u8; 32] {
        digest_meta("UCF:BIO:AMY:CFG", &self.runtime.config_digest())
    }
}

fn build_edges() -> (Vec<biophys_core::SynapseEdge>, Vec<StpParams>) {
    let mut edges = Vec::new();
    let mut params = Vec::new();

    for pool in 0..POOL_COUNT {
        let (start, end) = AmygdalaBiophysMicrocircuit::pool_bounds(pool);
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

    let (integrity_start, integrity_end) = AmygdalaBiophysMicrocircuit::pool_bounds(IDX_INTEGRITY);
    let (exfil_start, exfil_end) = AmygdalaBiophysMicrocircuit::pool_bounds(IDX_EXFIL);
    for pre in integrity_start..integrity_end {
        for post in exfil_start..exfil_end {
            edges.push(biophys_core::SynapseEdge {
                pre: NeuronId(pre as u32),
                post: NeuronId(post as u32),
                weight_base: CROSS_EXCITATORY_WEIGHT,
                weight_effective: CROSS_EXCITATORY_WEIGHT,
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
    for pre in exfil_start..exfil_end {
        for post in integrity_start..integrity_end {
            edges.push(biophys_core::SynapseEdge {
                pre: NeuronId(pre as u32),
                post: NeuronId(post as u32),
                weight_base: CROSS_EXCITATORY_WEIGHT,
                weight_effective: CROSS_EXCITATORY_WEIGHT,
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

    for pre in 0..EXCITATORY_COUNT {
        for inhibitory in INHIBITORY_START..NEURON_COUNT {
            edges.push(biophys_core::SynapseEdge {
                pre: NeuronId(pre as u32),
                post: NeuronId(inhibitory as u32),
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

    for post in 0..EXCITATORY_COUNT {
        for inhibitory in INHIBITORY_START..NEURON_COUNT {
            edges.push(biophys_core::SynapseEdge {
                pre: NeuronId(inhibitory as u32),
                post: NeuronId(post as u32),
                weight_base: INHIBIT_WEIGHT,
                weight_effective: INHIBIT_WEIGHT,
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
    use dbm_core::ThreatVector;

    fn base_input() -> AmyInput {
        AmyInput {
            integrity: IntegrityState::Ok,
            replay_mismatch_present: false,
            dlp_secret_present: false,
            dlp_obfuscation_present: false,
            dlp_stegano_present: false,
            dlp_critical_count_med: 0,
            receipt_invalid_medium: 0,
            policy_pressure: LevelClass::Low,
            deny_storm_present: false,
            sealed: None,
            tool_anomaly_present: false,
            cerebellum_tool_anomaly_present: None,
            tool_anomalies: Vec::new(),
            divergence: LevelClass::Low,
            modulators: ModulatorField::default(),
        }
    }

    fn threat_rank(level: LevelClass) -> u8 {
        match level {
            LevelClass::Low => 0,
            LevelClass::Med => 1,
            LevelClass::High => 2,
        }
    }

    #[test]
    fn deterministic_sequence_repeatable() {
        let sequence = vec![
            AmyInput {
                dlp_secret_present: true,
                ..base_input()
            },
            AmyInput {
                policy_pressure: LevelClass::High,
                ..base_input()
            },
            AmyInput {
                integrity: IntegrityState::Fail,
                ..base_input()
            },
            base_input(),
        ];

        let run_sequence = |inputs: &[AmyInput]| -> Vec<AmyOutput> {
            let mut circuit = AmygdalaBiophysMicrocircuit::new(CircuitConfig::default());
            inputs.iter().map(|input| circuit.step(input, 0)).collect()
        };

        let outputs_a = run_sequence(&sequence);
        let outputs_b = run_sequence(&sequence);

        assert_eq!(outputs_a, outputs_b);
    }

    #[test]
    fn dlp_secret_activates_exfil_quickly() {
        let mut circuit = AmygdalaBiophysMicrocircuit::new(CircuitConfig::default());
        let input = AmyInput {
            dlp_secret_present: true,
            ..base_input()
        };

        let first = circuit.step(&input, 0);
        let second = circuit.step(&input, 0);

        let active = first.vectors.contains(&ThreatVector::Exfil)
            || second.vectors.contains(&ThreatVector::Exfil);
        assert!(active);
        assert_eq!(second.threat, LevelClass::High);
    }

    #[test]
    fn integrity_fail_is_immediate_high() {
        let mut circuit = AmygdalaBiophysMicrocircuit::new(CircuitConfig::default());
        let output = circuit.step(
            &AmyInput {
                integrity: IntegrityState::Fail,
                ..base_input()
            },
            0,
        );

        assert_eq!(output.threat, LevelClass::High);
        assert!(output.vectors.contains(&ThreatVector::IntegrityCompromise));
    }

    #[test]
    fn micro_is_not_less_than_rules_on_critical_inputs() {
        let mut rules = microcircuit_amygdala_stub::AmyRules::new();
        let mut circuit = AmygdalaBiophysMicrocircuit::new(CircuitConfig::default());
        let input = AmyInput {
            dlp_secret_present: true,
            integrity: IntegrityState::Fail,
            ..base_input()
        };

        let rules_output = rules.tick(&input);
        let micro_output = circuit.step(&input, 0);

        assert!(threat_rank(micro_output.threat) >= threat_rank(rules_output.threat));
    }

    #[test]
    fn accumulators_and_latches_bounded() {
        let mut circuit = AmygdalaBiophysMicrocircuit::new(CircuitConfig::default());
        let input = AmyInput {
            integrity: IntegrityState::Fail,
            dlp_secret_present: true,
            policy_pressure: LevelClass::High,
            deny_storm_present: true,
            tool_anomaly_present: true,
            ..base_input()
        };

        for _ in 0..20 {
            circuit.step(&input, 0);
        }

        assert!(circuit
            .state
            .pool_acc
            .iter()
            .all(|&acc| (0..=ACCUMULATOR_MAX).contains(&acc)));
        assert!(circuit
            .state
            .latch_steps
            .iter()
            .all(|&latch| latch <= LATCH_STEPS_MAX));
        assert!(circuit.state.last_spike_count_exc <= MAX_SPIKES_PER_STEP);
    }
}
