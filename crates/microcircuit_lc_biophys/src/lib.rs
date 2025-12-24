#![forbid(unsafe_code)]

use biophys_core::{
    LifParams, LifState, ModChannel, NeuronId, PopCode, StpParams, StpState, STP_SCALE,
};
use biophys_runtime::BiophysRuntime;
use dbm_core::{DbmModule, IntegrityState, LevelClass, ReasonSet};
use microcircuit_core::{digest_meta, CircuitConfig, MicrocircuitBackend};
use microcircuit_lc_stub::{LcInput, LcOutput};

const NEURON_COUNT: usize = 10;
const EXCITATORY_COUNT: usize = 6;
const INHIBITORY_START: usize = 6;

const DT_MS: u16 = 1;
const MAX_SPIKES_PER_STEP: usize = 6;
const MAX_EVENTS_PER_STEP: usize = 256;
const SPIKE_COUNTER_SHORT_MAX: u8 = 4;

const CURRENT_STRONG: i32 = 30;
const CURRENT_CRITICAL: i32 = 20;
const CURRENT_SMALL: i32 = 8;

const INTEGRITY_TARGETS: [usize; 3] = [0, 1, 2];
const DLP_TARGETS: [usize; 3] = [0, 3, 5];
const DENY_TARGETS: [usize; 2] = [4, 5];

const EXCITATORY_WEIGHT: i32 = 8;
const INHIBITORY_WEIGHT: i32 = -10;

const EXCITATORY_STP: StpParams = StpParams {
    u: 250,
    tau_rec_steps: 5,
    tau_fac_steps: 3,
    mod_channel: None,
};

const NO_STP: StpParams = StpParams {
    u: STP_SCALE,
    tau_rec_steps: 0,
    tau_fac_steps: 0,
    mod_channel: None,
};

#[derive(Debug, Clone, Default)]
struct LcState {
    spike_counter_short: u8,
    last_spike_count_total: usize,
}

#[derive(Debug, Clone)]
pub struct LcMicrocircuit {
    _config: CircuitConfig,
    runtime: BiophysRuntime,
    state: LcState,
}

impl LcMicrocircuit {
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
            state: LcState::default(),
        }
    }

    fn tonic_current(level: LevelClass) -> i32 {
        match level {
            LevelClass::Low => 0,
            LevelClass::Med => 4,
            LevelClass::High => 8,
        }
    }

    fn max_level(a: LevelClass, b: LevelClass) -> LevelClass {
        if Self::severity(a) >= Self::severity(b) {
            a
        } else {
            b
        }
    }

    fn severity(level: LevelClass) -> u8 {
        match level {
            LevelClass::Low => 0,
            LevelClass::Med => 1,
            LevelClass::High => 2,
        }
    }

    fn rules_floor(input: &LcInput) -> LevelClass {
        if input.integrity != IntegrityState::Ok
            || input.receipt_invalid_count_short >= 1
            || input.dlp_critical_present_short
            || input.timeout_count_short >= 2
        {
            LevelClass::High
        } else if input.deny_count_short >= 2
            || input.receipt_missing_count_short >= 1
            || input.timeout_count_short == 1
        {
            LevelClass::Med
        } else {
            LevelClass::Low
        }
    }

    fn update_spike_counter(&mut self, spike_count_total: usize) {
        if spike_count_total > 0 {
            self.state.spike_counter_short = self
                .state
                .spike_counter_short
                .saturating_add(1)
                .min(SPIKE_COUNTER_SHORT_MAX);
        } else if self.state.spike_counter_short > 0 {
            self.state.spike_counter_short -= 1;
        }
    }

    fn build_inputs(input: &LcInput) -> [i32; NEURON_COUNT] {
        let mut currents = [0i32; NEURON_COUNT];
        let tonic = Self::tonic_current(input.arousal_floor);
        for current in currents.iter_mut().take(EXCITATORY_COUNT) {
            *current = current.saturating_add(tonic);
        }

        if input.integrity != IntegrityState::Ok {
            for &idx in &INTEGRITY_TARGETS {
                currents[idx] = currents[idx].saturating_add(CURRENT_STRONG);
            }
        }

        if input.receipt_invalid_count_short >= 1 {
            for current in currents.iter_mut().take(EXCITATORY_COUNT) {
                *current = current.saturating_add(CURRENT_STRONG);
            }
        }

        if input.dlp_critical_present_short {
            for &idx in &DLP_TARGETS {
                currents[idx] = currents[idx].saturating_add(CURRENT_CRITICAL);
            }
        }

        if input.deny_count_short >= 2 {
            for &idx in &DENY_TARGETS {
                currents[idx] = currents[idx].saturating_add(CURRENT_SMALL);
            }
        }

        currents
    }

    fn map_arousal(&self, spike_count_exc: usize, input: &LcInput) -> LevelClass {
        let base = if spike_count_exc >= 2 || self.state.spike_counter_short >= 2 {
            LevelClass::High
        } else if spike_count_exc == 1 || self.state.spike_counter_short == 1 {
            LevelClass::Med
        } else {
            LevelClass::Low
        };

        let rules_floor = Self::rules_floor(input);
        let mut arousal = Self::max_level(base, rules_floor);
        arousal = Self::max_level(arousal, input.arousal_floor);
        arousal
    }
}

impl MicrocircuitBackend<LcInput, LcOutput> for LcMicrocircuit {
    fn step(&mut self, input: &LcInput, _now_ms: u64) -> LcOutput {
        let currents = Self::build_inputs(input);
        self.runtime.set_modulators(input.modulators);
        let PopCode { spikes } = self.runtime.step(&currents);
        let spike_count_total = spikes.len();
        let spike_count_exc = spikes
            .iter()
            .filter(|id| (id.0 as usize) < EXCITATORY_COUNT)
            .count();

        self.state.last_spike_count_total = spike_count_total;
        self.update_spike_counter(spike_count_total);

        let arousal = self.map_arousal(spike_count_exc, input);
        let hint_simulate_first = arousal != LevelClass::Low;
        let hint_novelty_lock = arousal == LevelClass::High;

        let mut reason_codes = ReasonSet::default();
        if spike_count_total > 0 {
            reason_codes.insert("RC.GV.LC.SPIKE");
        }
        if input.integrity != IntegrityState::Ok {
            match input.integrity {
                IntegrityState::Degraded => reason_codes.insert("RC.RE.INTEGRITY.DEGRADED"),
                IntegrityState::Fail => reason_codes.insert("RC.RE.INTEGRITY.FAIL"),
                IntegrityState::Ok => {}
            }
        }
        if input.receipt_invalid_count_short >= 1 {
            reason_codes.insert("RC.GE.EXEC.DISPATCH_BLOCKED");
        }
        if input.dlp_critical_present_short {
            reason_codes.insert("RC.CD.DLP.SECRET_PATTERN");
        }

        LcOutput {
            arousal,
            hint_simulate_first,
            hint_novelty_lock,
            reason_codes,
        }
    }

    fn snapshot_digest(&self) -> [u8; 32] {
        digest_meta("UCF:BIO:LC:SNAP", &self.runtime.snapshot_digest())
    }

    fn config_digest(&self) -> [u8; 32] {
        digest_meta("UCF:BIO:LC:CFG", &self.runtime.config_digest())
    }
}

impl DbmModule for LcMicrocircuit {
    type Input = LcInput;
    type Output = LcOutput;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        self.step(input, 0)
    }
}

fn build_edges() -> (Vec<biophys_core::SynapseEdge>, Vec<StpParams>) {
    let mut edges = Vec::new();
    let mut params = Vec::new();

    for pre in 0..EXCITATORY_COUNT {
        for post in 0..EXCITATORY_COUNT {
            if pre == post {
                continue;
            }
            let delay_steps = 1 + ((pre + post) % 2) as u16;
            edges.push(biophys_core::SynapseEdge {
                pre: NeuronId(pre as u32),
                post: NeuronId(post as u32),
                weight_base: EXCITATORY_WEIGHT,
                weight_effective: EXCITATORY_WEIGHT,
                delay_steps,
                mod_channel: ModChannel::None,
                stp: StpState {
                    x: STP_SCALE,
                    u: EXCITATORY_STP.u,
                },
            });
            params.push(EXCITATORY_STP);
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
                    u: NO_STP.u,
                },
            });
            params.push(NO_STP);
        }
    }

    (edges, params)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_input() -> LcInput {
        LcInput {
            integrity: IntegrityState::Ok,
            arousal_floor: LevelClass::Low,
            ..Default::default()
        }
    }

    #[test]
    fn determinism_and_digest_consistency() {
        let inputs = (0..20)
            .map(|idx| LcInput {
                integrity: if idx % 7 == 0 {
                    IntegrityState::Degraded
                } else {
                    IntegrityState::Ok
                },
                receipt_invalid_count_short: if idx % 5 == 0 { 1 } else { 0 },
                dlp_critical_present_short: idx % 3 == 0,
                deny_count_short: if idx % 4 == 0 { 2 } else { 0 },
                arousal_floor: if idx % 6 == 0 {
                    LevelClass::Med
                } else {
                    LevelClass::Low
                },
                ..base_input()
            })
            .collect::<Vec<_>>();

        let run_sequence = |inputs: &[LcInput]| -> Vec<(LcOutput, [u8; 32])> {
            let mut circuit = LcMicrocircuit::new(CircuitConfig::default());
            inputs
                .iter()
                .map(|input| {
                    let output = circuit.step(input, 0);
                    let digest = circuit.snapshot_digest();
                    (output, digest)
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
    fn spike_counts_are_bounded() {
        let mut circuit = LcMicrocircuit::new(CircuitConfig::default());
        let input = LcInput {
            integrity: IntegrityState::Fail,
            receipt_invalid_count_short: 1,
            dlp_critical_present_short: true,
            deny_count_short: 2,
            arousal_floor: LevelClass::High,
            ..base_input()
        };

        circuit.step(&input, 0);
        assert!(circuit.state.last_spike_count_total <= MAX_SPIKES_PER_STEP);
    }
}
