#![forbid(unsafe_code)]

use biophys_core::{
    LifParams, LifState, ModChannel, NeuronId, StpParams, StpState, STP_SCALE,
};
use biophys_runtime::BiophysRuntime;
use dbm_core::{DbmModule, IntegrityState, LevelClass, ReasonSet};
use microcircuit_core::{digest_meta, CircuitConfig, MicrocircuitBackend};
use microcircuit_stn_stub::{StnInput, StnOutput};

const EXCITATORY_COUNT: usize = 6;
const INHIBITORY_COUNT: usize = 2;
const NEURON_COUNT: usize = EXCITATORY_COUNT + INHIBITORY_COUNT;
const INHIBITORY_START: usize = EXCITATORY_COUNT;

const DT_MS: u16 = 1;
const MAX_SPIKES_PER_STEP: usize = 16;
const MAX_EVENTS_PER_STEP: usize = 128;

const CURRENT_STRONG: i32 = 30;
const CURRENT_MEDIUM: i32 = 18;
const CURRENT_SMALL: i32 = 8;

const EXCITATORY_WEIGHT: i32 = 3;
const EXC_TO_INHIB_WEIGHT: i32 = 30;
const INHIB_TO_EXC_WEIGHT: i32 = -12;

const DLP_SUBSET: [usize; 3] = [0, 1, 2];
const POLICY_SUBSET: [usize; 3] = [3, 4, 5];

const NO_STP: StpParams = StpParams {
    u: STP_SCALE,
    tau_rec_steps: 0,
    tau_fac_steps: 0,
    mod_channel: None,
};

#[derive(Debug, Clone, Default)]
struct StnBiophysState {
    latch_steps: u8,
}

#[derive(Debug, Clone)]
pub struct StnBiophysMicrocircuit {
    _config: CircuitConfig,
    runtime: BiophysRuntime,
    state: StnBiophysState,
}

impl StnBiophysMicrocircuit {
    pub fn new(config: CircuitConfig) -> Self {
        let params = vec![
            LifParams {
                tau_ms: 1,
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
            state: StnBiophysState::default(),
        }
    }

    fn apply_current(currents: &mut [i32; NEURON_COUNT], idx: usize, current: i32) {
        currents[idx] = currents[idx].saturating_add(current);
    }

    fn apply_to_all_exc(currents: &mut [i32; NEURON_COUNT], current: i32) {
        for idx in 0..EXCITATORY_COUNT {
            Self::apply_current(currents, idx, current);
        }
    }

    fn apply_subset(currents: &mut [i32; NEURON_COUNT], subset: &[usize], current: i32) {
        for &idx in subset {
            Self::apply_current(currents, idx, current);
        }
    }

    fn encode_inputs(input: &StnInput) -> [i32; NEURON_COUNT] {
        let mut currents = [0i32; NEURON_COUNT];

        if input.integrity != IntegrityState::Ok {
            Self::apply_to_all_exc(&mut currents, CURRENT_STRONG);
        }
        if input.receipt_invalid_present {
            Self::apply_to_all_exc(&mut currents, CURRENT_STRONG);
        }
        if input.dlp_critical_present {
            Self::apply_subset(&mut currents, &DLP_SUBSET, CURRENT_STRONG);
        }
        if input.policy_pressure == LevelClass::High {
            Self::apply_subset(&mut currents, &POLICY_SUBSET, CURRENT_MEDIUM);
        }
        if input.threat == LevelClass::High {
            Self::apply_to_all_exc(&mut currents, CURRENT_MEDIUM);
        }
        if input.arousal == LevelClass::High {
            Self::apply_to_all_exc(&mut currents, CURRENT_SMALL);
        }

        currents
    }

    fn inhibitory_spike_count(spikes: &[NeuronId]) -> usize {
        spikes
            .iter()
            .filter(|id| id.0 as usize >= INHIBITORY_START)
            .count()
    }

    fn push_integrity_reason(reason_codes: &mut ReasonSet, integrity: IntegrityState) {
        match integrity {
            IntegrityState::Degraded => {
                reason_codes.insert("RC.RE.INTEGRITY.DEGRADED");
            }
            IntegrityState::Fail => {
                reason_codes.insert("RC.RE.INTEGRITY.FAIL");
            }
            IntegrityState::Ok => {}
        }
    }
}

impl MicrocircuitBackend<StnInput, StnOutput> for StnBiophysMicrocircuit {
    fn step(&mut self, input: &StnInput, _now_ms: u64) -> StnOutput {
        let currents = Self::encode_inputs(input);
        self.runtime.set_modulators(input.modulators);
        let pop = self.runtime.step(&currents);

        let inhib_spikes = Self::inhibitory_spike_count(&pop.spikes);
        let critical_hold = input.integrity != IntegrityState::Ok
            || input.receipt_invalid_present
            || input.dlp_critical_present;
        let policy_hold = input.policy_pressure == LevelClass::High;
        let mut hold_active = inhib_spikes >= 1 || critical_hold || policy_hold;

        if hold_active {
            self.state.latch_steps = 3;
        } else if self.state.latch_steps > 0 {
            self.state.latch_steps -= 1;
        }

        hold_active = hold_active || self.state.latch_steps > 0;

        let hint_simulate_first = hold_active;
        let hint_novelty_lock = hold_active
            && (input.policy_pressure == LevelClass::High || input.arousal == LevelClass::High);
        let hint_export_lock = input.dlp_critical_present;

        let mut hold_reason_codes = ReasonSet::default();
        if hold_active {
            hold_reason_codes.insert("RC.GV.HOLD.ON");
        }
        if input.integrity != IntegrityState::Ok {
            Self::push_integrity_reason(&mut hold_reason_codes, input.integrity);
        }
        if input.receipt_invalid_present {
            hold_reason_codes.insert("RC.GE.EXEC.DISPATCH_BLOCKED");
        }
        if input.dlp_critical_present {
            hold_reason_codes.insert("RC.CD.DLP.SECRET_PATTERN");
        }

        StnOutput {
            hold_active,
            hold_reason_codes,
            hint_simulate_first,
            hint_novelty_lock,
            hint_export_lock,
        }
    }

    fn snapshot_digest(&self) -> [u8; 32] {
        digest_meta("UCF:BIO:STN:SNAP", &self.runtime.snapshot_digest())
    }

    fn config_digest(&self) -> [u8; 32] {
        digest_meta("UCF:BIO:STN:CFG", &self.runtime.config_digest())
    }
}

impl DbmModule for StnBiophysMicrocircuit {
    type Input = StnInput;
    type Output = StnOutput;

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
            edges.push(biophys_core::SynapseEdge {
                pre: NeuronId(pre as u32),
                post: NeuronId(post as u32),
                weight_base: EXCITATORY_WEIGHT,
                weight_effective: EXCITATORY_WEIGHT,
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
                    u: NO_STP.u,
                },
            });
            params.push(NO_STP);
        }
    }

    for pre in INHIBITORY_START..NEURON_COUNT {
        for post in 0..EXCITATORY_COUNT {
            edges.push(biophys_core::SynapseEdge {
                pre: NeuronId(pre as u32),
                post: NeuronId(post as u32),
                weight_base: INHIB_TO_EXC_WEIGHT,
                weight_effective: INHIB_TO_EXC_WEIGHT,
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
    use biophys_core::ModulatorField;

    fn base_input() -> StnInput {
        StnInput {
            policy_pressure: LevelClass::Low,
            arousal: LevelClass::Low,
            threat: LevelClass::Low,
            receipt_invalid_present: false,
            dlp_critical_present: false,
            integrity: IntegrityState::Ok,
            tool_side_effects_present: false,
            cerebellum_divergence: LevelClass::Low,
            modulators: ModulatorField::default(),
        }
    }

    #[test]
    fn determinism_same_sequence_same_outputs() {
        let mut left = StnBiophysMicrocircuit::new(CircuitConfig::default());
        let mut right = StnBiophysMicrocircuit::new(CircuitConfig::default());
        let sequence = [
            base_input(),
            StnInput {
                receipt_invalid_present: true,
                ..base_input()
            },
            StnInput {
                integrity: IntegrityState::Fail,
                ..base_input()
            },
            StnInput {
                threat: LevelClass::High,
                ..base_input()
            },
            base_input(),
        ];

        let outputs_left: Vec<StnOutput> =
            sequence.iter().map(|input| left.step(input, 0)).collect();
        let outputs_right: Vec<StnOutput> =
            sequence.iter().map(|input| right.step(input, 0)).collect();

        assert_eq!(outputs_left, outputs_right);
    }

    #[test]
    fn receipt_invalid_triggers_hold_within_one_tick() {
        let mut micro = StnBiophysMicrocircuit::new(CircuitConfig::default());
        let _ = micro.step(
            &StnInput {
                receipt_invalid_present: true,
                ..base_input()
            },
            0,
        );

        let output = micro.step(&base_input(), 0);
        assert!(output.hold_active);
        assert!(output.hint_simulate_first);
    }

    #[test]
    fn latch_holds_for_three_ticks() {
        let mut micro = StnBiophysMicrocircuit::new(CircuitConfig::default());
        let _ = micro.step(
            &StnInput {
                receipt_invalid_present: true,
                ..base_input()
            },
            0,
        );

        let first_hold = micro.step(&base_input(), 0);
        assert!(first_hold.hold_active);

        let second_hold = micro.step(&base_input(), 0);
        assert!(second_hold.hold_active);

        let third_hold = micro.step(&base_input(), 0);
        assert!(third_hold.hold_active);

        let released = micro.step(&base_input(), 0);
        assert!(!released.hold_active);
    }
}
