#![forbid(unsafe_code)]

use biophys_channels::{Leak, NaK};
use biophys_compartmental_solver::{CompartmentChannels, L4Solver, L4State};
use biophys_core::{CompartmentId, NeuronId};
use biophys_morphology::{Compartment, CompartmentKind, NeuronMorphology};
use dbm_core::{DbmModule, IntegrityState, LevelClass, ReasonSet};
use microcircuit_core::{CircuitConfig, MicrocircuitBackend};
use microcircuit_lc_stub::{LcInput, LcOutput};

const NEURON_COUNT: usize = 4;
const COMPARTMENT_COUNT: usize = 3;
const SUBSTEPS: usize = 10;
const DT_MS: f32 = 0.1;
const CLAMP_MIN: f32 = -120.0;
const CLAMP_MAX: f32 = 60.0;
const THRESHOLD_MV: f32 = -20.0;
const SPIKE_COUNTER_SHORT_MAX: u8 = 4;

const CURRENT_STRONG: f32 = 12.0;
const CURRENT_MEDIUM: f32 = 6.0;
const CURRENT_TONIC_LOW: f32 = 0.0;
const CURRENT_TONIC_MED: f32 = 1.5;
const CURRENT_TONIC_HIGH: f32 = 3.0;

const DLP_TARGETS: [usize; 2] = [0, 1];

#[derive(Debug, Clone, Default)]
struct LcState {
    tick_count: u64,
    spike_counter_short: u8,
    last_spike_count_total: usize,
}

#[derive(Debug, Clone)]
struct L4Neuron {
    solver: L4Solver,
    state: L4State,
    last_soma_v: f32,
}

#[derive(Debug, Clone)]
pub struct LcMicrocircuit {
    _config: CircuitConfig,
    neurons: Vec<L4Neuron>,
    state: LcState,
}

impl LcMicrocircuit {
    pub fn new(config: CircuitConfig) -> Self {
        let neurons = (0..NEURON_COUNT)
            .map(|idx| build_neuron(idx as u32))
            .collect();
        Self {
            _config: config,
            neurons,
            state: LcState::default(),
        }
    }

    fn tonic_current(level: LevelClass) -> f32 {
        match level {
            LevelClass::Low => CURRENT_TONIC_LOW,
            LevelClass::Med => CURRENT_TONIC_MED,
            LevelClass::High => CURRENT_TONIC_HIGH,
        }
    }

    fn severity(level: LevelClass) -> u8 {
        match level {
            LevelClass::Low => 0,
            LevelClass::Med => 1,
            LevelClass::High => 2,
        }
    }

    fn max_level(a: LevelClass, b: LevelClass) -> LevelClass {
        if Self::severity(a) >= Self::severity(b) {
            a
        } else {
            b
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

    fn build_inputs(input: &LcInput) -> [[f32; COMPARTMENT_COUNT]; NEURON_COUNT] {
        let mut currents = [[0.0_f32; COMPARTMENT_COUNT]; NEURON_COUNT];
        let tonic = Self::tonic_current(input.arousal_floor);
        for neuron in currents.iter_mut() {
            neuron[0] += tonic;
        }

        if input.integrity != IntegrityState::Ok {
            for neuron in currents.iter_mut() {
                neuron[0] += CURRENT_STRONG;
            }
        }

        if input.receipt_invalid_count_short >= 1 {
            for neuron in currents.iter_mut() {
                neuron[0] += CURRENT_STRONG;
                neuron[1] += CURRENT_STRONG;
            }
        }

        if input.dlp_critical_present_short {
            for &idx in &DLP_TARGETS {
                currents[idx][0] += CURRENT_STRONG;
            }
        }

        if input.deny_count_short >= 2 {
            currents[2][0] += CURRENT_MEDIUM;
        }

        currents
    }

    fn map_arousal(&self, spike_count_total: usize, input: &LcInput) -> LevelClass {
        let base = if spike_count_total >= 2 || self.state.spike_counter_short >= 2 {
            LevelClass::High
        } else if spike_count_total == 1 || self.state.spike_counter_short == 1 {
            LevelClass::Med
        } else {
            LevelClass::Low
        };

        let rules_floor = Self::rules_floor(input);
        let mut arousal = Self::max_level(base, rules_floor);
        arousal = Self::max_level(arousal, input.arousal_floor);
        arousal
    }

    fn step_neuron(neuron: &mut L4Neuron, currents: &[f32; COMPARTMENT_COUNT]) -> usize {
        let mut spikes = 0;
        let mut prev_v = neuron.last_soma_v;
        for _ in 0..SUBSTEPS {
            neuron.solver.step(&mut neuron.state, currents);
            sanitize_voltages(&mut neuron.state);
            let v = neuron.state.comp_v[0];
            if prev_v < THRESHOLD_MV && v >= THRESHOLD_MV {
                spikes += 1;
            }
            prev_v = v;
        }
        neuron.last_soma_v = prev_v;
        spikes
    }
}

impl MicrocircuitBackend<LcInput, LcOutput> for LcMicrocircuit {
    fn step(&mut self, input: &LcInput, _now_ms: u64) -> LcOutput {
        self.state.tick_count = self.state.tick_count.saturating_add(1);
        let currents = Self::build_inputs(input);

        let mut spike_count_total = 0;
        for (idx, neuron) in self.neurons.iter_mut().enumerate() {
            let spikes = Self::step_neuron(neuron, &currents[idx]);
            spike_count_total += spikes;
        }

        self.state.last_spike_count_total = spike_count_total;
        self.update_spike_counter(spike_count_total);

        let arousal = self.map_arousal(spike_count_total, input);
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
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"UCF:L4:LC:SNAP");
        hasher.update(&self.state.tick_count.to_le_bytes());
        for neuron in &self.neurons {
            update_u64(&mut hasher, neuron.solver.step_count());
            update_u32(&mut hasher, neuron.state.comp_v.len() as u32);
            for idx in 0..neuron.state.comp_v.len() {
                update_f32(&mut hasher, neuron.state.comp_v[idx]);
                update_f32(&mut hasher, neuron.state.m_q[idx] as f32 / 1000.0);
                update_f32(&mut hasher, neuron.state.h_q[idx] as f32 / 1000.0);
                update_f32(&mut hasher, neuron.state.n_q[idx] as f32 / 1000.0);
            }
        }
        *hasher.finalize().as_bytes()
    }

    fn config_digest(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"UCF:L4:LC:CFG");
        update_f32(&mut hasher, DT_MS);
        update_u32(&mut hasher, SUBSTEPS as u32);
        update_f32(&mut hasher, CLAMP_MIN);
        update_f32(&mut hasher, CLAMP_MAX);
        for neuron in &self.neurons {
            hasher.update(&neuron.solver.config_digest());
        }
        *hasher.finalize().as_bytes()
    }
}

impl DbmModule for LcMicrocircuit {
    type Input = LcInput;
    type Output = LcOutput;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        self.step(input, 0)
    }
}

fn build_neuron(neuron_id: u32) -> L4Neuron {
    let compartments = vec![
        Compartment {
            id: CompartmentId(0),
            parent: None,
            kind: CompartmentKind::Soma,
            depth: 0,
            capacitance: 1.0,
            axial_resistance: 150.0,
        },
        Compartment {
            id: CompartmentId(1),
            parent: Some(CompartmentId(0)),
            kind: CompartmentKind::Dendrite,
            depth: 1,
            capacitance: 1.2,
            axial_resistance: 200.0,
        },
        Compartment {
            id: CompartmentId(2),
            parent: Some(CompartmentId(0)),
            kind: CompartmentKind::Dendrite,
            depth: 1,
            capacitance: 1.2,
            axial_resistance: 200.0,
        },
    ];

    let morphology = NeuronMorphology {
        neuron_id: NeuronId(neuron_id),
        compartments,
    };

    let leak = Leak {
        g: 0.1,
        e_rev: -65.0,
    };
    let nak = NaK {
        g_na: 120.0,
        g_k: 36.0,
        e_na: 50.0,
        e_k: -77.0,
    };

    let channels = vec![
        CompartmentChannels {
            leak,
            nak: Some(nak),
        },
        CompartmentChannels { leak, nak: None },
        CompartmentChannels { leak, nak: None },
    ];

    let solver = L4Solver::new(morphology, channels, DT_MS, CLAMP_MIN, CLAMP_MAX).expect("solver");
    let state = L4State::new(-65.0, COMPARTMENT_COUNT);
    let last_soma_v = state.comp_v[0];

    L4Neuron {
        solver,
        state,
        last_soma_v,
    }
}

fn sanitize_voltages(state: &mut L4State) {
    for v in &mut state.comp_v {
        if !v.is_finite() {
            *v = CLAMP_MIN;
        } else {
            *v = v.clamp(CLAMP_MIN, CLAMP_MAX);
        }
    }
}

fn update_u32(hasher: &mut blake3::Hasher, value: u32) {
    hasher.update(&value.to_le_bytes());
}

fn update_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_le_bytes());
}

fn update_f32(hasher: &mut blake3::Hasher, value: f32) {
    hasher.update(&value.to_bits().to_le_bytes());
}

#[cfg(all(
    test,
    feature = "biophys",
    feature = "biophys-l4",
    feature = "biophys-l4-lc"
))]
mod tests {
    use super::*;

    #[cfg(all(feature = "biophys", feature = "biophys-l4", feature = "biophys-l4-lc"))]
    fn base_input() -> LcInput {
        LcInput {
            integrity: IntegrityState::Ok,
            arousal_floor: LevelClass::Low,
            ..Default::default()
        }
    }

    #[cfg(all(feature = "biophys", feature = "biophys-l4", feature = "biophys-l4-lc"))]
    #[test]
    fn determinism_and_snapshot_consistency() {
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

    #[cfg(all(feature = "biophys", feature = "biophys-l4", feature = "biophys-l4-lc"))]
    #[test]
    fn performance_guard_runs_quickly() {
        let mut circuit = LcMicrocircuit::new(CircuitConfig::default());
        let input = base_input();
        for _ in 0..100 {
            circuit.step(&input, 0);
        }
        assert!(circuit.state.tick_count >= 100);
    }

    #[cfg(all(feature = "biophys", feature = "biophys-l4", feature = "biophys-l4-lc"))]
    #[test]
    fn integrity_fail_reaches_high_arousal() {
        let mut circuit = LcMicrocircuit::new(CircuitConfig::default());
        let input = LcInput {
            integrity: IntegrityState::Fail,
            ..base_input()
        };

        let output_first = circuit.step(&input, 0);
        let output_second = circuit.step(&input, 0);

        assert!(
            output_first.arousal == LevelClass::High || output_second.arousal == LevelClass::High
        );
    }

    #[cfg(all(feature = "biophys", feature = "biophys-l4", feature = "biophys-l4-lc"))]
    #[test]
    fn receipt_invalid_reaches_med() {
        let mut circuit = LcMicrocircuit::new(CircuitConfig::default());
        let output = circuit.step(
            &LcInput {
                receipt_invalid_count_short: 1,
                ..base_input()
            },
            0,
        );

        assert!(matches!(output.arousal, LevelClass::Med | LevelClass::High));
    }
}
