#![forbid(unsafe_code)]

use biophys_channels::{Leak, NaK};
use biophys_compartmental_solver::{CompartmentChannels, L4Solver, L4State};
use biophys_core::{CompartmentId, ModChannel, ModLevel, ModulatorField, NeuronId};
use biophys_event_queue_l4::SpikeEventQueueL4;
use biophys_morphology::{Compartment, CompartmentKind, NeuronMorphology};
use biophys_synapses_l4::{
    decay_k, f32_to_fixed_u32, StpParamsL4, StpStateL4, SynKind, SynapseAccumulator, SynapseL4,
    SynapseState,
};
use dbm_core::{IntegrityState, LevelClass, ReasonSet, ThreatVector};
use microcircuit_amygdala_stub::{AmyInput, AmyOutput};
use microcircuit_core::{CircuitConfig, MicrocircuitBackend};

const POOL_COUNT: usize = 4;
const POOL_SIZE: usize = 2;
const EXCITATORY_COUNT: usize = POOL_COUNT * POOL_SIZE;
const INHIBITORY_COUNT: usize = 1;
const NEURON_COUNT: usize = EXCITATORY_COUNT + INHIBITORY_COUNT;

const IDX_INTEGRITY: usize = 0;
const IDX_EXFIL: usize = 1;
const IDX_PROBING: usize = 2;
const IDX_TOOL: usize = 3;

const COMPARTMENT_COUNT: usize = 3;
const SUBSTEPS: usize = 20;
const DT_MS: f32 = 0.1;
const CLAMP_MIN: f32 = -120.0;
const CLAMP_MAX: f32 = 60.0;
const THRESHOLD_MV: f32 = -20.0;

const CURRENT_STRONG: f32 = 60.0;
const CURRENT_MED: f32 = 30.0;

const AMPA_G_MAX: f32 = 4.0;
const AMPA_G_MAX_WEAK: f32 = 2.0;
const AMPA_E_REV: f32 = 0.0;
const AMPA_TAU_RISE_MS: f32 = 0.0;
const AMPA_TAU_DECAY_MS: f32 = 8.0;

const GABA_G_MAX: f32 = 6.0;
const GABA_E_REV: f32 = -70.0;
const GABA_TAU_RISE_MS: f32 = 0.0;
const GABA_TAU_DECAY_MS: f32 = 10.0;

const MAX_EVENTS_PER_STEP: usize = 1024;
const ACCUMULATOR_MAX: i32 = 100;
const ACCUMULATOR_DECAY: i32 = 5;
const ACCUMULATOR_GAIN: i32 = 20;
const LATCH_HIGH: i32 = 70;
const LATCH_LOW: i32 = 40;
const LATCH_STEPS_MAX: u8 = 10;

#[derive(Debug, Clone)]
struct L4Neuron {
    solver: L4Solver,
    state: L4State,
    last_soma_v: f32,
}

#[derive(Debug, Clone)]
struct AmyL4State {
    tick_count: u64,
    step_count: u64,
    pool_acc: [i32; POOL_COUNT],
    latch_steps: [u8; POOL_COUNT],
    last_pool_spikes: [usize; POOL_COUNT],
    last_spike_count_total: usize,
}

impl Default for AmyL4State {
    fn default() -> Self {
        Self {
            tick_count: 0,
            step_count: 0,
            pool_acc: [0; POOL_COUNT],
            latch_steps: [0; POOL_COUNT],
            last_pool_spikes: [0; POOL_COUNT],
            last_spike_count_total: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AmygdalaL4Microcircuit {
    _config: CircuitConfig,
    neurons: Vec<L4Neuron>,
    synapses: Vec<SynapseL4>,
    syn_states: Vec<SynapseState>,
    syn_g_max_eff: Vec<u32>,
    syn_decay: Vec<u16>,
    syn_stp_params_eff: Vec<StpParamsL4>,
    pre_index: Vec<Vec<usize>>,
    queue: SpikeEventQueueL4,
    state: AmyL4State,
    current_modulators: ModulatorField,
}

impl AmygdalaL4Microcircuit {
    pub fn new(config: CircuitConfig) -> Self {
        let neurons = (0..NEURON_COUNT)
            .map(|idx| build_neuron(idx as u32))
            .collect::<Vec<_>>();
        let synapses = build_synapses();
        let syn_states = vec![SynapseState::default(); synapses.len()];
        let current_modulators = ModulatorField::default();
        let syn_g_max_eff = synapses
            .iter()
            .map(|synapse| synapse.effective_g_max_fixed(current_modulators))
            .collect::<Vec<_>>();
        let syn_decay = synapses
            .iter()
            .map(|synapse| decay_k(DT_MS, synapse.tau_decay_ms))
            .collect::<Vec<_>>();
        let syn_stp_params_eff = synapses
            .iter()
            .map(|synapse| synapse.stp_effective_params(current_modulators))
            .collect::<Vec<_>>();
        let pre_index = build_pre_index(NEURON_COUNT, &synapses);
        let max_delay = synapses
            .iter()
            .map(|synapse| synapse.delay_steps)
            .max()
            .unwrap_or(0);
        let queue = SpikeEventQueueL4::new(max_delay, MAX_EVENTS_PER_STEP);
        Self {
            _config: config,
            neurons,
            synapses,
            syn_states,
            syn_g_max_eff,
            syn_decay,
            syn_stp_params_eff,
            pre_index,
            queue,
            state: AmyL4State::default(),
            current_modulators,
        }
    }

    fn pool_bounds(pool: usize) -> (usize, usize) {
        let start = pool * POOL_SIZE;
        (start, start + POOL_SIZE)
    }

    fn tool_anomaly_present(input: &AmyInput) -> bool {
        input.tool_anomaly_present
            || input.cerebellum_tool_anomaly_present.unwrap_or(false)
            || input
                .tool_anomalies
                .iter()
                .any(|(_, level)| matches!(level, LevelClass::High))
    }

    fn apply_pool_current(
        currents: &mut [[f32; COMPARTMENT_COUNT]; NEURON_COUNT],
        pool: usize,
        soma: f32,
        dend1: f32,
    ) {
        let (start, end) = Self::pool_bounds(pool);
        for neuron in currents.iter_mut().take(end).skip(start) {
            neuron[0] += soma;
            neuron[1] += dend1;
        }
    }

    fn build_inputs(input: &AmyInput) -> [[f32; COMPARTMENT_COUNT]; NEURON_COUNT] {
        let mut currents = [[0.0_f32; COMPARTMENT_COUNT]; NEURON_COUNT];

        let integrity_strong = matches!(
            input.integrity,
            IntegrityState::Fail | IntegrityState::Degraded
        );
        let integrity_medium = input.replay_mismatch_present || input.receipt_invalid_medium >= 1;
        if integrity_strong {
            Self::apply_pool_current(&mut currents, IDX_INTEGRITY, CURRENT_STRONG, CURRENT_STRONG);
        }
        if integrity_medium {
            Self::apply_pool_current(&mut currents, IDX_INTEGRITY, CURRENT_MED, CURRENT_MED);
        }

        let exfil_strong =
            input.dlp_secret_present || input.dlp_obfuscation_present || input.dlp_stegano_present;
        if exfil_strong {
            Self::apply_pool_current(&mut currents, IDX_EXFIL, CURRENT_STRONG, CURRENT_STRONG);
        }

        if input.policy_pressure == LevelClass::High {
            Self::apply_pool_current(&mut currents, IDX_PROBING, CURRENT_MED, 0.0);
        }

        if Self::tool_anomaly_present(input) {
            Self::apply_pool_current(&mut currents, IDX_TOOL, CURRENT_MED, 0.0);
        }

        currents
    }

    fn update_pool_accumulators(&mut self, spike_counts: &[usize; NEURON_COUNT]) {
        let mut pool_counts = [0usize; POOL_COUNT];
        for (idx, count) in spike_counts.iter().enumerate().take(EXCITATORY_COUNT) {
            let pool = idx / POOL_SIZE;
            pool_counts[pool] = pool_counts[pool].saturating_add(*count);
        }

        self.state.last_pool_spikes = pool_counts;
        self.state.last_spike_count_total = spike_counts.iter().sum();

        for (acc, &count) in self.state.pool_acc.iter_mut().zip(pool_counts.iter()) {
            let delta = (count as i32).saturating_mul(ACCUMULATOR_GAIN);
            *acc = (*acc + delta - ACCUMULATOR_DECAY).clamp(0, ACCUMULATOR_MAX);
        }
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

    #[cfg(feature = "biophys-l4-modulation")]
    fn update_modulators(&mut self, input: &AmyInput) {
        self.current_modulators = input.modulators;
        for (idx, synapse) in self.synapses.iter().enumerate() {
            self.syn_g_max_eff[idx] = synapse.effective_g_max_fixed(self.current_modulators);
            self.syn_stp_params_eff[idx] = synapse.stp_effective_params(self.current_modulators);
        }
    }

    #[cfg(not(feature = "biophys-l4-modulation"))]
    fn update_modulators(&mut self, _input: &AmyInput) {
        self.current_modulators = ModulatorField::default();
        for (idx, synapse) in self.synapses.iter().enumerate() {
            self.syn_g_max_eff[idx] = synapse.effective_g_max_fixed(self.current_modulators);
            self.syn_stp_params_eff[idx] = synapse.stp_effective_params(self.current_modulators);
        }
    }

    fn substep(
        &mut self,
        injected_currents: &[[f32; COMPARTMENT_COUNT]; NEURON_COUNT],
    ) -> Vec<usize> {
        for (state, decay) in self.syn_states.iter_mut().zip(self.syn_decay.iter()) {
            state.decay(*decay);
        }

        let events = self.queue.drain_current(self.state.step_count);
        for event in events {
            let g_max_eff = self.syn_g_max_eff[event.synapse_index];
            self.syn_states[event.synapse_index].apply_spike(g_max_eff, event.release_gain_q);
        }

        let mut accumulators =
            vec![vec![SynapseAccumulator::default(); COMPARTMENT_COUNT]; NEURON_COUNT];
        for (idx, synapse) in self.synapses.iter().enumerate() {
            let mut g_fixed = self.syn_states[idx].g_fixed;
            if synapse.kind == SynKind::GABA {
                g_fixed = scale_fixed_by_level(g_fixed, self.current_modulators.ht);
            }
            if g_fixed == 0 {
                continue;
            }
            let post = synapse.post_neuron as usize;
            let compartment = synapse.post_compartment as usize;
            accumulators[post][compartment].add(synapse.kind, g_fixed, synapse.e_rev);
        }

        let mut spikes = Vec::new();
        for (idx, neuron) in self.neurons.iter_mut().enumerate() {
            let input = injected_currents[idx];
            let syn_input = &accumulators[idx];
            neuron
                .solver
                .step_with_synapses(&mut neuron.state, &input, syn_input);
            sanitize_voltages(&mut neuron.state);
            let v = neuron.state.voltages[0];
            if neuron.last_soma_v < THRESHOLD_MV && v >= THRESHOLD_MV {
                spikes.push(idx);
            }
            neuron.last_soma_v = v;
        }

        for spike_idx in &spikes {
            let indices = &self.pre_index[*spike_idx];
            self.queue.schedule_spike(
                self.state.step_count,
                indices,
                |idx| self.synapses[idx].delay_steps,
                |idx| {
                    #[cfg(feature = "biophys-l4-stp")]
                    {
                        let params = self.syn_stp_params_eff[idx];
                        return self.synapses[idx].stp_release_on_spike(params);
                    }
                    #[cfg(not(feature = "biophys-l4-stp"))]
                    {
                        let _ = idx;
                        biophys_core::STP_SCALE
                    }
                },
            );
        }

        self.state.step_count = self.state.step_count.saturating_add(1);
        spikes
    }
}

impl MicrocircuitBackend<AmyInput, AmyOutput> for AmygdalaL4Microcircuit {
    fn step(&mut self, input: &AmyInput, _now_ms: u64) -> AmyOutput {
        self.state.tick_count = self.state.tick_count.saturating_add(1);
        self.update_modulators(input);
        #[cfg(feature = "biophys-l4-stp")]
        {
            for (synapse, params) in self
                .synapses
                .iter_mut()
                .zip(self.syn_stp_params_eff.iter().copied())
            {
                synapse.stp_recover_tick(params);
            }
        }
        let currents = Self::build_inputs(input);

        let mut spike_counts = [0usize; NEURON_COUNT];
        for _ in 0..SUBSTEPS {
            let spikes = self.substep(&currents);
            for spike in spikes {
                spike_counts[spike] = spike_counts[spike].saturating_add(1);
            }
        }

        self.update_pool_accumulators(&spike_counts);
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
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"UCF:BIO:L4:AMY:SNAP");
        update_u64(&mut hasher, self.state.step_count);
        for value in self.state.pool_acc {
            update_i32(&mut hasher, value);
        }
        for value in self.state.latch_steps {
            update_u32(&mut hasher, value as u32);
        }
        for neuron in &self.neurons {
            update_u64(&mut hasher, neuron.solver.step_count());
            update_u32(&mut hasher, neuron.state.voltages.len() as u32);
            for (v, gates) in neuron.state.voltages.iter().zip(neuron.state.gates.iter()) {
                update_i32(&mut hasher, quantize_f32(*v, 100.0));
                update_i32(&mut hasher, quantize_f32(gates.m, 1000.0));
                update_i32(&mut hasher, quantize_f32(gates.h, 1000.0));
                update_i32(&mut hasher, quantize_f32(gates.n, 1000.0));
            }
        }
        for state in &self.syn_states {
            update_u32(&mut hasher, state.g_fixed);
        }
        *hasher.finalize().as_bytes()
    }

    fn config_digest(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"UCF:BIO:L4:AMY:CFG");
        update_f32(&mut hasher, DT_MS);
        update_u32(&mut hasher, SUBSTEPS as u32);
        update_f32(&mut hasher, CLAMP_MIN);
        update_f32(&mut hasher, CLAMP_MAX);
        for neuron in &self.neurons {
            hasher.update(&neuron.solver.config_digest());
        }
        update_u32(&mut hasher, self.synapses.len() as u32);
        for synapse in &self.synapses {
            update_u32(&mut hasher, synapse.pre_neuron);
            update_u32(&mut hasher, synapse.post_neuron);
            update_u32(&mut hasher, synapse.post_compartment);
            update_u32(&mut hasher, synapse.kind as u32);
            update_u32(&mut hasher, synapse.mod_channel as u32);
            update_u32(&mut hasher, synapse.g_max_base_q);
            update_f32(&mut hasher, synapse.e_rev);
            update_f32(&mut hasher, synapse.tau_rise_ms);
            update_f32(&mut hasher, synapse.tau_decay_ms);
            update_u32(&mut hasher, synapse.delay_steps as u32);
        }
        *hasher.finalize().as_bytes()
    }
}

fn build_neuron(neuron_id: u32) -> L4Neuron {
    let compartments = vec![
        Compartment {
            id: CompartmentId(0),
            parent: None,
            kind: CompartmentKind::Soma,
            capacitance: 1.0,
            axial_resistance: 150.0,
        },
        Compartment {
            id: CompartmentId(1),
            parent: Some(CompartmentId(0)),
            kind: CompartmentKind::Dendrite,
            capacitance: 1.2,
            axial_resistance: 200.0,
        },
        Compartment {
            id: CompartmentId(2),
            parent: Some(CompartmentId(0)),
            kind: CompartmentKind::Dendrite,
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
    let last_soma_v = state.voltages[0];

    L4Neuron {
        solver,
        state,
        last_soma_v,
    }
}

fn build_synapses() -> Vec<SynapseL4> {
    let mut synapses = Vec::new();

    for pool in 0..POOL_COUNT {
        let (start, end) = AmygdalaL4Microcircuit::pool_bounds(pool);
        for pre in start..end {
            for post in start..end {
                if pre == post {
                    continue;
                }
                let (stp_params, stp_state) = disabled_stp();
                synapses.push(SynapseL4 {
                    pre_neuron: pre as u32,
                    post_neuron: post as u32,
                    post_compartment: 1,
                    kind: SynKind::AMPA,
                    mod_channel: ModChannel::NaDa,
                    g_max_base_q: f32_to_fixed_u32(AMPA_G_MAX),
                    e_rev: AMPA_E_REV,
                    tau_rise_ms: AMPA_TAU_RISE_MS,
                    tau_decay_ms: AMPA_TAU_DECAY_MS,
                    delay_steps: 1,
                    stp_params,
                    stp_state,
                });
            }
        }
    }

    let (integrity_start, integrity_end) = AmygdalaL4Microcircuit::pool_bounds(IDX_INTEGRITY);
    let (exfil_start, exfil_end) = AmygdalaL4Microcircuit::pool_bounds(IDX_EXFIL);
    for pre in integrity_start..integrity_end {
        for post in exfil_start..exfil_end {
            let (stp_params, stp_state) = disabled_stp();
            synapses.push(SynapseL4 {
                pre_neuron: pre as u32,
                post_neuron: post as u32,
                post_compartment: 1,
                kind: SynKind::AMPA,
                mod_channel: ModChannel::NaDa,
                g_max_base_q: f32_to_fixed_u32(AMPA_G_MAX_WEAK),
                e_rev: AMPA_E_REV,
                tau_rise_ms: AMPA_TAU_RISE_MS,
                tau_decay_ms: AMPA_TAU_DECAY_MS,
                delay_steps: 2,
                stp_params,
                stp_state,
            });
        }
    }

    for pre in 0..EXCITATORY_COUNT {
        let post = EXCITATORY_COUNT;
        let (stp_params, stp_state) = disabled_stp();
        synapses.push(SynapseL4 {
            pre_neuron: pre as u32,
            post_neuron: post as u32,
            post_compartment: 0,
            kind: SynKind::AMPA,
            mod_channel: ModChannel::Na,
            g_max_base_q: f32_to_fixed_u32(AMPA_G_MAX),
            e_rev: AMPA_E_REV,
            tau_rise_ms: AMPA_TAU_RISE_MS,
            tau_decay_ms: AMPA_TAU_DECAY_MS,
            delay_steps: 1,
            stp_params,
            stp_state,
        });
    }

    let pre = EXCITATORY_COUNT;
    for post in 0..EXCITATORY_COUNT {
        let (stp_params, stp_state) = disabled_stp();
        synapses.push(SynapseL4 {
            pre_neuron: pre as u32,
            post_neuron: post as u32,
            post_compartment: 0,
            kind: SynKind::GABA,
            mod_channel: ModChannel::Ht,
            g_max_base_q: f32_to_fixed_u32(GABA_G_MAX),
            e_rev: GABA_E_REV,
            tau_rise_ms: GABA_TAU_RISE_MS,
            tau_decay_ms: GABA_TAU_DECAY_MS,
            delay_steps: 1,
            stp_params,
            stp_state,
        });
    }

    synapses
}

fn disabled_stp() -> (StpParamsL4, StpStateL4) {
    let params = StpParamsL4::disabled();
    let state = StpStateL4::new(params);
    (params, state)
}

fn build_pre_index(neuron_count: usize, synapses: &[SynapseL4]) -> Vec<Vec<usize>> {
    let mut pre_index = vec![Vec::new(); neuron_count];
    for (idx, synapse) in synapses.iter().enumerate() {
        let pre = synapse.pre_neuron as usize;
        pre_index[pre].push(idx);
    }
    pre_index
}

fn sanitize_voltages(state: &mut L4State) {
    for v in &mut state.voltages {
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

fn update_i32(hasher: &mut blake3::Hasher, value: i32) {
    hasher.update(&value.to_le_bytes());
}

fn update_f32(hasher: &mut blake3::Hasher, value: f32) {
    hasher.update(&value.to_bits().to_le_bytes());
}

fn scale_fixed_by_level(value: u32, level: ModLevel) -> u32 {
    let mult = match level {
        ModLevel::Low => 90,
        ModLevel::Med => 100,
        ModLevel::High => 110,
    };
    ((value as u64 * mult as u64) / 100) as u32
}

fn quantize_f32(value: f32, scale: f32) -> i32 {
    (value * scale).round() as i32
}

#[cfg(all(
    test,
    feature = "biophys",
    feature = "biophys-l4",
    feature = "biophys-l4-synapses",
    feature = "biophys-l4-amygdala"
))]
mod tests {
    use super::*;

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

    #[test]
    fn determinism_sequence_and_digests() {
        let inputs = vec![
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

        let run_sequence = |inputs: &[AmyInput]| -> Vec<(AmyOutput, [u8; 32])> {
            let mut circuit = AmygdalaL4Microcircuit::new(CircuitConfig::default());
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
    fn dlp_secret_activates_exfil_quickly() {
        let mut circuit = AmygdalaL4Microcircuit::new(CircuitConfig::default());
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
        let mut circuit = AmygdalaL4Microcircuit::new(CircuitConfig::default());
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
    fn cross_sensitization_synapses_are_directional() {
        let circuit = AmygdalaL4Microcircuit::new(CircuitConfig::default());
        let (integrity_start, integrity_end) = AmygdalaL4Microcircuit::pool_bounds(IDX_INTEGRITY);
        let (exfil_start, exfil_end) = AmygdalaL4Microcircuit::pool_bounds(IDX_EXFIL);

        for synapse in &circuit.synapses {
            let pre = synapse.pre_neuron as usize;
            let post = synapse.post_neuron as usize;
            let within_integrity = pre >= integrity_start && pre < integrity_end;
            let within_exfil = post >= exfil_start && post < exfil_end;
            if within_integrity && within_exfil {
                assert_eq!(synapse.delay_steps, 2);
                assert_eq!(synapse.kind, SynKind::AMPA);
            }
        }
    }

    #[test]
    fn bounded_state_values() {
        let mut circuit = AmygdalaL4Microcircuit::new(CircuitConfig::default());
        let input = AmyInput {
            integrity: IntegrityState::Fail,
            dlp_secret_present: true,
            policy_pressure: LevelClass::High,
            tool_anomaly_present: true,
            ..base_input()
        };

        for _ in 0..10 {
            circuit.step(&input, 0);
        }

        assert!(circuit
            .state
            .pool_acc
            .iter()
            .all(|&acc| acc >= 0 && acc <= ACCUMULATOR_MAX));
        assert!(circuit
            .state
            .latch_steps
            .iter()
            .all(|&latch| latch <= LATCH_STEPS_MAX));
        assert!(circuit.state.last_spike_count_total <= NEURON_COUNT * SUBSTEPS);
    }
}
