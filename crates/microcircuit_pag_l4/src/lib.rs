#![forbid(unsafe_code)]

use biophys_channels::{Leak, NaK};
use biophys_compartmental_solver::{CompartmentChannels, L4Solver, L4State};
use biophys_core::{CompartmentId, ModChannel, ModLevel, ModulatorField, NeuronId};
use biophys_event_queue_l4::SpikeEventQueueL4;
use biophys_morphology::{Compartment, CompartmentKind, NeuronMorphology};
use biophys_plasticity_l4::{plasticity_snapshot_digest, LearningMode, StdpConfig, StdpTrace};
use biophys_synapses_l4::{
    apply_stdp_updates, decay_k, f32_to_fixed_u32, max_synapse_g_fixed, StpParamsL4, StpStateL4,
    SynKind, SynapseAccumulator, SynapseL4, SynapseState,
};
use dbm_core::{DbmModule, IntegrityState, LevelClass, ReasonSet, ThreatVector};
use microcircuit_core::{CircuitConfig, MicrocircuitBackend};
use microcircuit_pag_stub::{DefensePattern, PagInput, PagOutput};

const POOL_COUNT: usize = 4;
const POOL_SIZE: usize = 2;
const EXCITATORY_COUNT: usize = POOL_COUNT * POOL_SIZE;
const INHIBITORY_COUNT: usize = 1;
const NEURON_COUNT: usize = EXCITATORY_COUNT + INHIBITORY_COUNT;

const IDX_DP1: usize = 0;
const IDX_DP2: usize = 1;
const IDX_DP3: usize = 2;
const IDX_DP4: usize = 3;

const COMPARTMENT_COUNT: usize = 3;
const SUBSTEPS: usize = 10;
const DT_MS: f32 = 0.1;
const CLAMP_MIN: f32 = -120.0;
const CLAMP_MAX: f32 = 60.0;
const THRESHOLD_MV: f32 = -20.0;

const CURRENT_BASELINE: f32 = 4.0;
const CURRENT_UNLOCK: f32 = 2.0;
const CURRENT_MEDIUM: f32 = 7.0;
const CURRENT_STRONG: f32 = 14.0;

const AMPA_G_MAX: f32 = 4.0;
const AMPA_E_REV: f32 = 0.0;
const AMPA_TAU_RISE_MS: f32 = 0.0;
const AMPA_TAU_DECAY_MS: f32 = 8.0;

const GABA_G_MAX: f32 = 6.0;
const GABA_E_REV: f32 = -70.0;
const GABA_TAU_RISE_MS: f32 = 0.0;
const GABA_TAU_DECAY_MS: f32 = 10.0;

const GABA_DOMINANCE_LIGHT: f32 = 4.0;
const GABA_DOMINANCE_MODERATE: f32 = 6.0;
const GABA_DOMINANCE_STRONG: f32 = 8.0;

const MAX_EVENTS_PER_STEP: usize = 512;
const ACCUMULATOR_MAX: i32 = 100;
const ACCUMULATOR_DECAY: i32 = 5;
const ACCUMULATOR_GAIN: i32 = 20;
const LATCH_MAX: u8 = 20;
const LATCH_THRESHOLD: i32 = 60;
const UNLOCK_REQUIRED_TICKS: u8 = 3;

#[derive(Debug, Clone)]
struct L4Neuron {
    solver: L4Solver,
    state: L4State,
    last_soma_v: f32,
}

#[derive(Debug, Clone)]
struct PagL4State {
    tick_count: u64,
    step_count: u64,
    pool_acc: [i32; POOL_COUNT],
    last_pool_spikes: [usize; POOL_COUNT],
    last_spike_count_total: usize,
    winner: usize,
    latch_steps: u8,
    calm_ticks: u8,
}

impl Default for PagL4State {
    fn default() -> Self {
        Self {
            tick_count: 0,
            step_count: 0,
            pool_acc: [0; POOL_COUNT],
            last_pool_spikes: [0; POOL_COUNT],
            last_spike_count_total: 0,
            winner: IDX_DP4,
            latch_steps: 0,
            calm_ticks: 0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct DriveState {
    integrity_fail: bool,
    exfil_any: bool,
    exfil_high: bool,
}

#[derive(Debug, Clone)]
pub struct PagL4Microcircuit {
    _config: CircuitConfig,
    neurons: Vec<L4Neuron>,
    synapses: Vec<SynapseL4>,
    syn_states: Vec<SynapseState>,
    syn_g_max_eff: Vec<u32>,
    syn_decay: Vec<u16>,
    syn_stp_params_eff: Vec<StpParamsL4>,
    pre_index: Vec<Vec<usize>>,
    queue: SpikeEventQueueL4,
    state: PagL4State,
    current_modulators: ModulatorField,
    stdp_config: StdpConfig,
    stdp_traces: Vec<StdpTrace>,
    stdp_spike_flags: Vec<bool>,
    learning_enabled: bool,
    in_replay_mode: bool,
}

impl PagL4Microcircuit {
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
        let stdp_traces = vec![StdpTrace::default(); NEURON_COUNT];
        let stdp_spike_flags = vec![false; NEURON_COUNT];
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
            state: PagL4State::default(),
            current_modulators,
            stdp_config: StdpConfig::default(),
            stdp_traces,
            stdp_spike_flags,
            learning_enabled: false,
            in_replay_mode: false,
        }
    }

    fn pool_bounds(pool: usize) -> (usize, usize) {
        let start = pool * POOL_SIZE;
        (start, start + POOL_SIZE)
    }

    fn severity_index(pool: usize) -> u8 {
        match pool {
            IDX_DP4 => 0,
            IDX_DP1 => 1,
            IDX_DP2 => 2,
            IDX_DP3 => 3,
            _ => 0,
        }
    }

    fn pattern_from_pool(pool: usize) -> DefensePattern {
        match pool {
            IDX_DP1 => DefensePattern::DP1_FREEZE,
            IDX_DP2 => DefensePattern::DP2_QUARANTINE,
            IDX_DP3 => DefensePattern::DP3_FORENSIC,
            _ => DefensePattern::DP4_CONTAINED_CONTINUE,
        }
    }

    fn drives_from_input(input: &PagInput) -> (DriveState, [f32; POOL_COUNT]) {
        let integrity_fail = matches!(input.integrity, IntegrityState::Fail);
        let exfil_any = input.vectors.contains(&ThreatVector::Exfil);
        let exfil_high = exfil_any && input.threat == LevelClass::High;
        let tool_side_effects = input.vectors.contains(&ThreatVector::ToolSideEffects);
        let arousal_high = input.modulators.na == ModLevel::High;

        let mut pool_drive = [0.0_f32; POOL_COUNT];

        if integrity_fail || exfil_high {
            pool_drive[IDX_DP3] = CURRENT_STRONG;
        }

        if input.threat == LevelClass::High && !integrity_fail {
            pool_drive[IDX_DP2] = pool_drive[IDX_DP2].max(CURRENT_STRONG);
        }

        if tool_side_effects {
            pool_drive[IDX_DP2] = pool_drive[IDX_DP2].max(CURRENT_MEDIUM);
        }

        if input.stability == LevelClass::High && arousal_high {
            pool_drive[IDX_DP1] = CURRENT_MEDIUM;
        }

        let no_threats = !integrity_fail
            && input.threat == LevelClass::Low
            && !exfil_any
            && input.vectors.is_empty();
        if no_threats {
            pool_drive[IDX_DP4] = CURRENT_BASELINE;
        }
        if input.unlock_present {
            pool_drive[IDX_DP4] += CURRENT_UNLOCK;
        }

        (
            DriveState {
                integrity_fail,
                exfil_any,
                exfil_high,
            },
            pool_drive,
        )
    }

    fn build_inputs(input: &PagInput) -> (DriveState, [f32; NEURON_COUNT]) {
        let (drive_state, pool_drive) = Self::drives_from_input(input);
        let mut currents = [0.0_f32; NEURON_COUNT];
        for (pool, drive) in pool_drive.iter().enumerate().take(POOL_COUNT) {
            let (start, end) = Self::pool_bounds(pool);
            for current in currents.iter_mut().take(end).skip(start) {
                *current += *drive;
            }
        }
        (drive_state, currents)
    }

    fn update_modulators(&mut self, input: &PagInput) {
        self.current_modulators = input.modulators;
        for (idx, synapse) in self.synapses.iter().enumerate() {
            self.syn_g_max_eff[idx] = synapse.effective_g_max_fixed(self.current_modulators);
            self.syn_stp_params_eff[idx] = synapse.stp_effective_params(self.current_modulators);
        }
    }

    fn set_learning_context(&mut self, in_replay: bool, mods: ModulatorField, reward_block: bool) {
        self.in_replay_mode = in_replay;
        if !cfg!(feature = "biophys-l4-plasticity") {
            self.learning_enabled = false;
            return;
        }
        if !self.stdp_config.enabled {
            self.learning_enabled = false;
            return;
        }
        let mode_allowed = match self.stdp_config.learning_mode {
            LearningMode::OFF => false,
            LearningMode::REPLAY_ONLY => in_replay,
            LearningMode::ALWAYS => true,
        };
        let da_allowed = matches!(mods.da, ModLevel::Med | ModLevel::High);
        self.learning_enabled = mode_allowed && da_allowed && !reward_block;
    }

    fn substep(&mut self, injected_currents: &[f32; NEURON_COUNT]) -> Vec<usize> {
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
            let mut input = [0.0_f32; COMPARTMENT_COUNT];
            input[0] = injected_currents[idx];
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

        self.update_stdp_traces(&spikes);
        self.apply_stdp_updates(&spikes);

        self.state.step_count = self.state.step_count.saturating_add(1);
        spikes
    }

    fn update_stdp_traces(&mut self, spikes: &[usize]) {
        for trace in &mut self.stdp_traces {
            trace.decay_tick(
                self.stdp_config.tau_plus_steps,
                self.stdp_config.tau_minus_steps,
            );
        }
        for &idx in spikes {
            if let Some(trace) = self.stdp_traces.get_mut(idx) {
                trace.on_pre_spike();
                trace.on_post_spike();
            }
        }
    }

    fn apply_stdp_updates(&mut self, spikes: &[usize]) {
        if !self.learning_enabled || spikes.is_empty() {
            return;
        }
        self.stdp_spike_flags.fill(false);
        for &idx in spikes {
            if let Some(flag) = self.stdp_spike_flags.get_mut(idx) {
                *flag = true;
            }
        }
        apply_stdp_updates(
            &mut self.synapses,
            &self.stdp_spike_flags,
            &self.stdp_traces,
            self.stdp_config,
        );
        for (idx, synapse) in self.synapses.iter().enumerate() {
            self.syn_g_max_eff[idx] = synapse.effective_g_max_fixed(self.current_modulators);
        }
    }

    pub fn plasticity_snapshot_digest(&self) -> [u8; 32] {
        let g_max_values = self
            .synapses
            .iter()
            .map(|synapse| synapse.g_max_base_q)
            .collect::<Vec<_>>();
        plasticity_snapshot_digest(self.state.step_count, &g_max_values)
    }

    fn update_pool_accumulators(&mut self, spike_counts: &[usize; NEURON_COUNT]) {
        let mut pool_counts = [0usize; POOL_COUNT];
        for (idx, count) in spike_counts.iter().enumerate().take(EXCITATORY_COUNT) {
            let pool = idx / POOL_SIZE;
            pool_counts[pool] = pool_counts[pool].saturating_add(*count);
        }

        self.state.last_pool_spikes = pool_counts;
        self.state.last_spike_count_total = spike_counts.iter().sum();

        for (idx, acc) in self.state.pool_acc.iter_mut().enumerate() {
            let delta = pool_counts[idx] as i32 * ACCUMULATOR_GAIN - ACCUMULATOR_DECAY;
            let next = (*acc + delta).clamp(0, ACCUMULATOR_MAX);
            *acc = next;
        }
    }

    fn select_winner(&self) -> usize {
        let max_score = self.state.pool_acc.iter().copied().max().unwrap_or(0);
        let order = [IDX_DP3, IDX_DP2, IDX_DP1, IDX_DP4];
        for candidate in order {
            if self.state.pool_acc[candidate] == max_score {
                return candidate;
            }
        }
        IDX_DP4
    }

    fn build_reason_codes(input: &PagInput, pool: usize) -> ReasonSet {
        let mut reason_codes = ReasonSet::default();

        reason_codes.insert(match pool {
            IDX_DP1 => "RC.RX.ACTION.FREEZE",
            IDX_DP2 => "RC.RX.ACTION.QUARANTINE",
            IDX_DP3 => "RC.RX.ACTION.FORENSIC",
            _ => "RC.RX.ACTION.CONTAINED",
        });

        if matches!(input.integrity, IntegrityState::Fail)
            || input.vectors.contains(&ThreatVector::IntegrityCompromise)
        {
            reason_codes.insert("RC.TH.INTEGRITY_COMPROMISE");
        }

        for vector in &input.vectors {
            match vector {
                ThreatVector::Exfil => reason_codes.insert("RC.TH.EXFIL.HIGH_CONFIDENCE"),
                ThreatVector::Probing => reason_codes.insert("RC.TH.POLICY_PROBING"),
                ThreatVector::IntegrityCompromise => {
                    reason_codes.insert("RC.TH.INTEGRITY_COMPROMISE")
                }
                ThreatVector::RuntimeEscape => reason_codes.insert("RC.TH.RUNTIME_ESCAPE"),
                ThreatVector::ToolSideEffects => reason_codes.insert("RC.TH.TOOL_SIDE_EFFECTS"),
            }
        }

        reason_codes
    }
}

impl MicrocircuitBackend<PagInput, PagOutput> for PagL4Microcircuit {
    fn step(&mut self, input: &PagInput, _now_ms: u64) -> PagOutput {
        self.state.tick_count = self.state.tick_count.saturating_add(1);
        self.update_modulators(input);
        self.set_learning_context(false, self.current_modulators, false);
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
        let (drive_state, currents) = Self::build_inputs(input);

        let mut spike_counts = [0usize; NEURON_COUNT];
        for _ in 0..SUBSTEPS {
            let spikes = self.substep(&currents);
            for spike in spikes {
                spike_counts[spike] = spike_counts[spike].saturating_add(1);
            }
        }

        self.update_pool_accumulators(&spike_counts);

        if drive_state.integrity_fail || drive_state.exfil_any {
            self.state.pool_acc[IDX_DP3] = ACCUMULATOR_MAX;
        }
        if input.threat == LevelClass::High && !drive_state.integrity_fail {
            self.state.pool_acc[IDX_DP2] = self.state.pool_acc[IDX_DP2].max(LATCH_THRESHOLD);
        }

        if drive_state.integrity_fail
            || drive_state.exfil_high
            || self.state.pool_acc[IDX_DP3] >= LATCH_THRESHOLD
            || self.state.pool_acc[IDX_DP2] >= LATCH_THRESHOLD
        {
            self.state.latch_steps = LATCH_MAX;
        } else if self.state.latch_steps > 0 {
            self.state.latch_steps = self.state.latch_steps.saturating_sub(1);
        }

        let calm_condition = input.unlock_present
            && input.integrity == IntegrityState::Ok
            && !drive_state.exfil_any
            && input.threat != LevelClass::High;
        if calm_condition {
            self.state.calm_ticks = self.state.calm_ticks.saturating_add(1);
        } else {
            self.state.calm_ticks = 0;
        }

        let allow_relax = calm_condition && self.state.calm_ticks >= UNLOCK_REQUIRED_TICKS;
        if allow_relax {
            self.state.pool_acc[IDX_DP3] = 0;
            self.state.pool_acc[IDX_DP2] = 0;
        }

        let candidate = self.select_winner();
        let previous = self.state.winner;
        let mut winner = candidate;

        if self.state.latch_steps > 0
            && Self::severity_index(candidate) < Self::severity_index(previous)
            && !allow_relax
        {
            winner = previous;
        } else if allow_relax && Self::severity_index(candidate) < Self::severity_index(previous) {
            self.state.latch_steps = 0;
        }

        self.state.winner = winner;

        let pattern = Self::pattern_from_pool(winner);
        let pattern_latched =
            (winner == IDX_DP2 || winner == IDX_DP3) && self.state.latch_steps > 0;
        let reason_codes = Self::build_reason_codes(input, winner);

        PagOutput {
            pattern,
            pattern_latched,
            reason_codes,
        }
    }

    fn snapshot_digest(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"UCF:BIO:L4:PAG:SNAP");
        update_u64(&mut hasher, self.state.tick_count);
        update_u64(&mut hasher, self.state.step_count);
        update_u32(&mut hasher, self.state.winner as u32);
        update_u32(&mut hasher, self.state.latch_steps as u32);
        update_u32(&mut hasher, self.state.calm_ticks as u32);
        for value in self.state.pool_acc {
            update_i32(&mut hasher, value);
        }
        for value in self.state.last_pool_spikes {
            update_u32(&mut hasher, value as u32);
        }
        update_u32(&mut hasher, self.state.last_spike_count_total as u32);
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
        update_u32(&mut hasher, mod_level_code(self.current_modulators.na));
        update_u32(&mut hasher, mod_level_code(self.current_modulators.da));
        update_u32(&mut hasher, mod_level_code(self.current_modulators.ht));
        update_u64(&mut hasher, self.queue.dropped_event_count);
        *hasher.finalize().as_bytes()
    }

    fn config_digest(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"UCF:BIO:L4:PAG:CFG");
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

impl DbmModule for PagL4Microcircuit {
    type Input = PagInput;
    type Output = PagOutput;

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
            capacitance: 1.0,
            axial_resistance: 120.0,
        },
        Compartment {
            id: CompartmentId(1),
            parent: Some(CompartmentId(0)),
            kind: CompartmentKind::Dendrite,
            capacitance: 1.0,
            axial_resistance: 150.0,
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
        let (start, end) = PagL4Microcircuit::pool_bounds(pool);
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
                    mod_channel: ModChannel::Na,
                    g_max_base_q: f32_to_fixed_u32(AMPA_G_MAX),
                    g_max_min_q: 0,
                    g_max_max_q: max_synapse_g_fixed(),
                    e_rev: AMPA_E_REV,
                    tau_rise_ms: AMPA_TAU_RISE_MS,
                    tau_decay_ms: AMPA_TAU_DECAY_MS,
                    delay_steps: 1,
                    stp_params,
                    stp_state,
                    stdp_enabled: true,
                    stdp_trace: StdpTrace::default(),
                });
            }
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
            g_max_min_q: 0,
            g_max_max_q: max_synapse_g_fixed(),
            e_rev: AMPA_E_REV,
            tau_rise_ms: AMPA_TAU_RISE_MS,
            tau_decay_ms: AMPA_TAU_DECAY_MS,
            delay_steps: 1,
            stp_params,
            stp_state,
            stdp_enabled: true,
            stdp_trace: StdpTrace::default(),
        });
    }

    let inhibitory = EXCITATORY_COUNT;
    for post in 0..EXCITATORY_COUNT {
        let (stp_params, stp_state) = disabled_stp();
        synapses.push(SynapseL4 {
            pre_neuron: inhibitory as u32,
            post_neuron: post as u32,
            post_compartment: 0,
            kind: SynKind::GABA,
            mod_channel: ModChannel::Ht,
            g_max_base_q: f32_to_fixed_u32(GABA_G_MAX),
            g_max_min_q: 0,
            g_max_max_q: max_synapse_g_fixed(),
            e_rev: GABA_E_REV,
            tau_rise_ms: GABA_TAU_RISE_MS,
            tau_decay_ms: GABA_TAU_DECAY_MS,
            delay_steps: 1,
            stp_params,
            stp_state,
            stdp_enabled: false,
            stdp_trace: StdpTrace::default(),
        });
    }

    let dp4_start = IDX_DP4 * POOL_SIZE;
    let dp4_end = dp4_start + POOL_SIZE;
    let dp1_start = IDX_DP1 * POOL_SIZE;
    let dp1_end = dp1_start + POOL_SIZE;
    let dp2_start = IDX_DP2 * POOL_SIZE;
    let dp2_end = dp2_start + POOL_SIZE;
    let dp3_start = IDX_DP3 * POOL_SIZE;
    let dp3_end = dp3_start + POOL_SIZE;

    for pre in dp3_start..dp3_end {
        for post in dp4_start..dp4_end {
            let (stp_params, stp_state) = disabled_stp();
            synapses.push(SynapseL4 {
                pre_neuron: pre as u32,
                post_neuron: post as u32,
                post_compartment: 0,
                kind: SynKind::GABA,
                mod_channel: ModChannel::Ht,
                g_max_base_q: f32_to_fixed_u32(GABA_DOMINANCE_STRONG),
                g_max_min_q: 0,
                g_max_max_q: max_synapse_g_fixed(),
                e_rev: GABA_E_REV,
                tau_rise_ms: GABA_TAU_RISE_MS,
                tau_decay_ms: GABA_TAU_DECAY_MS,
                delay_steps: 1,
                stp_params,
                stp_state,
                stdp_enabled: false,
                stdp_trace: StdpTrace::default(),
            });
        }
    }

    for pre in dp2_start..dp2_end {
        for post in dp4_start..dp4_end {
            let (stp_params, stp_state) = disabled_stp();
            synapses.push(SynapseL4 {
                pre_neuron: pre as u32,
                post_neuron: post as u32,
                post_compartment: 0,
                kind: SynKind::GABA,
                mod_channel: ModChannel::Ht,
                g_max_base_q: f32_to_fixed_u32(GABA_DOMINANCE_MODERATE),
                g_max_min_q: 0,
                g_max_max_q: max_synapse_g_fixed(),
                e_rev: GABA_E_REV,
                tau_rise_ms: GABA_TAU_RISE_MS,
                tau_decay_ms: GABA_TAU_DECAY_MS,
                delay_steps: 1,
                stp_params,
                stp_state,
                stdp_enabled: false,
                stdp_trace: StdpTrace::default(),
            });
        }
    }

    for pre in dp1_start..dp1_end {
        for post in dp4_start..dp4_end {
            let (stp_params, stp_state) = disabled_stp();
            synapses.push(SynapseL4 {
                pre_neuron: pre as u32,
                post_neuron: post as u32,
                post_compartment: 0,
                kind: SynKind::GABA,
                mod_channel: ModChannel::Ht,
                g_max_base_q: f32_to_fixed_u32(GABA_DOMINANCE_LIGHT),
                g_max_min_q: 0,
                g_max_max_q: max_synapse_g_fixed(),
                e_rev: GABA_E_REV,
                tau_rise_ms: GABA_TAU_RISE_MS,
                tau_decay_ms: GABA_TAU_DECAY_MS,
                delay_steps: 1,
                stp_params,
                stp_state,
                stdp_enabled: false,
                stdp_trace: StdpTrace::default(),
            });
        }
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
    let mult = mod_level_code(level);
    ((value as u64 * mult as u64) / 100) as u32
}

fn mod_level_code(level: ModLevel) -> u32 {
    match level {
        ModLevel::Low => 90,
        ModLevel::Med => 100,
        ModLevel::High => 110,
    }
}

fn quantize_f32(value: f32, scale: f32) -> i32 {
    (value * scale).round() as i32
}

#[cfg(all(
    test,
    feature = "biophys",
    feature = "biophys-l4",
    feature = "biophys-l4-synapses",
    feature = "biophys-l4-pag"
))]
mod tests {
    use super::*;
    use biophys_core::ModulatorField;
    use dbm_core::CooldownClass;

    fn base_input() -> PagInput {
        PagInput {
            integrity: IntegrityState::Ok,
            threat: LevelClass::Low,
            vectors: Vec::new(),
            unlock_present: false,
            stability: LevelClass::Low,
            serotonin_cooldown: CooldownClass::Base,
            modulators: ModulatorField::default(),
        }
    }

    fn severity(pattern: DefensePattern) -> u8 {
        match pattern {
            DefensePattern::DP4_CONTAINED_CONTINUE => 0,
            DefensePattern::DP1_FREEZE => 1,
            DefensePattern::DP2_QUARANTINE => 2,
            DefensePattern::DP3_FORENSIC => 3,
        }
    }

    #[test]
    fn determinism_for_sequence() {
        let inputs = vec![
            PagInput {
                stability: LevelClass::High,
                modulators: ModulatorField {
                    na: ModLevel::High,
                    ..ModulatorField::default()
                },
                ..base_input()
            },
            PagInput {
                threat: LevelClass::Med,
                ..base_input()
            },
            PagInput {
                threat: LevelClass::High,
                ..base_input()
            },
            PagInput {
                vectors: vec![ThreatVector::ToolSideEffects],
                ..base_input()
            },
        ];

        let run = |inputs: &[PagInput]| {
            let mut circuit = PagL4Microcircuit::new(CircuitConfig::default());
            inputs
                .iter()
                .map(|input| (circuit.step(input, 0).pattern, circuit.snapshot_digest()))
                .collect::<Vec<_>>()
        };

        assert_eq!(run(&inputs), run(&inputs));
    }

    #[test]
    fn critical_invariants_hold() {
        let mut circuit = PagL4Microcircuit::new(CircuitConfig::default());
        let integrity = PagInput {
            integrity: IntegrityState::Fail,
            ..base_input()
        };
        let exfil = PagInput {
            vectors: vec![ThreatVector::Exfil],
            threat: LevelClass::Med,
            ..base_input()
        };
        let threat = PagInput {
            threat: LevelClass::High,
            ..base_input()
        };

        let integrity_out = circuit.step(&integrity, 0);
        assert_eq!(integrity_out.pattern, DefensePattern::DP3_FORENSIC);

        let exfil_out = circuit.step(&exfil, 0);
        assert_eq!(exfil_out.pattern, DefensePattern::DP3_FORENSIC);

        let threat_out = circuit.step(&threat, 0);
        assert!(severity(threat_out.pattern) >= severity(DefensePattern::DP2_QUARANTINE));
    }

    #[test]
    fn latch_prevents_immediate_relax() {
        let mut circuit = PagL4Microcircuit::new(CircuitConfig::default());
        let forensic = PagInput {
            integrity: IntegrityState::Fail,
            ..base_input()
        };
        let calm = base_input();

        circuit.step(&forensic, 0);
        let first = circuit.step(&calm, 0);

        assert_eq!(first.pattern, DefensePattern::DP3_FORENSIC);
    }

    #[test]
    fn unlock_recovery_requires_multiple_ticks() {
        let mut circuit = PagL4Microcircuit::new(CircuitConfig::default());
        let forensic = PagInput {
            integrity: IntegrityState::Fail,
            ..base_input()
        };
        let calm_unlock = PagInput {
            unlock_present: true,
            ..base_input()
        };

        circuit.step(&forensic, 0);

        let first = circuit.step(&calm_unlock, 0);
        let second = circuit.step(&calm_unlock, 0);
        let third = circuit.step(&calm_unlock, 0);

        assert_eq!(first.pattern, DefensePattern::DP3_FORENSIC);
        assert_eq!(second.pattern, DefensePattern::DP3_FORENSIC);
        assert_eq!(third.pattern, DefensePattern::DP4_CONTAINED_CONTINUE);
    }
}
