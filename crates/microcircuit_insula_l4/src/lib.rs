#![forbid(unsafe_code)]

use biophys_channels::{Leak, NaK};
use biophys_compartmental_solver::{CompartmentChannels, L4Solver, L4State};
use biophys_core::{CompartmentId, ModChannel, ModLevel, ModulatorField, NeuronId};
use biophys_event_queue_l4::SpikeEventQueueL4;
use biophys_homeostasis_l4::{
    homeostasis_tick, scale_g_max_fixed, HomeoMode, HomeostasisConfig, HomeostasisState,
};
use biophys_morphology::{Compartment, CompartmentKind, NeuronMorphology};
use biophys_plasticity_l4::{plasticity_snapshot_digest, LearningMode, StdpConfig, StdpTrace};
use biophys_synapses_l4::{
    apply_stdp_updates, decay_k, f32_to_fixed_u32, max_synapse_g_fixed, StpParamsL4, StpStateL4,
    SynKind, SynapseAccumulator, SynapseL4, SynapseState,
};
use dbm_core::{IntegrityState, IsvSnapshot, LevelClass, ReasonSet, ThreatVector};
use microcircuit_core::{CircuitConfig, MicrocircuitBackend};
use microcircuit_insula_stub::InsulaInput;

const POOL_COUNT: usize = 6;
const POOL_SIZE: usize = 2;
const EXCITATORY_COUNT: usize = POOL_COUNT * POOL_SIZE;
const INHIBITORY_COUNT: usize = 2;
const NEURON_COUNT: usize = EXCITATORY_COUNT + INHIBITORY_COUNT;

const IDX_INTEGRITY: usize = 0;
const IDX_THREAT: usize = 1;
const IDX_STABILITY: usize = 2;
const IDX_AROUSAL: usize = 3;
const IDX_POLICY: usize = 4;
const IDX_PROGRESS: usize = 5;

const COMPARTMENT_COUNT: usize = 3;
const SUBSTEPS: usize = 10;
const DT_MS: f32 = 0.1;
const CLAMP_MIN: f32 = -120.0;
const CLAMP_MAX: f32 = 60.0;
const THRESHOLD_MV: f32 = -20.0;

const CURRENT_STRONG: f32 = 2.0;
const CURRENT_MED: f32 = 1.0;
const CURRENT_LOW: f32 = 0.4;
const CURRENT_BASE: f32 = 0.1;

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
const LATCH_HIGH: i32 = 60;
const LATCH_LOW: i32 = 40;
const LATCH_STEPS_MAX: u8 = 10;

#[derive(Debug, Clone)]
struct L4Neuron {
    solver: L4Solver,
    state: L4State,
    last_soma_v: f32,
}

#[derive(Debug, Clone)]
struct InsulaL4State {
    tick_count: u64,
    step_count: u64,
    pool_acc: [i32; POOL_COUNT],
    latch_steps: [u8; POOL_COUNT],
}

impl Default for InsulaL4State {
    fn default() -> Self {
        Self {
            tick_count: 0,
            step_count: 0,
            pool_acc: [0; POOL_COUNT],
            latch_steps: [0; POOL_COUNT],
        }
    }
}

#[derive(Debug, Clone)]
pub struct InsulaL4Microcircuit {
    _config: CircuitConfig,
    neurons: Vec<L4Neuron>,
    synapses: Vec<SynapseL4>,
    syn_states: Vec<SynapseState>,
    syn_g_max_eff: Vec<u32>,
    syn_decay: Vec<u16>,
    syn_stp_params_eff: Vec<StpParamsL4>,
    pre_index: Vec<Vec<usize>>,
    queue: SpikeEventQueueL4,
    state: InsulaL4State,
    current_modulators: ModulatorField,
    stdp_config: StdpConfig,
    stdp_traces: Vec<StdpTrace>,
    stdp_spike_flags: Vec<bool>,
    learning_enabled: bool,
    in_replay_mode: bool,
    homeostasis_config: HomeostasisConfig,
    homeostasis_state: HomeostasisState,
}

impl InsulaL4Microcircuit {
    pub fn new(config: CircuitConfig) -> Self {
        let neurons = (0..NEURON_COUNT)
            .map(|idx| build_neuron(idx as u32))
            .collect::<Vec<_>>();
        let synapses = build_synapses();
        let syn_states = vec![SynapseState::default(); synapses.len()];
        let current_modulators = ModulatorField::default();
        let mut homeostasis_config = HomeostasisConfig::default();
        if cfg!(feature = "biophys-l4-homeostasis") {
            homeostasis_config.enabled = true;
            homeostasis_config.mode = HomeoMode::REPLAY_ONLY;
        }
        let homeostasis_state = HomeostasisState::default();
        let syn_g_max_eff = synapses
            .iter()
            .map(|synapse| {
                Self::scaled_g_max_fixed(synapse, current_modulators, homeostasis_state.scale_q)
            })
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
        let mut stdp_config = StdpConfig::default();
        if cfg!(feature = "biophys-l4-plasticity") {
            stdp_config.enabled = true;
            stdp_config.learning_mode = LearningMode::REPLAY_ONLY;
        }
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
            state: InsulaL4State::default(),
            current_modulators,
            stdp_config,
            stdp_traces,
            stdp_spike_flags,
            learning_enabled: false,
            in_replay_mode: false,
            homeostasis_config,
            homeostasis_state,
        }
    }

    fn pool_bounds(pool: usize) -> (usize, usize) {
        let start = pool * POOL_SIZE;
        (start, start + POOL_SIZE)
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

    fn apply_pool_soma_current(
        currents: &mut [[f32; COMPARTMENT_COUNT]; NEURON_COUNT],
        pool: usize,
        soma: f32,
    ) {
        let (start, end) = Self::pool_bounds(pool);
        for neuron in currents.iter_mut().take(end).skip(start) {
            neuron[0] += soma;
        }
    }

    fn build_inputs(input: &InsulaInput) -> [[f32; COMPARTMENT_COUNT]; NEURON_COUNT] {
        let mut currents = [[0.0_f32; COMPARTMENT_COUNT]; NEURON_COUNT];

        let integrity_current = match input.integrity {
            IntegrityState::Fail => CURRENT_STRONG,
            IntegrityState::Degraded => CURRENT_MED,
            IntegrityState::Ok => CURRENT_BASE,
        };
        Self::apply_pool_current(
            &mut currents,
            IDX_INTEGRITY,
            integrity_current,
            integrity_current,
        );

        let threat_current = match input.threat {
            LevelClass::High => CURRENT_STRONG,
            LevelClass::Med => CURRENT_MED,
            LevelClass::Low => CURRENT_BASE,
        };
        Self::apply_pool_soma_current(&mut currents, IDX_THREAT, threat_current);
        if input.threat_vectors.iter().any(|vector| {
            matches!(
                vector,
                ThreatVector::Exfil | ThreatVector::IntegrityCompromise
            )
        }) {
            Self::apply_pool_current(&mut currents, IDX_THREAT, 0.0, CURRENT_LOW);
        }

        let stability_current = match input.stability {
            LevelClass::High => CURRENT_STRONG,
            LevelClass::Med => CURRENT_MED,
            LevelClass::Low => CURRENT_BASE,
        };
        Self::apply_pool_soma_current(&mut currents, IDX_STABILITY, stability_current);

        let arousal_current = match input.arousal {
            LevelClass::High => CURRENT_STRONG,
            LevelClass::Med => CURRENT_MED,
            LevelClass::Low => CURRENT_BASE,
        };
        Self::apply_pool_soma_current(&mut currents, IDX_AROUSAL, arousal_current);

        let policy_current = match input.policy_pressure {
            LevelClass::High => CURRENT_STRONG,
            LevelClass::Med => CURRENT_MED,
            LevelClass::Low => CURRENT_BASE,
        };
        Self::apply_pool_soma_current(&mut currents, IDX_POLICY, policy_current);

        let progress_current = if input.reward_block {
            CURRENT_BASE
        } else {
            match input.progress {
                LevelClass::High => CURRENT_MED,
                LevelClass::Med => CURRENT_LOW,
                LevelClass::Low => CURRENT_BASE,
            }
        };
        Self::apply_pool_soma_current(&mut currents, IDX_PROGRESS, progress_current);

        currents
    }

    fn update_pool_accumulators(&mut self, spike_counts: &[usize; NEURON_COUNT]) {
        let mut pool_counts = [0usize; POOL_COUNT];
        for (idx, count) in spike_counts.iter().enumerate().take(EXCITATORY_COUNT) {
            let pool = idx / POOL_SIZE;
            pool_counts[pool] = pool_counts[pool].saturating_add(*count);
        }

        for (acc, &count) in self.state.pool_acc.iter_mut().zip(pool_counts.iter()) {
            let delta = (count as i32).saturating_mul(ACCUMULATOR_GAIN);
            *acc = (*acc + delta - ACCUMULATOR_DECAY).clamp(0, ACCUMULATOR_MAX);
        }
    }

    fn update_latches(&mut self) {
        for (idx, acc) in self.state.pool_acc.iter().enumerate() {
            if idx != IDX_INTEGRITY && idx != IDX_THREAT {
                continue;
            }
            let latch = &mut self.state.latch_steps[idx];
            if *acc >= LATCH_HIGH {
                *latch = LATCH_STEPS_MAX;
            } else if *acc < LATCH_LOW {
                *latch = latch.saturating_sub(1);
            }
        }
    }

    fn level_from_acc(acc: i32) -> LevelClass {
        if acc >= 70 {
            LevelClass::High
        } else if acc >= 40 {
            LevelClass::Med
        } else {
            LevelClass::Low
        }
    }

    fn integrity_from_acc(acc: i32, input: IntegrityState) -> IntegrityState {
        if acc >= 80 || input == IntegrityState::Fail {
            IntegrityState::Fail
        } else if acc >= 50 || input == IntegrityState::Degraded {
            IntegrityState::Degraded
        } else {
            IntegrityState::Ok
        }
    }

    fn severity(level: LevelClass) -> u8 {
        match level {
            LevelClass::Low => 0,
            LevelClass::Med => 1,
            LevelClass::High => 2,
        }
    }

    fn integrity_severity(state: IntegrityState) -> u8 {
        match state {
            IntegrityState::Ok => 0,
            IntegrityState::Degraded => 1,
            IntegrityState::Fail => 2,
        }
    }

    fn level_max(a: LevelClass, b: LevelClass) -> LevelClass {
        match (a, b) {
            (LevelClass::High, _) | (_, LevelClass::High) => LevelClass::High,
            (LevelClass::Med, _) | (_, LevelClass::Med) => LevelClass::Med,
            _ => LevelClass::Low,
        }
    }

    fn build_reason_codes(
        integrity: IntegrityState,
        threat: LevelClass,
        policy_pressure: LevelClass,
        arousal: LevelClass,
        vectors: &[ThreatVector],
    ) -> ReasonSet {
        let mut reasons = ReasonSet::default();
        if integrity == IntegrityState::Fail {
            reasons.insert("RC.RE.INTEGRITY.FAIL");
        } else if integrity == IntegrityState::Degraded {
            reasons.insert("RC.RE.INTEGRITY.DEGRADED");
        }

        if threat == LevelClass::High {
            if vectors.contains(&ThreatVector::Exfil) {
                reasons.insert("RC.TH.EXFIL.HIGH_CONFIDENCE");
            }
            if vectors.contains(&ThreatVector::IntegrityCompromise) {
                reasons.insert("RC.TH.INTEGRITY_COMPROMISE");
            }
        }

        if policy_pressure == LevelClass::High {
            reasons.insert("RC.TH.POLICY_PROBING");
        }
        if arousal == LevelClass::High {
            reasons.insert("RC.RG.STATE.AROUSAL_UP");
        }

        reasons
    }

    #[cfg(feature = "biophys-l4-modulation")]
    fn update_modulators(&mut self, input: &InsulaInput) {
        let da_level = if input.reward_block {
            ModLevel::Low
        } else {
            level_to_mod(input.progress)
        };
        self.current_modulators = ModulatorField {
            na: level_to_mod(input.arousal),
            ht: level_to_mod(input.stability),
            da: da_level,
        };
        for (idx, synapse) in self.synapses.iter().enumerate() {
            self.syn_g_max_eff[idx] = Self::scaled_g_max_fixed(
                synapse,
                self.current_modulators,
                self.homeostasis_state.scale_q,
            );
            self.syn_stp_params_eff[idx] = synapse.stp_effective_params(self.current_modulators);
        }
    }

    #[cfg(not(feature = "biophys-l4-modulation"))]
    fn update_modulators(&mut self, _input: &InsulaInput) {
        self.current_modulators = ModulatorField::default();
        for (idx, synapse) in self.synapses.iter().enumerate() {
            self.syn_g_max_eff[idx] = Self::scaled_g_max_fixed(
                synapse,
                self.current_modulators,
                self.homeostasis_state.scale_q,
            );
            self.syn_stp_params_eff[idx] = synapse.stp_effective_params(self.current_modulators);
        }
    }

    fn scaled_g_max_fixed(synapse: &SynapseL4, mods: ModulatorField, scale_q: u16) -> u32 {
        let g_max_eff = synapse.effective_g_max_fixed(mods);
        if synapse.kind == SynKind::AMPA {
            scale_g_max_fixed(g_max_eff, scale_q, max_synapse_g_fixed())
        } else {
            g_max_eff
        }
    }

    fn refresh_syn_g_max_eff(&mut self) {
        for (idx, synapse) in self.synapses.iter().enumerate() {
            self.syn_g_max_eff[idx] = Self::scaled_g_max_fixed(
                synapse,
                self.current_modulators,
                self.homeostasis_state.scale_q,
            );
        }
    }

    fn update_homeostasis(&mut self, spike_counts: &[usize; NEURON_COUNT]) {
        let excitatory_spikes = spike_counts
            .iter()
            .take(EXCITATORY_COUNT)
            .map(|value| *value as u32)
            .sum();
        let updated = homeostasis_tick(
            self.homeostasis_config,
            &mut self.homeostasis_state,
            excitatory_spikes,
            self.in_replay_mode,
        );
        if updated {
            self.refresh_syn_g_max_eff();
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
        self.refresh_syn_g_max_eff();
    }

    pub fn plasticity_snapshot_digest(&self) -> [u8; 32] {
        let g_max_values = self
            .synapses
            .iter()
            .map(|synapse| synapse.g_max_base_q)
            .collect::<Vec<_>>();
        plasticity_snapshot_digest(self.state.step_count, &g_max_values)
    }

    pub fn plasticity_snapshot_digest_opt(&self) -> Option<[u8; 32]> {
        if !cfg!(feature = "biophys-l4-plasticity") {
            return None;
        }
        if self.learning_enabled && self.stdp_config.enabled {
            Some(self.plasticity_snapshot_digest())
        } else {
            None
        }
    }
}

impl MicrocircuitBackend<InsulaInput, IsvSnapshot> for InsulaL4Microcircuit {
    fn step(&mut self, input: &InsulaInput, _now_ms: u64) -> IsvSnapshot {
        self.state.tick_count = self.state.tick_count.saturating_add(1);
        self.update_modulators(input);
        self.set_learning_context(false, self.current_modulators, input.reward_block);
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

        self.update_homeostasis(&spike_counts);
        self.update_pool_accumulators(&spike_counts);
        self.update_latches();

        let mut effective_acc = self.state.pool_acc;
        for idx in [IDX_INTEGRITY, IDX_THREAT] {
            if self.state.latch_steps[idx] > 0 {
                effective_acc[idx] = effective_acc[idx].max(LATCH_HIGH);
            }
        }

        let mut arousal = Self::level_from_acc(effective_acc[IDX_AROUSAL]);
        let mut threat = Self::level_from_acc(effective_acc[IDX_THREAT]);
        let mut stability = Self::level_from_acc(effective_acc[IDX_STABILITY]);
        let mut policy_pressure = Self::level_from_acc(effective_acc[IDX_POLICY]);
        let progress = Self::level_from_acc(effective_acc[IDX_PROGRESS]);
        let mut integrity = Self::integrity_from_acc(effective_acc[IDX_INTEGRITY], input.integrity);

        if Self::integrity_severity(integrity) < Self::integrity_severity(input.integrity) {
            integrity = input.integrity;
        }
        if Self::severity(threat) < Self::severity(input.threat) {
            threat = input.threat;
        }

        if input.integrity == IntegrityState::Fail {
            threat = LevelClass::High;
            arousal = LevelClass::High;
            stability = LevelClass::High;
        }
        if input.receipt_invalid_present {
            threat = LevelClass::High;
        }
        if input.policy_pressure == LevelClass::High {
            policy_pressure = LevelClass::High;
            arousal = Self::level_max(arousal, LevelClass::Med);
        }
        if input.timeout_burst || input.exec_reliability == LevelClass::High {
            arousal = LevelClass::High;
        }
        if input.integrity != IntegrityState::Ok {
            stability = LevelClass::High;
        }

        let mut dominant_reason_codes = Self::build_reason_codes(
            integrity,
            threat,
            policy_pressure,
            arousal,
            &input.threat_vectors,
        );
        dominant_reason_codes.extend(input.dominant_reason_codes.clone());

        IsvSnapshot {
            arousal,
            threat,
            stability,
            policy_pressure,
            progress,
            integrity,
            dominant_reason_codes,
            threat_vectors: if input.threat_vectors.is_empty() {
                None
            } else {
                Some(input.threat_vectors.clone())
            },
            replay_hint: false,
        }
    }

    fn snapshot_digest(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"UCF:BIO:L4:INSULA:SNAP");
        update_u64(&mut hasher, self.state.step_count);
        for value in self.state.pool_acc {
            update_i32(&mut hasher, value);
        }
        for value in self.state.latch_steps {
            update_u32(&mut hasher, value as u32);
        }
        *hasher.finalize().as_bytes()
    }

    fn config_digest(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"UCF:BIO:L4:INSULA:CFG");
        update_f32(&mut hasher, DT_MS);
        update_u32(&mut hasher, SUBSTEPS as u32);
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

    fn plasticity_snapshot_digest_opt(&self) -> Option<[u8; 32]> {
        InsulaL4Microcircuit::plasticity_snapshot_digest_opt(self)
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
        let (start, end) = InsulaL4Microcircuit::pool_bounds(pool);
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

    for (source_pool, target_pool) in [
        (IDX_INTEGRITY, IDX_THREAT),
        (IDX_INTEGRITY, IDX_STABILITY),
        (IDX_THREAT, IDX_AROUSAL),
        (IDX_THREAT, IDX_STABILITY),
    ] {
        let (src_start, src_end) = InsulaL4Microcircuit::pool_bounds(source_pool);
        let (dst_start, dst_end) = InsulaL4Microcircuit::pool_bounds(target_pool);
        for pre in src_start..src_end {
            for post in dst_start..dst_end {
                let (stp_params, stp_state) = disabled_stp();
                synapses.push(SynapseL4 {
                    pre_neuron: pre as u32,
                    post_neuron: post as u32,
                    post_compartment: 1,
                    kind: SynKind::AMPA,
                    mod_channel: ModChannel::NaDa,
                    g_max_base_q: f32_to_fixed_u32(AMPA_G_MAX_WEAK),
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
        for post in EXCITATORY_COUNT..NEURON_COUNT {
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
    }

    for pre in EXCITATORY_COUNT..NEURON_COUNT {
        for post in 0..EXCITATORY_COUNT {
            let (stp_params, stp_state) = disabled_stp();
            synapses.push(SynapseL4 {
                pre_neuron: pre as u32,
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

#[cfg(feature = "biophys-l4-modulation")]
fn level_to_mod(level: LevelClass) -> ModLevel {
    match level {
        LevelClass::Low => ModLevel::Low,
        LevelClass::Med => ModLevel::Med,
        LevelClass::High => ModLevel::High,
    }
}

#[cfg(all(
    test,
    feature = "biophys",
    feature = "biophys-l4",
    feature = "biophys-l4-synapses",
    feature = "biophys-l4-insula"
))]
mod tests {
    use super::*;
    use dbm_core::DbmModule;

    fn base_input() -> InsulaInput {
        InsulaInput {
            policy_pressure: LevelClass::Low,
            receipt_failures: LevelClass::Low,
            receipt_invalid_present: false,
            exec_reliability: LevelClass::Low,
            integrity: IntegrityState::Ok,
            timeout_burst: false,
            cbv_present: false,
            pev_present: false,
            hbv_present: false,
            progress: LevelClass::Low,
            dominant_reason_codes: Vec::new(),
            arousal: LevelClass::Low,
            stability: LevelClass::Low,
            threat: LevelClass::Low,
            threat_vectors: Vec::new(),
            pag_pattern: None,
            stn_hold_active: false,
            reward_block: false,
        }
    }

    fn severity(level: LevelClass) -> u8 {
        match level {
            LevelClass::Low => 0,
            LevelClass::Med => 1,
            LevelClass::High => 2,
        }
    }

    fn integrity_severity(state: IntegrityState) -> u8 {
        match state {
            IntegrityState::Ok => 0,
            IntegrityState::Degraded => 1,
            IntegrityState::Fail => 2,
        }
    }

    #[test]
    fn determinism_sequence_outputs_match() {
        let inputs = vec![
            InsulaInput {
                integrity: IntegrityState::Degraded,
                threat: LevelClass::Med,
                stability: LevelClass::Med,
                arousal: LevelClass::Med,
                ..base_input()
            },
            InsulaInput {
                policy_pressure: LevelClass::High,
                ..base_input()
            },
            InsulaInput {
                threat: LevelClass::High,
                threat_vectors: vec![ThreatVector::Exfil],
                ..base_input()
            },
            InsulaInput {
                progress: LevelClass::High,
                reward_block: true,
                ..base_input()
            },
            InsulaInput {
                integrity: IntegrityState::Fail,
                threat: LevelClass::High,
                threat_vectors: vec![ThreatVector::IntegrityCompromise],
                ..base_input()
            },
        ];

        let mut circuit_a = InsulaL4Microcircuit::new(CircuitConfig::default());
        let mut circuit_b = InsulaL4Microcircuit::new(CircuitConfig::default());

        for tick in 0..20 {
            let input = &inputs[tick % inputs.len()];
            let out_a = circuit_a.step(input, 0);
            let out_b = circuit_b.step(input, 0);
            assert_eq!(out_a, out_b);
        }
    }

    #[test]
    fn conservative_clamps_hold() {
        let mut circuit = InsulaL4Microcircuit::new(CircuitConfig::default());
        let input_fail = InsulaInput {
            integrity: IntegrityState::Fail,
            ..base_input()
        };
        let output_fail = circuit.step(&input_fail, 0);
        assert_eq!(output_fail.integrity, IntegrityState::Fail);

        let input_threat = InsulaInput {
            threat: LevelClass::High,
            ..base_input()
        };
        let output_threat = circuit.step(&input_threat, 0);
        assert_eq!(output_threat.threat, LevelClass::High);

        let input_mixed = InsulaInput {
            integrity: IntegrityState::Degraded,
            threat: LevelClass::Med,
            ..base_input()
        };
        let output_mixed = circuit.step(&input_mixed, 0);
        assert!(
            integrity_severity(output_mixed.integrity) >= integrity_severity(input_mixed.integrity)
        );
        assert!(severity(output_mixed.threat) >= severity(input_mixed.threat));
    }

    #[test]
    fn cross_coupling_integrity_biases_threat_and_stability() {
        let mut circuit = InsulaL4Microcircuit::new(CircuitConfig::default());
        let baseline = base_input();
        let mut baseline_output = IsvSnapshot::default();
        for _ in 0..6 {
            baseline_output = circuit.step(&baseline, 0);
        }

        let input = InsulaInput {
            integrity: IntegrityState::Degraded,
            ..base_input()
        };
        let mut coupled_output = IsvSnapshot::default();
        for _ in 0..12 {
            coupled_output = circuit.step(&input, 0);
        }

        assert!(
            severity(coupled_output.threat) >= severity(baseline_output.threat),
            "threat did not increase with integrity coupling"
        );
        assert!(
            severity(coupled_output.stability) >= severity(baseline_output.stability),
            "stability did not increase with integrity coupling"
        );
    }

    #[test]
    fn bounded_state_and_reason_codes() {
        let mut circuit = InsulaL4Microcircuit::new(CircuitConfig::default());
        let input = InsulaInput {
            integrity: IntegrityState::Fail,
            threat: LevelClass::High,
            arousal: LevelClass::High,
            policy_pressure: LevelClass::High,
            threat_vectors: vec![ThreatVector::Exfil, ThreatVector::IntegrityCompromise],
            ..base_input()
        };

        for _ in 0..8 {
            let output = circuit.step(&input, 0);
            assert!(output.dominant_reason_codes.codes.len() <= ReasonSet::DEFAULT_MAX_LEN);
        }

        for acc in circuit.state.pool_acc {
            assert!((0..=100).contains(&acc));
        }
    }

    #[test]
    fn l4_is_conservative_vs_rules() {
        let mut circuit = InsulaL4Microcircuit::new(CircuitConfig::default());
        let mut rules = microcircuit_insula_stub::InsulaRules::new();
        let input = InsulaInput {
            integrity: IntegrityState::Fail,
            receipt_invalid_present: true,
            policy_pressure: LevelClass::High,
            threat: LevelClass::High,
            ..base_input()
        };

        let l4_output = circuit.step(&input, 0);
        let rules_output = rules.tick(&input);

        assert!(
            integrity_severity(l4_output.integrity) >= integrity_severity(rules_output.integrity)
        );
        assert!(severity(l4_output.threat) >= severity(rules_output.threat));
    }
}
