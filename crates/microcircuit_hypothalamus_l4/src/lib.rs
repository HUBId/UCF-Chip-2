#![forbid(unsafe_code)]

use biophys_channels::{Leak, NaK};
use biophys_compartmental_solver::{CompartmentChannels, L4Solver, L4State};
use biophys_core::{CompartmentId, ModChannel, ModLevel, ModulatorField, NeuronId, STP_SCALE};
use biophys_event_queue_l4::SpikeEventQueueL4;
use biophys_homeostasis_l4::{
    homeostasis_tick, scale_g_max_fixed, HomeoMode, HomeostasisConfig, HomeostasisState,
};
use biophys_morphology::{Compartment, CompartmentKind, NeuronMorphology};
use biophys_plasticity_l4::{plasticity_snapshot_digest, LearningMode, StdpConfig, StdpTrace};
use biophys_synapses_l4::{
    apply_stdp_updates, decay_k, f32_to_fixed_u32, max_synapse_g_fixed, StpMode, StpParamsL4,
    StpStateL4, SynKind, SynapseAccumulator, SynapseL4, SynapseState,
};
use dbm_core::{
    CooldownClass, IntegrityState, LevelClass, OverlaySet, ProfileState, ReasonSet, ThreatVector,
};
use microcircuit_core::{CircuitConfig, MicrocircuitBackend};
use microcircuit_hypothalamus_setpoint::{HypoInput, HypoOutput};
use microcircuit_pag_stub::DefensePattern;
use microcircuit_pmrf_stub::SequenceMode;

const POOL_COUNT: usize = 7;
const POOL_SIZE: usize = 2;
const PROFILE_POOL_COUNT: usize = 4;
const EXCITATORY_COUNT: usize = POOL_COUNT * POOL_SIZE;
const INHIBITORY_COUNT: usize = 2;
const NEURON_COUNT: usize = EXCITATORY_COUNT + INHIBITORY_COUNT;

const IDX_P0: usize = 0;
const IDX_P1: usize = 1;
const IDX_P2: usize = 2;
const IDX_P3: usize = 3;
const IDX_O_SIM: usize = 4;
const IDX_O_EXP: usize = 5;
const IDX_O_NOV: usize = 6;

const COMPARTMENT_COUNT: usize = 3;
const SUBSTEPS: usize = 10;
const DT_MS: f32 = 0.1;
const CLAMP_MIN: f32 = -120.0;
const CLAMP_MAX: f32 = 60.0;
const THRESHOLD_MV: f32 = -20.0;

const CURRENT_STRONG: f32 = 2.0;
const CURRENT_MED: f32 = 1.0;
const CURRENT_LOW: f32 = 0.4;
const CURRENT_BASE: f32 = 0.2;
const UNLOCK_RELIEF: f32 = 0.3;

const AMPA_G_MAX_BASE: f32 = 3.0;
const AMPA_G_MAX_DOMINANT: f32 = 4.0;
const AMPA_E_REV: f32 = 0.0;
const AMPA_TAU_RISE_MS: f32 = 0.0;
const AMPA_TAU_DECAY_MS: f32 = 8.0;

const GABA_G_MAX: f32 = 6.0;
const GABA_E_REV: f32 = -70.0;
const GABA_TAU_RISE_MS: f32 = 0.0;
const GABA_TAU_DECAY_MS: f32 = 10.0;

const MAX_EVENTS_PER_STEP: usize = 2048;
const ACCUMULATOR_MAX: i32 = 100;
const ACCUMULATOR_DECAY: i32 = 5;
const ACCUMULATOR_GAIN: i32 = 20;
const HYSTERESIS_TICKS: u8 = 3;
const STABLE_COUNTER_TARGET: u8 = 2;
const RECOVERY_GUARD_TICKS: u8 = 3;

#[derive(Debug, Clone)]
struct L4Neuron {
    solver: L4Solver,
    state: L4State,
    last_soma_v: f32,
}

#[derive(Debug, Clone)]
struct HypoL4State {
    tick_count: u64,
    step_count: u64,
    pool_acc: [i32; POOL_COUNT],
    last_pool_spikes: [usize; POOL_COUNT],
    last_spike_count_total: usize,
    winner_profile: usize,
    winner_streak: [u8; PROFILE_POOL_COUNT],
    forensic_latched: bool,
    stable_counter: u8,
    recovery_guard: u8,
}

impl Default for HypoL4State {
    fn default() -> Self {
        Self {
            tick_count: 0,
            step_count: 0,
            pool_acc: [0; POOL_COUNT],
            last_pool_spikes: [0; POOL_COUNT],
            last_spike_count_total: 0,
            winner_profile: IDX_P0,
            winner_streak: [0; PROFILE_POOL_COUNT],
            forensic_latched: false,
            stable_counter: 0,
            recovery_guard: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HypothalamusL4Microcircuit {
    _config: CircuitConfig,
    neurons: Vec<L4Neuron>,
    synapses: Vec<SynapseL4>,
    syn_states: Vec<SynapseState>,
    syn_g_max_eff: Vec<u32>,
    syn_decay: Vec<u16>,
    syn_stp_params_eff: Vec<StpParamsL4>,
    pre_index: Vec<Vec<usize>>,
    queue: SpikeEventQueueL4,
    state: HypoL4State,
    current_modulators: ModulatorField,
    stdp_config: StdpConfig,
    stdp_traces: Vec<StdpTrace>,
    stdp_spike_flags: Vec<bool>,
    learning_enabled: bool,
    in_replay_mode: bool,
    homeostasis_config: HomeostasisConfig,
    homeostasis_state: HomeostasisState,
}

impl HypothalamusL4Microcircuit {
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
            state: HypoL4State::default(),
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

    fn split_required(mode: SequenceMode) -> bool {
        matches!(mode, SequenceMode::SplitRequired)
    }

    fn is_dp2(pattern: Option<DefensePattern>) -> bool {
        matches!(pattern, Some(DefensePattern::DP2_QUARANTINE))
    }

    fn is_dp3(pattern: Option<DefensePattern>) -> bool {
        matches!(pattern, Some(DefensePattern::DP3_FORENSIC))
    }

    fn exfil_present(input: &HypoInput) -> bool {
        input
            .isv
            .threat_vectors
            .as_ref()
            .map(|vectors| vectors.iter().any(|v| matches!(v, ThreatVector::Exfil)))
            .unwrap_or(false)
    }

    fn tool_side_effects_present(input: &HypoInput) -> bool {
        input
            .isv
            .threat_vectors
            .as_ref()
            .map(|vectors| {
                vectors
                    .iter()
                    .any(|v| matches!(v, ThreatVector::ToolSideEffects))
            })
            .unwrap_or(false)
    }

    fn receipt_failures_present(input: &HypoInput) -> bool {
        input.isv.dominant_reason_codes.codes.iter().any(|code| {
            let lower = code.to_ascii_lowercase();
            lower.contains("receipt") || lower.contains("dispatch_blocked")
        })
    }

    fn stable_condition_met(input: &HypoInput) -> bool {
        input.isv.integrity == IntegrityState::Ok
            && input.isv.threat == LevelClass::Low
            && !Self::receipt_failures_present(input)
    }

    fn update_stable_counter(&mut self, input: &HypoInput) {
        if Self::stable_condition_met(input) {
            self.state.stable_counter = self.state.stable_counter.saturating_add(1);
        } else {
            self.state.stable_counter = 0;
        }
    }

    fn update_latch(&mut self, input: &HypoInput) -> bool {
        if input.isv.integrity == IntegrityState::Fail || Self::is_dp3(input.pag_pattern) {
            self.state.forensic_latched = true;
        }
        let release_ready = self.state.forensic_latched
            && input.unlock_ready
            && self.state.stable_counter >= STABLE_COUNTER_TARGET;
        if release_ready {
            self.state.forensic_latched = false;
            self.state.recovery_guard = RECOVERY_GUARD_TICKS;
        }
        if !input.unlock_ready {
            self.state.recovery_guard = 0;
        }
        release_ready
    }

    fn strictness_rank(pool: usize) -> u8 {
        match pool {
            IDX_P3 => 3,
            IDX_P2 => 2,
            IDX_P1 => 1,
            _ => 0,
        }
    }

    fn winner_with_tiebreak(acc: &[i32; PROFILE_POOL_COUNT]) -> usize {
        let mut winner = IDX_P0;
        let mut best = acc[IDX_P0];
        for (idx, value) in acc.iter().enumerate().take(PROFILE_POOL_COUNT).skip(1) {
            let rank = Self::strictness_rank(idx);
            let best_rank = Self::strictness_rank(winner);
            if *value > best || (*value == best && rank > best_rank) {
                best = *value;
                winner = idx;
            }
        }
        winner
    }

    fn apply_hysteresis(&mut self, raw_winner: usize) -> usize {
        for idx in 0..PROFILE_POOL_COUNT {
            if idx == raw_winner {
                self.state.winner_streak[idx] = self.state.winner_streak[idx].saturating_add(1);
            } else {
                self.state.winner_streak[idx] = 0;
            }
        }

        let current = self.state.winner_profile;
        if Self::strictness_rank(raw_winner) > Self::strictness_rank(current) {
            self.state.winner_profile = raw_winner;
            self.state.winner_streak = [0; PROFILE_POOL_COUNT];
            self.state.winner_streak[raw_winner] = 1;
            return raw_winner;
        }

        if raw_winner == current {
            return current;
        }

        if self.state.winner_streak[raw_winner] >= HYSTERESIS_TICKS {
            self.state.winner_profile = raw_winner;
        }

        self.state.winner_profile
    }

    fn baseline_strictness(input: &HypoInput) -> bool {
        matches!(
            input.baseline.approval_strictness,
            LevelClass::Med | LevelClass::High
        ) || matches!(
            input.baseline.export_strictness,
            LevelClass::Med | LevelClass::High
        ) || matches!(
            input.baseline.chain_conservatism,
            LevelClass::Med | LevelClass::High
        )
    }

    fn build_pool_drives(&self, input: &HypoInput) -> [f32; POOL_COUNT] {
        let mut drives = [0.0_f32; POOL_COUNT];

        drives[IDX_P0] += CURRENT_BASE;
        let safe = input.isv.integrity == IntegrityState::Ok
            && input.isv.threat == LevelClass::Low
            && input.isv.policy_pressure == LevelClass::Low
            && input.isv.stability == LevelClass::Low
            && !self.state.forensic_latched;
        if safe {
            drives[IDX_P0] += CURRENT_LOW;
        }

        if input.isv.policy_pressure == LevelClass::High
            || Self::receipt_failures_present(input)
            || input.isv.replay_hint
            || Self::baseline_strictness(input)
        {
            drives[IDX_P1] += CURRENT_MED;
        }

        if input.isv.threat == LevelClass::High
            || Self::is_dp2(input.pag_pattern)
            || Self::exfil_present(input)
            || Self::tool_side_effects_present(input)
        {
            drives[IDX_P2] += CURRENT_STRONG;
        }

        if input.isv.integrity == IntegrityState::Fail
            || Self::is_dp3(input.pag_pattern)
            || self.state.forensic_latched
        {
            drives[IDX_P3] += CURRENT_STRONG;
        }

        if input.stn_hold_active
            || Self::split_required(input.pmrf_sequence_mode)
            || input.isv.policy_pressure == LevelClass::High
        {
            drives[IDX_O_SIM] += CURRENT_STRONG;
        }
        if input.baseline.chain_conservatism == LevelClass::High {
            drives[IDX_O_SIM] += CURRENT_MED;
        }

        if Self::exfil_present(input) || input.baseline.export_strictness == LevelClass::High {
            drives[IDX_O_EXP] += CURRENT_STRONG;
        }

        if input.isv.arousal == LevelClass::High
            || input.baseline.novelty_dampening == LevelClass::High
        {
            drives[IDX_O_NOV] += CURRENT_STRONG;
        }

        if input.unlock_present && input.unlock_ready && input.isv.integrity != IntegrityState::Fail
        {
            drives[IDX_P3] = (drives[IDX_P3] - UNLOCK_RELIEF).max(0.0);
            drives[IDX_P1] += UNLOCK_RELIEF;
        }

        drives
    }

    fn build_inputs(&self, input: &HypoInput) -> [[f32; COMPARTMENT_COUNT]; NEURON_COUNT] {
        let mut currents = [[0.0_f32; COMPARTMENT_COUNT]; NEURON_COUNT];
        let drives = self.build_pool_drives(input);
        for (pool, drive) in drives.iter().enumerate().take(POOL_COUNT) {
            let (start, end) = Self::pool_bounds(pool);
            for neuron in currents.iter_mut().take(end).skip(start) {
                neuron[0] += drive;
            }
        }
        currents
    }

    fn mod_level_from_class(level: LevelClass) -> ModLevel {
        match level {
            LevelClass::Low => ModLevel::Low,
            LevelClass::Med => ModLevel::Med,
            LevelClass::High => ModLevel::High,
        }
    }

    fn modulators_from_input(input: &HypoInput) -> ModulatorField {
        let da_level = if input.baseline.reward_block_bias == LevelClass::High {
            LevelClass::Low
        } else {
            input.isv.progress
        };
        ModulatorField {
            na: Self::mod_level_from_class(input.isv.arousal),
            ht: Self::mod_level_from_class(input.isv.stability),
            da: Self::mod_level_from_class(da_level),
        }
    }

    fn update_modulators(&mut self, input: &HypoInput) {
        let mods = Self::modulators_from_input(input);
        self.current_modulators = mods;
        for (idx, synapse) in self.synapses.iter().enumerate() {
            self.syn_g_max_eff[idx] =
                Self::scaled_g_max_fixed(synapse, mods, self.homeostasis_state.scale_q);
            self.syn_stp_params_eff[idx] = synapse.stp_effective_params(mods);
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
            let syn_input = &accumulators[idx];
            let mut input = injected_currents[idx];
            for comp in 0..COMPARTMENT_COUNT {
                input[comp] += syn_input[comp].total_current(neuron.state.voltages[comp]);
            }
            neuron.solver.step(&mut neuron.state, &input);
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
                        STP_SCALE
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

    fn update_pool_accumulators(&mut self, spike_counts: &[usize; NEURON_COUNT]) {
        let mut pool_spikes = [0usize; POOL_COUNT];
        for (pool, pool_spike) in pool_spikes.iter_mut().enumerate().take(POOL_COUNT) {
            let (start, end) = Self::pool_bounds(pool);
            *pool_spike = spike_counts[start..end].iter().sum();
        }

        for (idx, spikes) in pool_spikes.iter().enumerate() {
            let delta = (*spikes as i32) * ACCUMULATOR_GAIN - ACCUMULATOR_DECAY;
            let updated = (self.state.pool_acc[idx] + delta).clamp(0, ACCUMULATOR_MAX);
            self.state.pool_acc[idx] = updated;
        }

        self.state.last_pool_spikes = pool_spikes;
    }

    fn profile_from_pool(pool_idx: usize) -> ProfileState {
        match pool_idx {
            IDX_P3 => ProfileState::M3,
            IDX_P2 => ProfileState::M2,
            IDX_P1 => ProfileState::M1,
            _ => ProfileState::M0,
        }
    }

    fn rules_floor(input: &HypoInput) -> ProfileState {
        if input.isv.integrity == IntegrityState::Fail || Self::is_dp3(input.pag_pattern) {
            ProfileState::M3
        } else if input.isv.threat == LevelClass::High
            || Self::exfil_present(input)
            || Self::is_dp2(input.pag_pattern)
        {
            ProfileState::M2
        } else if input.isv.policy_pressure == LevelClass::High
            || Self::receipt_failures_present(input)
        {
            ProfileState::M1
        } else {
            ProfileState::M0
        }
    }

    fn overlay_triggers(input: &HypoInput) -> (bool, bool, bool) {
        let simulate = input.stn_hold_active
            || Self::split_required(input.pmrf_sequence_mode)
            || input.isv.policy_pressure == LevelClass::High
            || input.baseline.chain_conservatism == LevelClass::High;
        let export =
            Self::exfil_present(input) || input.baseline.export_strictness == LevelClass::High;
        let novelty = input.isv.arousal == LevelClass::High
            || input.baseline.novelty_dampening == LevelClass::High;
        (simulate, export, novelty)
    }

    fn overlay_set(&self, profile: ProfileState, input: &HypoInput) -> OverlaySet {
        let (sim_trigger, exp_trigger, nov_trigger) = Self::overlay_triggers(input);
        let mut overlays = OverlaySet {
            simulate_first: self.state.pool_acc[IDX_O_SIM] >= 60 || sim_trigger,
            export_lock: self.state.pool_acc[IDX_O_EXP] >= 60 || exp_trigger,
            novelty_lock: self.state.pool_acc[IDX_O_NOV] >= 60 || nov_trigger,
        };

        if profile == ProfileState::M3 || profile == ProfileState::M2 || self.state.forensic_latched
        {
            overlays = OverlaySet::all_enabled();
        }

        if self.state.recovery_guard > 0 {
            overlays = OverlaySet::all_enabled();
        }

        overlays
    }

    fn cooldown_level(input: &HypoInput, profile: ProfileState) -> LevelClass {
        if input.isv.stability == LevelClass::High
            || input.baseline.cooldown_bias == CooldownClass::Longer
            || matches!(profile, ProfileState::M2 | ProfileState::M3)
        {
            LevelClass::High
        } else {
            LevelClass::Low
        }
    }

    fn reason_codes(
        &self,
        input: &HypoInput,
        profile: ProfileState,
        overlays: &OverlaySet,
        release_ready: bool,
    ) -> ReasonSet {
        let mut codes = Vec::new();
        codes.extend(input.isv.dominant_reason_codes.codes.iter().cloned());

        let profile_code = match profile {
            ProfileState::M3 => "RC.RG.PROFILE.M3",
            ProfileState::M2 => "RC.RG.PROFILE.M2",
            ProfileState::M1 => "RC.RG.PROFILE.M1",
            ProfileState::M0 => "RC.RG.PROFILE.M0",
        };
        codes.push(profile_code.to_string());

        if profile == ProfileState::M3 || self.state.forensic_latched {
            codes.push("RC.RX.ACTION.FORENSIC".to_string());
        }

        if release_ready {
            codes.push("RC.GV.RECOVERY.UNLOCK_GRANTED".to_string());
        }

        if overlays.simulate_first {
            codes.push("RC.RG.OVERLAY.SIMULATE_FIRST".to_string());
        }
        if overlays.export_lock {
            codes.push("RC.CD.DLP.EXPORT_BLOCKED".to_string());
        }
        if overlays.novelty_lock {
            codes.push("RC.RG.OVERLAY.NOVELTY_LOCK".to_string());
        }

        codes.sort();
        codes.dedup();

        let mut reason_set = ReasonSet::new(ReasonSet::DEFAULT_MAX_LEN);
        for code in codes {
            reason_set.insert(code);
        }
        reason_set
    }
}

impl MicrocircuitBackend<HypoInput, HypoOutput> for HypothalamusL4Microcircuit {
    fn step(&mut self, input: &HypoInput, _now_ms: u64) -> HypoOutput {
        self.state.tick_count = self.state.tick_count.saturating_add(1);
        self.update_modulators(input);
        let reward_block = input.baseline.reward_block_bias == LevelClass::High;
        self.set_learning_context(input.isv.replay_hint, self.current_modulators, reward_block);
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

        self.update_stable_counter(input);
        let release_ready = self.update_latch(input);

        let currents = self.build_inputs(input);
        let mut spike_counts = [0usize; NEURON_COUNT];
        for _ in 0..SUBSTEPS {
            let spikes = self.substep(&currents);
            for spike in spikes {
                spike_counts[spike] = spike_counts[spike].saturating_add(1);
            }
        }

        self.update_homeostasis(&spike_counts);
        self.update_pool_accumulators(&spike_counts);
        self.state.last_spike_count_total = spike_counts.iter().sum();

        let mut profile_pool_acc = [0i32; PROFILE_POOL_COUNT];
        profile_pool_acc.copy_from_slice(&self.state.pool_acc[..PROFILE_POOL_COUNT]);

        let raw_winner = Self::winner_with_tiebreak(&profile_pool_acc);
        let winner = self.apply_hysteresis(raw_winner);

        let mut profile = Self::profile_from_pool(winner);
        let floor = Self::rules_floor(input);
        if Self::strictness_rank(winner) < Self::strictness_rank(Self::pool_from_profile(floor)) {
            profile = floor;
        }

        if self.state.forensic_latched && !release_ready {
            profile = ProfileState::M3;
        }

        if release_ready || (self.state.recovery_guard > 0 && matches!(profile, ProfileState::M0)) {
            profile = ProfileState::M1;
        }

        let updated_pool = Self::pool_from_profile(profile);
        if updated_pool != self.state.winner_profile {
            self.state.winner_profile = updated_pool;
            self.state.winner_streak = [0; PROFILE_POOL_COUNT];
            if updated_pool < PROFILE_POOL_COUNT {
                self.state.winner_streak[updated_pool] = 1;
            }
        }

        if self.state.recovery_guard > 0 {
            self.state.recovery_guard -= 1;
        }

        let overlays = self.overlay_set(profile, input);

        let deescalation_lock = profile != ProfileState::M0
            || input.isv.stability == LevelClass::High
            || self.state.forensic_latched;

        let cooldown_class = Self::cooldown_level(input, profile);
        let reason_codes = self.reason_codes(input, profile, &overlays, release_ready);

        HypoOutput {
            profile_state: profile,
            overlays,
            deescalation_lock,
            cooldown_class,
            reason_codes,
        }
    }

    fn snapshot_digest(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"UCF:BIO:L4:HYPO:SNAP");
        hasher.update(&self.state.tick_count.to_le_bytes());
        hasher.update(&self.state.step_count.to_le_bytes());
        for acc in self.state.pool_acc {
            hasher.update(&acc.to_le_bytes());
        }
        for spikes in self.state.last_pool_spikes {
            hasher.update(&(spikes as u32).to_le_bytes());
        }
        hasher.update(&(self.state.last_spike_count_total as u32).to_le_bytes());
        hasher.update(&(self.state.winner_profile as u32).to_le_bytes());
        for streak in self.state.winner_streak {
            hasher.update(&[streak]);
        }
        hasher.update(&[self.state.forensic_latched as u8]);
        hasher.update(&[self.state.stable_counter]);
        hasher.update(&[self.state.recovery_guard]);

        for neuron in &self.neurons {
            update_u64(&mut hasher, neuron.solver.step_count());
            update_u32(&mut hasher, neuron.state.voltages.len() as u32);
            for (v, gates) in neuron.state.voltages.iter().zip(neuron.state.gates.iter()) {
                update_f32(&mut hasher, *v);
                update_f32(&mut hasher, gates.m);
                update_f32(&mut hasher, gates.h);
                update_f32(&mut hasher, gates.n);
            }
        }
        for (synapse, state) in self.synapses.iter().zip(self.syn_states.iter()) {
            update_u32(&mut hasher, synapse.pre_neuron);
            update_u32(&mut hasher, synapse.post_neuron);
            update_u32(&mut hasher, synapse.post_compartment);
            update_u32(&mut hasher, synapse.g_max_base_q);
            update_f32(&mut hasher, synapse.e_rev);
            update_u32(&mut hasher, synapse.delay_steps as u32);
            update_u32(&mut hasher, state.g_fixed);
            hasher.update(&synapse.stp_state.x_q.to_le_bytes());
            hasher.update(&synapse.stp_state.u_q.to_le_bytes());
        }
        *hasher.finalize().as_bytes()
    }

    fn config_digest(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"UCF:BIO:L4:HYPO:CFG");
        update_f32(&mut hasher, DT_MS);
        update_u32(&mut hasher, SUBSTEPS as u32);
        update_u32(&mut hasher, NEURON_COUNT as u32);
        for neuron in &self.neurons {
            hasher.update(&neuron.solver.config_digest());
        }
        for synapse in &self.synapses {
            update_u32(&mut hasher, synapse.pre_neuron);
            update_u32(&mut hasher, synapse.post_neuron);
            update_u32(&mut hasher, synapse.post_compartment);
            update_u32(&mut hasher, synapse.g_max_base_q);
            update_f32(&mut hasher, synapse.e_rev);
            update_f32(&mut hasher, synapse.tau_decay_ms);
            update_u32(&mut hasher, synapse.delay_steps as u32);
            update_u32(&mut hasher, syn_kind_code(synapse.kind));
            update_u32(&mut hasher, stp_mode_code(synapse.stp_params.mode));
            update_u32(&mut hasher, synapse.stp_params.u_base_q as u32);
            update_u32(&mut hasher, synapse.stp_params.tau_rec_steps as u32);
            update_u32(&mut hasher, synapse.stp_params.tau_fac_steps as u32);
        }
        *hasher.finalize().as_bytes()
    }

    fn plasticity_snapshot_digest_opt(&self) -> Option<[u8; 32]> {
        HypothalamusL4Microcircuit::plasticity_snapshot_digest_opt(self)
    }
}

impl HypothalamusL4Microcircuit {
    fn pool_from_profile(profile: ProfileState) -> usize {
        match profile {
            ProfileState::M3 => IDX_P3,
            ProfileState::M2 => IDX_P2,
            ProfileState::M1 => IDX_P1,
            ProfileState::M0 => IDX_P0,
        }
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
        let (start, end) = HypothalamusL4Microcircuit::pool_bounds(pool);
        for pre in start..end {
            for post in start..end {
                if pre == post {
                    continue;
                }
                let (stp_params, stp_state) = within_pool_stp();
                synapses.push(SynapseL4 {
                    pre_neuron: pre as u32,
                    post_neuron: post as u32,
                    post_compartment: 1,
                    kind: SynKind::AMPA,
                    mod_channel: ModChannel::Na,
                    g_max_base_q: f32_to_fixed_u32(AMPA_G_MAX_BASE),
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
        let pool = pre / POOL_SIZE;
        let g_max = match pool {
            IDX_P3 => AMPA_G_MAX_DOMINANT,
            IDX_P2 => AMPA_G_MAX_DOMINANT * 0.9,
            IDX_P1 => AMPA_G_MAX_BASE * 1.1,
            _ => AMPA_G_MAX_BASE,
        };
        for inh in 0..INHIBITORY_COUNT {
            let post = EXCITATORY_COUNT + inh;
            let (stp_params, stp_state) = disabled_stp();
            synapses.push(SynapseL4 {
                pre_neuron: pre as u32,
                post_neuron: post as u32,
                post_compartment: 0,
                kind: SynKind::AMPA,
                mod_channel: ModChannel::Na,
                g_max_base_q: f32_to_fixed_u32(g_max),
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

    for inh in 0..INHIBITORY_COUNT {
        let pre = EXCITATORY_COUNT + inh;
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

fn within_pool_stp() -> (StpParamsL4, StpStateL4) {
    if cfg!(feature = "biophys-l4-stp") {
        let params = StpParamsL4 {
            mode: StpMode::STP_TM,
            u_base_q: 200,
            tau_rec_steps: 6,
            tau_fac_steps: 4,
        };
        let state = StpStateL4::new(params);
        (params, state)
    } else {
        disabled_stp()
    }
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

fn scale_fixed_by_level(value: u32, level: ModLevel) -> u32 {
    let mult = match level {
        ModLevel::Low => 90,
        ModLevel::Med => 100,
        ModLevel::High => 110,
    };
    ((value as u64 * mult as u64) / 100) as u32
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

fn syn_kind_code(kind: SynKind) -> u32 {
    match kind {
        SynKind::AMPA => 0,
        SynKind::NMDA => 1,
        SynKind::GABA => 2,
    }
}

fn stp_mode_code(mode: StpMode) -> u32 {
    match mode {
        StpMode::STP_NONE => 0,
        StpMode::STP_TM => 1,
    }
}

#[cfg(all(
    test,
    feature = "biophys",
    feature = "biophys-l4",
    feature = "biophys-l4-synapses",
    feature = "biophys-l4-hypothalamus"
))]
mod tests {
    use super::*;

    fn profile_rank(profile: ProfileState) -> u8 {
        match profile {
            ProfileState::M0 => 0,
            ProfileState::M1 => 1,
            ProfileState::M2 => 2,
            ProfileState::M3 => 3,
        }
    }

    fn base_input() -> HypoInput {
        HypoInput::default()
    }

    #[test]
    fn determinism_sequence_and_digests() {
        let inputs = (0..20)
            .map(|idx| HypoInput {
                isv: dbm_core::IsvSnapshot {
                    integrity: if idx % 9 == 0 {
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
                    stability: if idx % 6 == 0 {
                        LevelClass::High
                    } else {
                        LevelClass::Low
                    },
                    arousal: if idx % 4 == 0 {
                        LevelClass::High
                    } else {
                        LevelClass::Low
                    },
                    progress: if idx % 3 == 0 {
                        LevelClass::High
                    } else {
                        LevelClass::Low
                    },
                    threat_vectors: if idx % 8 == 0 {
                        Some(vec![ThreatVector::Exfil])
                    } else {
                        None
                    },
                    ..dbm_core::IsvSnapshot::default()
                },
                pag_pattern: if idx % 10 == 0 {
                    Some(DefensePattern::DP3_FORENSIC)
                } else if idx % 11 == 0 {
                    Some(DefensePattern::DP2_QUARANTINE)
                } else {
                    None
                },
                stn_hold_active: idx % 3 == 0,
                pmrf_sequence_mode: if idx % 7 == 0 {
                    SequenceMode::SplitRequired
                } else {
                    SequenceMode::Normal
                },
                baseline: dbm_core::BaselineVector {
                    chain_conservatism: if idx % 4 == 0 {
                        LevelClass::High
                    } else {
                        LevelClass::Low
                    },
                    export_strictness: if idx % 6 == 0 {
                        LevelClass::High
                    } else {
                        LevelClass::Low
                    },
                    novelty_dampening: if idx % 5 == 0 {
                        LevelClass::High
                    } else {
                        LevelClass::Low
                    },
                    approval_strictness: if idx % 9 == 0 {
                        LevelClass::High
                    } else {
                        LevelClass::Low
                    },
                    ..dbm_core::BaselineVector::default()
                },
                unlock_present: idx % 12 == 0,
                unlock_ready: idx % 13 == 0,
                now_ms: idx as u64,
            })
            .collect::<Vec<_>>();

        let run_sequence = |inputs: &[HypoInput]| -> Vec<(HypoOutput, [u8; 32])> {
            let mut circuit = HypothalamusL4Microcircuit::new(CircuitConfig::default());
            inputs
                .iter()
                .map(|input| {
                    let output = circuit.step(input, input.now_ms);
                    let digest = circuit.snapshot_digest();
                    assert!(output.reason_codes.codes.windows(2).all(|w| w[0] <= w[1]));
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
    fn critical_invariants_hold() {
        let mut circuit = HypothalamusL4Microcircuit::new(CircuitConfig::default());

        let integrity = HypoInput {
            isv: dbm_core::IsvSnapshot {
                integrity: IntegrityState::Fail,
                ..dbm_core::IsvSnapshot::default()
            },
            ..base_input()
        };
        let exfil = HypoInput {
            isv: dbm_core::IsvSnapshot {
                threat_vectors: Some(vec![ThreatVector::Exfil]),
                ..dbm_core::IsvSnapshot::default()
            },
            ..base_input()
        };
        let threat = HypoInput {
            isv: dbm_core::IsvSnapshot {
                threat: LevelClass::High,
                ..dbm_core::IsvSnapshot::default()
            },
            ..base_input()
        };
        let policy = HypoInput {
            isv: dbm_core::IsvSnapshot {
                policy_pressure: LevelClass::High,
                ..dbm_core::IsvSnapshot::default()
            },
            ..base_input()
        };

        let integrity_out = circuit.step(&integrity, 0);
        assert_eq!(integrity_out.profile_state, ProfileState::M3);

        let exfil_out = circuit.step(&exfil, 0);
        assert!(profile_rank(exfil_out.profile_state) >= profile_rank(ProfileState::M2));

        let threat_out = circuit.step(&threat, 0);
        assert!(profile_rank(threat_out.profile_state) >= profile_rank(ProfileState::M2));

        let policy_out = circuit.step(&policy, 0);
        assert!(profile_rank(policy_out.profile_state) >= profile_rank(ProfileState::M1));
    }

    #[test]
    fn unlock_gating_respects_stability_threshold() {
        let mut circuit = HypothalamusL4Microcircuit::new(CircuitConfig::default());

        let fail_input = HypoInput {
            isv: dbm_core::IsvSnapshot {
                integrity: IntegrityState::Fail,
                ..dbm_core::IsvSnapshot::default()
            },
            ..base_input()
        };

        let output = circuit.step(&fail_input, 0);
        assert_eq!(output.profile_state, ProfileState::M3);

        let stable_unlock = HypoInput {
            unlock_present: true,
            unlock_ready: true,
            isv: dbm_core::IsvSnapshot {
                integrity: IntegrityState::Ok,
                threat: LevelClass::Low,
                stability: LevelClass::Low,
                ..dbm_core::IsvSnapshot::default()
            },
            ..base_input()
        };

        let output = circuit.step(&stable_unlock, 0);
        assert_eq!(output.profile_state, ProfileState::M3);

        let output = circuit.step(&stable_unlock, 0);
        assert_eq!(output.profile_state, ProfileState::M1);
        assert_ne!(output.profile_state, ProfileState::M0);
        assert!(output.overlays.simulate_first);
        assert!(output.overlays.export_lock);
        assert!(output.overlays.novelty_lock);
    }

    #[test]
    fn tighten_only_overlay_respects_hints() {
        let mut circuit = HypothalamusL4Microcircuit::new(CircuitConfig::default());
        let input = HypoInput {
            isv: dbm_core::IsvSnapshot {
                policy_pressure: LevelClass::High,
                ..dbm_core::IsvSnapshot::default()
            },
            ..base_input()
        };

        let output = circuit.step(&input, 0);
        assert!(output.overlays.simulate_first);
    }
}
