#![forbid(unsafe_code)]

use biophys_channels::{Leak, NaK};
use biophys_compartmental_solver::{CompartmentChannels, L4Solver, L4State};
use biophys_core::{CompartmentId, ModChannel, ModLevel, ModulatorField, NeuronId, STP_SCALE};
use biophys_event_queue_l4::{QueueLimits, RuntimeHealth, SpikeEventQueueL4};
use biophys_homeostasis_l4::{
    homeostasis_tick, scale_g_max_fixed, HomeoMode, HomeostasisConfig, HomeostasisState,
};
use biophys_morphology::{Compartment, CompartmentKind, NeuronMorphology};
use biophys_plasticity_l4::{plasticity_snapshot_digest, LearningMode, StdpConfig, StdpTrace};
use biophys_synapses_l4::{
    apply_stdp_updates, decay_k, f32_to_fixed_u32, max_synapse_g_fixed, NmdaVDepMode, StpMode,
    StpParamsL4, StpStateL4, SynKind, SynapseAccumulator, SynapseL4, SynapseState,
};
use biophys_targeting_l4::{select_post_compartment, EdgeKey, TargetRule, TargetingPolicy};
use dbm_core::{
    CooldownClass, IntegrityState, LevelClass, OverlaySet, ProfileState, ReasonSet, ThreatVector,
};
use microcircuit_core::{CircuitConfig, MicrocircuitBackend};
use microcircuit_hypothalamus_setpoint::{HypoInput, HypoOutput};
use microcircuit_pag_stub::DefensePattern;
use microcircuit_pmrf_stub::SequenceMode;
#[cfg(feature = "biophys-l4-hypothalamus-assets")]
use {
    asset_rehydration::{
        decode_channel_params, decode_connectivity, decode_morphology, decode_synapse_params,
        AssetRehydrator,
    },
    biophys_asset_builder::{CircuitBuilderFromAssets, Error as AssetBuildError},
    biophys_assets::{
        channel_params_from_payload, channel_params_payload_digest, connectivity_from_payload,
        connectivity_payload_digest, morphology_from_payload, morphology_payload_digest,
        synapse_params_from_payload, synapse_params_payload_digest, ChannelParamsSet,
        ConnectivityGraph, MorphologySet, SynapseParamsSet,
    },
    ucf::v1::{AssetBundle, AssetKind},
};

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
#[cfg(feature = "biophys-l4-hypothalamus-assets")]
const ASSET_POOL_CONVENTION_V1: u32 = 1;
#[cfg(feature = "biophys-l4-hypothalamus-assets")]
const ASSET_POOL_CONVENTION_V2: u32 = 2;
#[cfg(feature = "biophys-l4-hypothalamus-assets")]
const MAX_ASSET_EDGES: usize = 10_000;
#[cfg(feature = "biophys-l4-hypothalamus-assets")]
const LEAK_G_SCALE: f32 = 0.1;

#[cfg(feature = "biophys-l4-hypothalamus-assets")]
const LABEL_KEY_POOL: &str = "pool";
#[cfg(feature = "biophys-l4-hypothalamus-assets")]
const LABEL_KEY_ROLE: &str = "role";
#[cfg(feature = "biophys-l4-hypothalamus-assets")]
const ROLE_EXCITATORY: &str = "E";
#[cfg(feature = "biophys-l4-hypothalamus-assets")]
const ROLE_INHIBITORY: &str = "I";
#[cfg(feature = "biophys-l4-hypothalamus-assets")]
const POOL_LABELS: [&str; 8] = ["P0", "P1", "P2", "P3", "O_SIM", "O_EXP", "O_NOV", "INH"];

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
    last_queue_health: RuntimeHealth,
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
            last_queue_health: RuntimeHealth::default(),
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
    #[cfg(feature = "biophys-l4-hypothalamus-assets")]
    asset_manifest_digest: Option<[u8; 32]>,
    #[cfg(feature = "biophys-l4-hypothalamus-assets")]
    asset_pool_mapping_version: u32,
    #[cfg(feature = "biophys-l4-hypothalamus-assets")]
    asset_pool_mapping_digest: [u8; 32],
}

impl HypothalamusL4Microcircuit {
    pub fn new(config: CircuitConfig) -> Self {
        let neurons = (0..NEURON_COUNT)
            .map(|idx| build_neuron(idx as u32))
            .collect::<Vec<_>>();
        let synapses = build_synapses();
        Self::build_from_parts(config, neurons, synapses, None, 0, [0u8; 32])
    }

    #[cfg(feature = "biophys-l4-hypothalamus-assets")]
    pub fn new_from_asset_bundle(
        bundle: &AssetBundle,
        rehydrator: &AssetRehydrator,
    ) -> Result<Self, AssetBuildError> {
        rehydrator.verify_bundle(bundle)?;
        let manifest = bundle
            .manifest
            .as_ref()
            .ok_or(AssetBuildError::MissingManifest)?;
        let manifest_digest = digest_from_vec(&manifest.manifest_digest, "manifest_digest")?;
        let morph_digest = manifest_digest_for_kind(manifest, AssetKind::MorphologySet)?;
        let chan_digest = manifest_digest_for_kind(manifest, AssetKind::ChannelParamsSet)?;
        let syn_digest = manifest_digest_for_kind(manifest, AssetKind::SynapseParamsSet)?;
        let conn_digest = manifest_digest_for_kind(manifest, AssetKind::ConnectivityGraph)?;

        let morph_bytes = rehydrator.reassemble(bundle, AssetKind::MorphologySet, morph_digest)?;
        let chan_bytes = rehydrator.reassemble(bundle, AssetKind::ChannelParamsSet, chan_digest)?;
        let syn_bytes = rehydrator.reassemble(bundle, AssetKind::SynapseParamsSet, syn_digest)?;
        let conn_bytes =
            rehydrator.reassemble(bundle, AssetKind::ConnectivityGraph, conn_digest)?;

        let morph_payload = decode_morphology(&morph_bytes)?;
        let chan_payload = decode_channel_params(&chan_bytes)?;
        let syn_payload = decode_synapse_params(&syn_bytes)?;
        let conn_payload = decode_connectivity(&conn_bytes)?;

        verify_asset_digest(
            "morphology",
            morphology_payload_digest(&morph_payload),
            morph_digest,
        )?;
        verify_asset_digest(
            "channel_params",
            channel_params_payload_digest(&chan_payload),
            chan_digest,
        )?;
        verify_asset_digest(
            "synapse_params",
            synapse_params_payload_digest(&syn_payload),
            syn_digest,
        )?;
        verify_asset_digest(
            "connectivity",
            connectivity_payload_digest(&conn_payload),
            conn_digest,
        )?;

        let morph = morphology_from_payload(&morph_payload)
            .map_err(|message| AssetBuildError::InvalidAssetData { message })?;
        let chan = channel_params_from_payload(&chan_payload)
            .map_err(|message| AssetBuildError::InvalidAssetData { message })?;
        let syn = synapse_params_from_payload(&syn_payload)
            .map_err(|message| AssetBuildError::InvalidAssetData { message })?;
        let conn = connectivity_from_payload(&conn_payload, &syn_payload)
            .map_err(|message| AssetBuildError::InvalidAssetData { message })?;

        if conn.edges.len() > MAX_ASSET_EDGES {
            return Err(AssetBuildError::BoundsExceeded {
                label: "edges",
                count: conn.edges.len(),
                max: MAX_ASSET_EDGES,
            });
        }
        if morph.neurons.len() != NEURON_COUNT {
            return Err(AssetBuildError::InvalidAssetData {
                message: format!(
                    "expected {NEURON_COUNT} neurons, got {}",
                    morph.neurons.len()
                ),
            });
        }

        let mut circuit = Self::build_from_assets(&morph, &chan, &syn, &conn)?;
        circuit.asset_manifest_digest = Some(manifest_digest);
        Ok(circuit)
    }

    fn build_from_parts(
        config: CircuitConfig,
        neurons: Vec<L4Neuron>,
        synapses: Vec<SynapseL4>,
        asset_manifest_digest: Option<[u8; 32]>,
        asset_pool_mapping_version: u32,
        asset_pool_mapping_digest: [u8; 32],
    ) -> Self {
        #[cfg(not(feature = "biophys-l4-hypothalamus-assets"))]
        let _ = (
            asset_manifest_digest,
            asset_pool_mapping_version,
            asset_pool_mapping_digest,
        );
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
        let limits = QueueLimits::new(
            MAX_EVENTS_PER_STEP.saturating_mul(max_delay as usize + 1),
            MAX_EVENTS_PER_STEP,
        );
        let queue = SpikeEventQueueL4::new(max_delay, limits);
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
            #[cfg(feature = "biophys-l4-hypothalamus-assets")]
            asset_manifest_digest,
            #[cfg(feature = "biophys-l4-hypothalamus-assets")]
            asset_pool_mapping_version,
            #[cfg(feature = "biophys-l4-hypothalamus-assets")]
            asset_pool_mapping_digest,
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
        for (idx, state) in self.syn_states.iter_mut().enumerate() {
            let synapse = &self.synapses[idx];
            let decay = self.syn_decay[idx];
            state.decay(synapse.kind, decay, synapse.tau_decay_nmda_steps);
        }

        let events = self.queue.drain_current(self.state.step_count);
        for event in events {
            let g_max_eff = self.syn_g_max_eff[event.synapse_index];
            let synapse = &self.synapses[event.synapse_index];
            self.syn_states[event.synapse_index].apply_spike(
                synapse.kind,
                g_max_eff,
                event.release_gain_q,
            );
        }

        let mut accumulators =
            vec![vec![SynapseAccumulator::default(); COMPARTMENT_COUNT]; NEURON_COUNT];
        for (idx, synapse) in self.synapses.iter().enumerate() {
            let mut g_fixed = self.syn_states[idx].g_fixed_for(synapse.kind);
            if synapse.kind == SynKind::GABA {
                g_fixed = scale_fixed_by_level(g_fixed, self.current_modulators.ht);
            }
            if g_fixed == 0 {
                continue;
            }
            let post = synapse.post_neuron as usize;
            let compartment = synapse.post_compartment as usize;
            accumulators[post][compartment].add(
                synapse.kind,
                g_fixed,
                synapse.e_rev,
                synapse.nmda_vdep_mode,
            );
        }

        let mut spikes = Vec::new();
        for (idx, neuron) in self.neurons.iter_mut().enumerate() {
            let syn_input = &accumulators[idx];
            let mut input = injected_currents[idx];
            for comp in 0..COMPARTMENT_COUNT {
                input[comp] += syn_input[comp].total_current(neuron.state.comp_v[comp]);
            }
            neuron.solver.step(&mut neuron.state, &input);
            sanitize_voltages(&mut neuron.state);
            let v = neuron.state.comp_v[0];
            if neuron.last_soma_v < THRESHOLD_MV && v >= THRESHOLD_MV {
                spikes.push(idx);
            }
            neuron.last_soma_v = v;
        }

        let delay_steps: Vec<u16> = self.synapses.iter().map(|syn| syn.delay_steps).collect();
        for spike_idx in &spikes {
            let indices = &self.pre_index[*spike_idx];
            self.queue.schedule_spike(
                self.state.step_count,
                indices,
                |idx| delay_steps[idx],
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
        if self.state.last_queue_health.overflowed {
            codes.push("RC.GV.BIO.QUEUE_OVERFLOW".to_string());
        }
        if self.state.last_queue_health.dropped_events > 0 {
            codes.push("RC.GV.BIO.EVENTS_DROPPED".to_string());
        }
        if self.state.last_queue_health.compacted {
            codes.push("RC.GV.BIO.QUEUE_COMPACTED".to_string());
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

        self.state.last_queue_health = self.queue.finish_tick();

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
            update_u32(&mut hasher, neuron.state.comp_v.len() as u32);
            for idx in 0..neuron.state.comp_v.len() {
                update_f32(&mut hasher, neuron.state.comp_v[idx]);
                update_f32(&mut hasher, neuron.state.m_q[idx] as f32 / 1000.0);
                update_f32(&mut hasher, neuron.state.h_q[idx] as f32 / 1000.0);
                update_f32(&mut hasher, neuron.state.n_q[idx] as f32 / 1000.0);
            }
        }
        for (synapse, state) in self.synapses.iter().zip(self.syn_states.iter()) {
            update_u32(&mut hasher, synapse.pre_neuron);
            update_u32(&mut hasher, synapse.post_neuron);
            update_u32(&mut hasher, synapse.post_compartment);
            update_u32(&mut hasher, synapse.g_max_base_q);
            update_u32(&mut hasher, synapse.g_nmda_base_q);
            update_f32(&mut hasher, synapse.e_rev);
            update_u32(&mut hasher, synapse.delay_steps as u32);
            update_u32(&mut hasher, state.g_ampa_q);
            update_u32(&mut hasher, state.g_nmda_q);
            update_u32(&mut hasher, state.g_gaba_q);
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
        #[cfg(feature = "biophys-l4-hypothalamus-assets")]
        {
            let digest = self.asset_manifest_digest.unwrap_or([0u8; 32]);
            hasher.update(&digest);
            update_u32(&mut hasher, self.asset_pool_mapping_version);
            hasher.update(&self.asset_pool_mapping_digest);
        }
        for neuron in &self.neurons {
            hasher.update(&neuron.solver.config_digest());
        }
        for synapse in &self.synapses {
            update_u32(&mut hasher, synapse.pre_neuron);
            update_u32(&mut hasher, synapse.post_neuron);
            update_u32(&mut hasher, synapse.post_compartment);
            update_u32(&mut hasher, synapse.g_max_base_q);
            update_u32(&mut hasher, synapse.g_nmda_base_q);
            update_f32(&mut hasher, synapse.e_rev);
            update_f32(&mut hasher, synapse.tau_decay_ms);
            update_u32(&mut hasher, synapse.tau_decay_nmda_steps as u32);
            update_u32(&mut hasher, synapse.nmda_vdep_mode as u32);
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

fn build_morphology(neuron_id: u32) -> NeuronMorphology {
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

    NeuronMorphology {
        neuron_id: NeuronId(neuron_id),
        compartments,
    }
}

fn default_targeting_policy() -> TargetingPolicy {
    TargetingPolicy {
        ampa_rule: TargetRule::ProximalDendrite,
        nmda_rule: TargetRule::DistalDendrite,
        gaba_rule: TargetRule::SomaOnly,
        seed_digest: *blake3::hash(b"UCF:L4:TARGETING:HYPOTHALAMUS").as_bytes(),
    }
}

fn build_neuron(neuron_id: u32) -> L4Neuron {
    let morphology = build_morphology(neuron_id);

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
            #[cfg(feature = "biophys-l4-ca")]
            ca: None,
        },
        CompartmentChannels {
            leak,
            nak: None,
            #[cfg(feature = "biophys-l4-ca")]
            ca: None,
        },
        CompartmentChannels {
            leak,
            nak: None,
            #[cfg(feature = "biophys-l4-ca")]
            ca: None,
        },
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

fn build_synapses() -> Vec<SynapseL4> {
    let mut synapses = Vec::new();
    let morphologies = (0..NEURON_COUNT)
        .map(|idx| build_morphology(idx as u32))
        .collect::<Vec<_>>();
    let policy = default_targeting_policy();

    for pool in 0..POOL_COUNT {
        let (start, end) = HypothalamusL4Microcircuit::pool_bounds(pool);
        for pre in start..end {
            for (post, morphology) in morphologies.iter().enumerate().take(end).skip(start) {
                if pre == post {
                    continue;
                }
                let (stp_params, stp_state) = within_pool_stp();
                let edge_key = EdgeKey {
                    pre_neuron_id: NeuronId(pre as u32),
                    post_neuron_id: NeuronId(post as u32),
                    synapse_index: synapses.len() as u32,
                };
                let post_compartment =
                    select_post_compartment(morphology, SynKind::AMPA, &policy, edge_key).0;
                synapses.push(SynapseL4 {
                    pre_neuron: pre as u32,
                    post_neuron: post as u32,
                    post_compartment,
                    kind: SynKind::AMPA,
                    mod_channel: ModChannel::Na,
                    g_max_base_q: f32_to_fixed_u32(AMPA_G_MAX_BASE),
                    g_nmda_base_q: 0,
                    g_max_min_q: 0,
                    g_max_max_q: max_synapse_g_fixed(),
                    e_rev: AMPA_E_REV,
                    tau_rise_ms: AMPA_TAU_RISE_MS,
                    tau_decay_ms: AMPA_TAU_DECAY_MS,
                    tau_decay_nmda_steps: 100,
                    nmda_vdep_mode: NmdaVDepMode::PiecewiseLinear,
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
            let edge_key = EdgeKey {
                pre_neuron_id: NeuronId(pre as u32),
                post_neuron_id: NeuronId(post as u32),
                synapse_index: synapses.len() as u32,
            };
            let post_compartment =
                select_post_compartment(&morphologies[post], SynKind::AMPA, &policy, edge_key).0;
            synapses.push(SynapseL4 {
                pre_neuron: pre as u32,
                post_neuron: post as u32,
                post_compartment,
                kind: SynKind::AMPA,
                mod_channel: ModChannel::Na,
                g_max_base_q: f32_to_fixed_u32(g_max),
                g_nmda_base_q: 0,
                g_max_min_q: 0,
                g_max_max_q: max_synapse_g_fixed(),
                e_rev: AMPA_E_REV,
                tau_rise_ms: AMPA_TAU_RISE_MS,
                tau_decay_ms: AMPA_TAU_DECAY_MS,
                tau_decay_nmda_steps: 100,
                nmda_vdep_mode: NmdaVDepMode::PiecewiseLinear,
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
        for (post, morphology) in morphologies.iter().enumerate().take(EXCITATORY_COUNT) {
            let (stp_params, stp_state) = disabled_stp();
            let edge_key = EdgeKey {
                pre_neuron_id: NeuronId(pre as u32),
                post_neuron_id: NeuronId(post as u32),
                synapse_index: synapses.len() as u32,
            };
            let post_compartment =
                select_post_compartment(morphology, SynKind::GABA, &policy, edge_key).0;
            synapses.push(SynapseL4 {
                pre_neuron: pre as u32,
                post_neuron: post as u32,
                post_compartment,
                kind: SynKind::GABA,
                mod_channel: ModChannel::Ht,
                g_max_base_q: f32_to_fixed_u32(GABA_G_MAX),
                g_nmda_base_q: 0,
                g_max_min_q: 0,
                g_max_max_q: max_synapse_g_fixed(),
                e_rev: GABA_E_REV,
                tau_rise_ms: GABA_TAU_RISE_MS,
                tau_decay_ms: GABA_TAU_DECAY_MS,
                tau_decay_nmda_steps: 100,
                nmda_vdep_mode: NmdaVDepMode::PiecewiseLinear,
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
    for v in &mut state.comp_v {
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

#[cfg(feature = "biophys-l4-hypothalamus-assets")]
impl CircuitBuilderFromAssets for HypothalamusL4Microcircuit {
    fn build_from_assets(
        morph: &MorphologySet,
        chan: &ChannelParamsSet,
        syn: &SynapseParamsSet,
        conn: &ConnectivityGraph,
    ) -> Result<Self, AssetBuildError> {
        let labels_present = morph.neurons.iter().any(|neuron| !neuron.labels.is_empty());
        let (pool_map, pool_mapping_version) = if labels_present {
            (pool_map_from_labels(morph)?, ASSET_POOL_CONVENTION_V2)
        } else {
            (pool_map_from_ranges(morph)?, ASSET_POOL_CONVENTION_V1)
        };
        let pool_mapping_digest = pool_map_digest(&pool_map);
        let ordered_neuron_ids = ordered_neurons_from_pool_map(&pool_map);

        let mut asset_to_internal = std::collections::BTreeMap::new();
        for (internal_idx, asset_id) in ordered_neuron_ids.iter().enumerate() {
            asset_to_internal.insert(*asset_id, internal_idx as u32);
        }

        let mut morph_map = std::collections::BTreeMap::new();
        for neuron in &morph.neurons {
            morph_map.insert(neuron.neuron_id, neuron);
        }

        let mut channel_map = std::collections::BTreeMap::new();
        for params in &chan.per_compartment {
            channel_map.insert((params.neuron_id, params.comp_id), params);
        }

        let mut morphologies = Vec::with_capacity(ordered_neuron_ids.len());
        for (internal_idx, asset_id) in ordered_neuron_ids.iter().enumerate() {
            let neuron = morph_map
                .get(asset_id)
                .ok_or(AssetBuildError::UnknownNeuron {
                    neuron_id: *asset_id,
                })?;
            let mut compartments = neuron.compartments.clone();
            compartments.sort_by_key(|comp| comp.comp_id);
            let depths = compute_compartment_depths(&compartments)?;
            let morph_comps = compartments
                .iter()
                .map(|comp| {
                    let kind = match comp.kind {
                        biophys_assets::CompartmentKind::Soma => CompartmentKind::Soma,
                        biophys_assets::CompartmentKind::Dendrite => CompartmentKind::Dendrite,
                        biophys_assets::CompartmentKind::Axon => CompartmentKind::Axon,
                    };
                    Compartment {
                        id: CompartmentId(comp.comp_id),
                        parent: comp.parent.map(CompartmentId),
                        kind,
                        depth: *depths.get(&comp.comp_id).unwrap_or(&0),
                        capacitance: asset_compartment_capacitance(comp.kind),
                        axial_resistance: asset_compartment_axial_resistance(comp.kind),
                    }
                })
                .collect::<Vec<_>>();
            let morphology = NeuronMorphology {
                neuron_id: NeuronId(internal_idx as u32),
                compartments: morph_comps,
            };
            morphology
                .validate(biophys_morphology::MAX_COMPARTMENTS)
                .map_err(|error| AssetBuildError::InvalidAssetData {
                    message: format!("morphology validation failed: {error:?}"),
                })?;
            morphologies.push(morphology);
        }

        let mut neurons = Vec::with_capacity(morphologies.len());
        for (internal_idx, morphology) in morphologies.iter().enumerate() {
            let asset_id = ordered_neuron_ids
                .get(internal_idx)
                .copied()
                .unwrap_or(morphology.neuron_id.0);
            let channels = morphology
                .compartments
                .iter()
                .map(|comp| {
                    let params = channel_map.get(&(asset_id, comp.id.0)).ok_or(
                        AssetBuildError::MissingChannelParams {
                            neuron_id: asset_id,
                            comp_id: comp.id.0,
                        },
                    )?;
                    let leak = Leak {
                        g: params.leak_g as f32 * LEAK_G_SCALE,
                        e_rev: -65.0,
                    };
                    let nak = if params.na_g == 0 && params.k_g == 0 {
                        None
                    } else {
                        Some(NaK {
                            g_na: params.na_g as f32,
                            g_k: params.k_g as f32,
                            e_na: 50.0,
                            e_k: -77.0,
                        })
                    };
                    Ok(CompartmentChannels {
                        leak,
                        nak,
                        #[cfg(feature = "biophys-l4-ca")]
                        ca: None,
                    })
                })
                .collect::<Result<Vec<_>, AssetBuildError>>()?;

            let solver = L4Solver::new(morphology.clone(), channels, DT_MS, CLAMP_MIN, CLAMP_MAX)
                .map_err(|error| AssetBuildError::InvalidAssetData {
                message: format!("solver init failed: {error:?}"),
            })?;
            let state = L4State::new(-65.0, morphology.compartments.len());
            let last_soma_v = state.comp_v[0];
            neurons.push(L4Neuron {
                solver,
                state,
                last_soma_v,
            });
        }

        let mut edges = conn.edges.clone();
        edges.sort_by_key(|edge| {
            (
                edge.pre,
                edge.post,
                edge.syn_type as u8,
                edge.delay_steps,
                edge.syn_param_id,
            )
        });

        let policy = default_targeting_policy();
        let mut synapses = Vec::with_capacity(edges.len());
        for edge in edges {
            let pre = *asset_to_internal
                .get(&edge.pre)
                .ok_or(AssetBuildError::UnknownNeuron {
                    neuron_id: edge.pre,
                })? as usize;
            let post = *asset_to_internal
                .get(&edge.post)
                .ok_or(AssetBuildError::UnknownNeuron {
                    neuron_id: edge.post,
                })? as usize;
            let syn_params = syn.params.get(edge.syn_param_id as usize).ok_or(
                AssetBuildError::MissingSynapseParams {
                    syn_param_id: edge.syn_param_id,
                },
            )?;
            if syn_params.syn_type != edge.syn_type {
                return Err(AssetBuildError::InvalidAssetData {
                    message: format!("synapse type mismatch for edge {}->{}", edge.pre, edge.post),
                });
            }
            let kind = match edge.syn_type {
                biophys_assets::SynType::Exc => SynKind::AMPA,
                biophys_assets::SynType::Inh => SynKind::GABA,
            };
            let mod_channel = map_asset_mod_channel(syn_params.mod_channel, syn_params.syn_type);
            let g_max_base_q = f32_to_fixed_u32(syn_params.weight_base.abs() as f32);
            let (e_rev, tau_rise_ms, tau_decay_ms, stdp_enabled) = match kind {
                SynKind::AMPA => (AMPA_E_REV, AMPA_TAU_RISE_MS, AMPA_TAU_DECAY_MS, true),
                SynKind::GABA => (GABA_E_REV, GABA_TAU_RISE_MS, GABA_TAU_DECAY_MS, false),
                SynKind::NMDA => (AMPA_E_REV, AMPA_TAU_RISE_MS, AMPA_TAU_DECAY_MS, true),
            };
            let (stp_params, stp_state) = if syn_params.stp_u == 0 {
                disabled_stp()
            } else {
                let params = StpParamsL4 {
                    mode: biophys_synapses_l4::StpMode::STP_TM,
                    u_base_q: syn_params.stp_u.min(STP_SCALE),
                    tau_rec_steps: syn_params.tau_rec.max(1),
                    tau_fac_steps: syn_params.tau_fac.max(1),
                };
                let state = StpStateL4::new(params);
                (params, state)
            };
            let edge_key = EdgeKey {
                pre_neuron_id: NeuronId(pre as u32),
                post_neuron_id: NeuronId(post as u32),
                synapse_index: synapses.len() as u32,
            };
            let post_compartment =
                select_post_compartment(&morphologies[post], kind, &policy, edge_key).0;
            synapses.push(SynapseL4 {
                pre_neuron: pre as u32,
                post_neuron: post as u32,
                post_compartment,
                kind,
                mod_channel,
                g_max_base_q,
                g_nmda_base_q: 0,
                g_max_min_q: 0,
                g_max_max_q: max_synapse_g_fixed(),
                e_rev,
                tau_rise_ms,
                tau_decay_ms,
                tau_decay_nmda_steps: 100,
                nmda_vdep_mode: NmdaVDepMode::PiecewiseLinear,
                delay_steps: edge.delay_steps,
                stp_params,
                stp_state,
                stdp_enabled,
                stdp_trace: StdpTrace::default(),
            });
        }

        Ok(HypothalamusL4Microcircuit::build_from_parts(
            CircuitConfig::default(),
            neurons,
            synapses,
            None,
            pool_mapping_version,
            pool_mapping_digest,
        ))
    }
}

#[cfg(feature = "biophys-l4-hypothalamus-assets")]
type PoolMap = std::collections::BTreeMap<String, Vec<u32>>;

#[cfg(feature = "biophys-l4-hypothalamus-assets")]
fn pool_map_from_labels(morph: &MorphologySet) -> Result<PoolMap, AssetBuildError> {
    let mut pool_map: PoolMap = std::collections::BTreeMap::new();
    let mut seen = std::collections::BTreeSet::new();
    for neuron in &morph.neurons {
        if !seen.insert(neuron.neuron_id) {
            return Err(AssetBuildError::InvalidAssetData {
                message: format!("duplicate neuron id {}", neuron.neuron_id),
            });
        }
        let mut pool_label: Option<&str> = None;
        let mut role_label: Option<&str> = None;
        for label in &neuron.labels {
            if label.k == LABEL_KEY_POOL {
                if pool_label.replace(label.v.as_str()).is_some() {
                    return Err(AssetBuildError::InvalidAssetData {
                        message: format!("multiple pool labels for neuron {}", neuron.neuron_id),
                    });
                }
            } else if label.k == LABEL_KEY_ROLE && role_label.replace(label.v.as_str()).is_some() {
                return Err(AssetBuildError::InvalidAssetData {
                    message: format!("multiple role labels for neuron {}", neuron.neuron_id),
                });
            }
        }
        let pool = pool_label.ok_or_else(|| AssetBuildError::InvalidAssetData {
            message: format!("missing pool label for neuron {}", neuron.neuron_id),
        })?;
        let role = role_label.ok_or_else(|| AssetBuildError::InvalidAssetData {
            message: format!("missing role label for neuron {}", neuron.neuron_id),
        })?;
        let expected_role = if pool == "INH" {
            ROLE_INHIBITORY
        } else {
            ROLE_EXCITATORY
        };
        if role != expected_role {
            return Err(AssetBuildError::InvalidAssetData {
                message: format!(
                    "role label {role} invalid for pool {pool} on neuron {}",
                    neuron.neuron_id
                ),
            });
        }
        if !POOL_LABELS.contains(&pool) {
            return Err(AssetBuildError::InvalidAssetData {
                message: format!("unknown pool label {pool} for neuron {}", neuron.neuron_id),
            });
        }
        pool_map
            .entry(pool.to_string())
            .or_default()
            .push(neuron.neuron_id);
    }

    for pool in POOL_LABELS {
        let expected = if pool == "INH" {
            INHIBITORY_COUNT
        } else {
            POOL_SIZE
        };
        let members = pool_map
            .get_mut(pool)
            .ok_or_else(|| AssetBuildError::InvalidAssetData {
                message: format!("missing required pool {pool}"),
            })?;
        members.sort_unstable();
        if members.len() != expected {
            return Err(AssetBuildError::InvalidAssetData {
                message: format!(
                    "pool {pool} expected {expected} neurons, got {}",
                    members.len()
                ),
            });
        }
    }
    Ok(pool_map)
}

#[cfg(feature = "biophys-l4-hypothalamus-assets")]
fn pool_map_from_ranges(morph: &MorphologySet) -> Result<PoolMap, AssetBuildError> {
    let mut seen = std::collections::BTreeSet::new();
    for neuron in &morph.neurons {
        if !seen.insert(neuron.neuron_id) {
            return Err(AssetBuildError::InvalidAssetData {
                message: format!("duplicate neuron id {}", neuron.neuron_id),
            });
        }
    }
    if seen.len() != NEURON_COUNT {
        return Err(AssetBuildError::InvalidAssetData {
            message: format!("expected {NEURON_COUNT} neurons, got {}", seen.len()),
        });
    }
    for neuron_id in 0..NEURON_COUNT as u32 {
        if !seen.contains(&neuron_id) {
            return Err(AssetBuildError::InvalidAssetData {
                message: "Asset Convention v1 requires neuron ids 0..N-1".to_string(),
            });
        }
    }

    let mut pool_map: PoolMap = std::collections::BTreeMap::new();
    pool_map.insert("P0".to_string(), (0..POOL_SIZE as u32).collect());
    pool_map.insert(
        "P1".to_string(),
        ((POOL_SIZE as u32)..(2 * POOL_SIZE) as u32).collect(),
    );
    pool_map.insert(
        "P2".to_string(),
        ((2 * POOL_SIZE) as u32..(3 * POOL_SIZE) as u32).collect(),
    );
    pool_map.insert(
        "P3".to_string(),
        ((3 * POOL_SIZE) as u32..(4 * POOL_SIZE) as u32).collect(),
    );
    pool_map.insert(
        "O_SIM".to_string(),
        ((4 * POOL_SIZE) as u32..(5 * POOL_SIZE) as u32).collect(),
    );
    pool_map.insert(
        "O_EXP".to_string(),
        ((5 * POOL_SIZE) as u32..(6 * POOL_SIZE) as u32).collect(),
    );
    pool_map.insert(
        "O_NOV".to_string(),
        ((6 * POOL_SIZE) as u32..(7 * POOL_SIZE) as u32).collect(),
    );
    pool_map.insert(
        "INH".to_string(),
        (EXCITATORY_COUNT as u32..NEURON_COUNT as u32).collect(),
    );
    Ok(pool_map)
}

#[cfg(feature = "biophys-l4-hypothalamus-assets")]
fn ordered_neurons_from_pool_map(pool_map: &PoolMap) -> Vec<u32> {
    let mut ordered = Vec::with_capacity(NEURON_COUNT);
    for pool in POOL_LABELS {
        if let Some(neurons) = pool_map.get(pool) {
            ordered.extend_from_slice(neurons);
        }
    }
    ordered
}

#[cfg(feature = "biophys-l4-hypothalamus-assets")]
fn pool_map_digest(pool_map: &PoolMap) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"UCF:BIO:L4:HYPO:POOLMAP");
    for pool in POOL_LABELS {
        if let Some(neurons) = pool_map.get(pool) {
            update_u32(&mut hasher, pool.len() as u32);
            hasher.update(pool.as_bytes());
            update_u32(&mut hasher, neurons.len() as u32);
            for neuron_id in neurons {
                update_u32(&mut hasher, *neuron_id);
            }
        }
    }
    *hasher.finalize().as_bytes()
}

#[cfg(feature = "biophys-l4-hypothalamus-assets")]
fn asset_compartment_capacitance(kind: biophys_assets::CompartmentKind) -> f32 {
    match kind {
        biophys_assets::CompartmentKind::Soma => 1.0,
        biophys_assets::CompartmentKind::Dendrite => 1.2,
        biophys_assets::CompartmentKind::Axon => 1.0,
    }
}

#[cfg(feature = "biophys-l4-hypothalamus-assets")]
fn asset_compartment_axial_resistance(kind: biophys_assets::CompartmentKind) -> f32 {
    match kind {
        biophys_assets::CompartmentKind::Soma => 150.0,
        biophys_assets::CompartmentKind::Dendrite => 200.0,
        biophys_assets::CompartmentKind::Axon => 150.0,
    }
}

#[cfg(feature = "biophys-l4-hypothalamus-assets")]
fn compute_compartment_depths(
    compartments: &[biophys_assets::Compartment],
) -> Result<std::collections::BTreeMap<u32, u32>, AssetBuildError> {
    let mut parent_map = std::collections::BTreeMap::new();
    for comp in compartments {
        parent_map.insert(comp.comp_id, comp.parent);
    }
    let mut depths = std::collections::BTreeMap::new();
    let mut visiting = std::collections::BTreeSet::new();
    for comp in compartments {
        compute_depth_for_comp(comp.comp_id, &parent_map, &mut depths, &mut visiting)?;
    }
    Ok(depths)
}

#[cfg(feature = "biophys-l4-hypothalamus-assets")]
fn compute_depth_for_comp(
    comp_id: u32,
    parent_map: &std::collections::BTreeMap<u32, Option<u32>>,
    depths: &mut std::collections::BTreeMap<u32, u32>,
    visiting: &mut std::collections::BTreeSet<u32>,
) -> Result<u32, AssetBuildError> {
    if let Some(depth) = depths.get(&comp_id) {
        return Ok(*depth);
    }
    if !visiting.insert(comp_id) {
        return Err(AssetBuildError::InvalidAssetData {
            message: format!("cycle detected at compartment {comp_id}"),
        });
    }
    let depth = match parent_map.get(&comp_id).copied().flatten() {
        None => 0,
        Some(parent) => {
            if !parent_map.contains_key(&parent) {
                return Err(AssetBuildError::InvalidAssetData {
                    message: format!("missing parent compartment {parent} for {comp_id}"),
                });
            }
            compute_depth_for_comp(parent, parent_map, depths, visiting)?.saturating_add(1)
        }
    };
    visiting.remove(&comp_id);
    depths.insert(comp_id, depth);
    Ok(depth)
}

#[cfg(feature = "biophys-l4-hypothalamus-assets")]
fn digest_from_vec(bytes: &[u8], label: &'static str) -> Result<[u8; 32], AssetBuildError> {
    if bytes.len() != 32 {
        return Err(AssetBuildError::InvalidDigestLength {
            label,
            len: bytes.len(),
        });
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(bytes);
    Ok(out)
}

#[cfg(feature = "biophys-l4-hypothalamus-assets")]
fn manifest_digest_for_kind(
    manifest: &ucf::v1::AssetManifest,
    kind: AssetKind,
) -> Result<[u8; 32], AssetBuildError> {
    let component = manifest
        .components
        .iter()
        .find(|component| component.kind == kind as i32)
        .ok_or(AssetBuildError::MissingAssetDigest {
            kind: kind_name(kind),
        })?;
    digest_from_vec(&component.digest, "asset_digest")
}

#[cfg(feature = "biophys-l4-hypothalamus-assets")]
fn kind_name(kind: AssetKind) -> &'static str {
    match kind {
        AssetKind::MorphologySet => "morphology",
        AssetKind::ChannelParamsSet => "channel_params",
        AssetKind::SynapseParamsSet => "synapse_params",
        AssetKind::ConnectivityGraph => "connectivity",
        AssetKind::Unknown => "unknown",
    }
}

#[cfg(feature = "biophys-l4-hypothalamus-assets")]
fn verify_asset_digest(
    kind: &'static str,
    computed: [u8; 32],
    expected: [u8; 32],
) -> Result<(), AssetBuildError> {
    if computed != expected {
        return Err(AssetBuildError::AssetDigestMismatch { kind });
    }
    Ok(())
}

#[cfg(feature = "biophys-l4-hypothalamus-assets")]
fn map_asset_mod_channel(
    mod_channel: biophys_assets::ModChannel,
    syn_type: biophys_assets::SynType,
) -> ModChannel {
    match mod_channel {
        biophys_assets::ModChannel::None => ModChannel::None,
        biophys_assets::ModChannel::A => ModChannel::NaDa,
        biophys_assets::ModChannel::B => match syn_type {
            biophys_assets::SynType::Exc => ModChannel::Na,
            biophys_assets::SynType::Inh => ModChannel::Ht,
        },
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

#[cfg(all(
    test,
    feature = "biophys",
    feature = "biophys-l4",
    feature = "biophys-l4-synapses",
    feature = "biophys-l4-hypothalamus",
    feature = "biophys-l4-hypothalamus-assets"
))]
mod asset_tests {
    use super::{AssetBuildError, *};
    use asset_chunker::{
        build_asset_bundle_with_policy, chunk_asset, BundleIdPolicy, ChunkerConfig,
    };
    use asset_rehydration::ASSET_MANIFEST_DOMAIN;
    use biophys_assets::{
        to_asset_digest, ChannelParams, ChannelParamsSet, Compartment as AssetCompartment,
        CompartmentKind as AssetCompartmentKind, ConnEdge, ConnectivityGraph, LabelKV, ModChannel,
        MorphNeuron, MorphologySet, SynType, SynapseParams, SynapseParamsSet,
    };
    use prost::Message;
    use ucf::v1::{AssetBundle, AssetKind, AssetManifest, Compression};

    fn profile_rank(profile: ProfileState) -> u8 {
        match profile {
            ProfileState::M0 => 0,
            ProfileState::M1 => 1,
            ProfileState::M2 => 2,
            ProfileState::M3 => 3,
        }
    }

    fn hypo_morphology_assets(neuron_count: usize) -> MorphologySet {
        let neurons = (0..neuron_count as u32)
            .map(|neuron_id| {
                let pool = match neuron_id as usize {
                    0..=1 => "P0",
                    2..=3 => "P1",
                    4..=5 => "P2",
                    6..=7 => "P3",
                    8..=9 => "O_SIM",
                    10..=11 => "O_EXP",
                    12..=13 => "O_NOV",
                    _ => "INH",
                };
                let role = if neuron_id as usize >= EXCITATORY_COUNT {
                    "I"
                } else {
                    "E"
                };
                let compartments = vec![
                    AssetCompartment {
                        comp_id: 0,
                        parent: None,
                        kind: AssetCompartmentKind::Soma,
                        length_um: 10,
                        diameter_um: 8,
                    },
                    AssetCompartment {
                        comp_id: 1,
                        parent: Some(0),
                        kind: AssetCompartmentKind::Dendrite,
                        length_um: 20,
                        diameter_um: 4,
                    },
                    AssetCompartment {
                        comp_id: 2,
                        parent: Some(0),
                        kind: AssetCompartmentKind::Dendrite,
                        length_um: 20,
                        diameter_um: 4,
                    },
                ];
                MorphNeuron {
                    neuron_id,
                    compartments,
                    labels: vec![
                        LabelKV {
                            k: "pool".to_string(),
                            v: pool.to_string(),
                        },
                        LabelKV {
                            k: "role".to_string(),
                            v: role.to_string(),
                        },
                    ],
                }
            })
            .collect();
        MorphologySet {
            version: 2,
            neurons,
        }
    }

    fn hypo_channel_params_assets(morph: &MorphologySet) -> ChannelParamsSet {
        let mut per_compartment = Vec::new();
        for neuron in &morph.neurons {
            for comp in &neuron.compartments {
                let (na_g, k_g) = if comp.kind == AssetCompartmentKind::Soma {
                    (120, 36)
                } else {
                    (0, 0)
                };
                per_compartment.push(ChannelParams {
                    neuron_id: neuron.neuron_id,
                    comp_id: comp.comp_id,
                    leak_g: 1,
                    na_g,
                    k_g,
                });
            }
        }
        ChannelParamsSet {
            version: 1,
            per_compartment,
        }
    }

    fn hypo_synapse_params_assets() -> SynapseParamsSet {
        let p1_weight = (AMPA_G_MAX_BASE * 1.1).round() as i32;
        let p2_weight = (AMPA_G_MAX_DOMINANT * 0.9).round() as i32;
        SynapseParamsSet {
            version: 1,
            params: vec![
                SynapseParams {
                    syn_type: SynType::Exc,
                    weight_base: AMPA_G_MAX_BASE as i32,
                    stp_u: 0,
                    tau_rec: 1,
                    tau_fac: 1,
                    mod_channel: ModChannel::B,
                },
                SynapseParams {
                    syn_type: SynType::Exc,
                    weight_base: p1_weight,
                    stp_u: 0,
                    tau_rec: 1,
                    tau_fac: 1,
                    mod_channel: ModChannel::B,
                },
                SynapseParams {
                    syn_type: SynType::Exc,
                    weight_base: p2_weight,
                    stp_u: 0,
                    tau_rec: 1,
                    tau_fac: 1,
                    mod_channel: ModChannel::B,
                },
                SynapseParams {
                    syn_type: SynType::Exc,
                    weight_base: AMPA_G_MAX_DOMINANT as i32,
                    stp_u: 0,
                    tau_rec: 1,
                    tau_fac: 1,
                    mod_channel: ModChannel::B,
                },
                SynapseParams {
                    syn_type: SynType::Inh,
                    weight_base: GABA_G_MAX as i32,
                    stp_u: 0,
                    tau_rec: 1,
                    tau_fac: 1,
                    mod_channel: ModChannel::B,
                },
            ],
        }
    }

    fn hypo_connectivity_assets() -> ConnectivityGraph {
        let mut edges = Vec::new();
        for pool in 0..POOL_COUNT {
            let (start, end) = HypothalamusL4Microcircuit::pool_bounds(pool);
            for pre in start..end {
                for post in start..end {
                    if pre == post {
                        continue;
                    }
                    edges.push(ConnEdge {
                        pre: pre as u32,
                        post: post as u32,
                        syn_type: SynType::Exc,
                        delay_steps: 1,
                        syn_param_id: 0,
                    });
                }
            }
        }

        for pre in 0..EXCITATORY_COUNT {
            let pool = pre / POOL_SIZE;
            let syn_param_id = match pool {
                IDX_P1 => 1,
                IDX_P2 => 2,
                IDX_P3 => 3,
                _ => 0,
            };
            for inh in 0..INHIBITORY_COUNT {
                let post = EXCITATORY_COUNT + inh;
                edges.push(ConnEdge {
                    pre: pre as u32,
                    post: post as u32,
                    syn_type: SynType::Exc,
                    delay_steps: 1,
                    syn_param_id,
                });
            }
        }

        for inh in 0..INHIBITORY_COUNT {
            let pre = EXCITATORY_COUNT + inh;
            for post in 0..EXCITATORY_COUNT {
                edges.push(ConnEdge {
                    pre: pre as u32,
                    post: post as u32,
                    syn_type: SynType::Inh,
                    delay_steps: 1,
                    syn_param_id: 4,
                });
            }
        }

        ConnectivityGraph { version: 1, edges }
    }

    fn hypo_empty_connectivity_assets() -> ConnectivityGraph {
        ConnectivityGraph {
            version: 1,
            edges: Vec::new(),
        }
    }

    fn compute_manifest_digest(manifest: &AssetManifest) -> [u8; 32] {
        let mut normalized = manifest.clone();
        normalized.manifest_digest = vec![0u8; 32];
        let mut hasher = blake3::Hasher::new();
        hasher.update(ASSET_MANIFEST_DOMAIN.as_bytes());
        hasher.update(&normalized.encode_to_vec());
        *hasher.finalize().as_bytes()
    }

    fn build_asset_bundle(
        morph: &MorphologySet,
        chan: &ChannelParamsSet,
        syn: &SynapseParamsSet,
        conn: &ConnectivityGraph,
        conn_digest_override: Option<[u8; 32]>,
        conn_bytes_override: Option<Vec<u8>>,
    ) -> AssetBundle {
        let created_at_ms = 10;
        let conn_digest = conn_digest_override.unwrap_or_else(|| conn.digest());
        let mut manifest = AssetManifest {
            manifest_version: 1,
            created_at_ms,
            manifest_digest: vec![0u8; 32],
            components: vec![
                to_asset_digest(
                    AssetKind::MorphologySet,
                    morph.version,
                    morph.digest(),
                    created_at_ms,
                    None,
                ),
                to_asset_digest(
                    AssetKind::ChannelParamsSet,
                    chan.version,
                    chan.digest(),
                    created_at_ms,
                    None,
                ),
                to_asset_digest(
                    AssetKind::SynapseParamsSet,
                    syn.version,
                    syn.digest(),
                    created_at_ms,
                    None,
                ),
                to_asset_digest(
                    AssetKind::ConnectivityGraph,
                    conn.version,
                    conn_digest,
                    created_at_ms,
                    None,
                ),
            ],
        };
        let manifest_digest = compute_manifest_digest(&manifest);
        manifest.manifest_digest = manifest_digest.to_vec();

        let chunker = ChunkerConfig {
            max_chunk_bytes: 128,
            compression: Compression::None,
            max_chunks_total: 2048,
            bundle_id_policy: BundleIdPolicy::ManifestDigestPrefix { prefix_len: 8 },
        };
        let mut chunks = Vec::new();
        chunks.extend(
            chunk_asset(
                AssetKind::MorphologySet,
                morph.version,
                morph.digest(),
                &morph.to_canonical_bytes(),
                &chunker,
                created_at_ms,
            )
            .expect("morph chunks"),
        );
        chunks.extend(
            chunk_asset(
                AssetKind::ChannelParamsSet,
                chan.version,
                chan.digest(),
                &chan.to_canonical_bytes(),
                &chunker,
                created_at_ms,
            )
            .expect("channel chunks"),
        );
        chunks.extend(
            chunk_asset(
                AssetKind::SynapseParamsSet,
                syn.version,
                syn.digest(),
                &syn.to_canonical_bytes(),
                &chunker,
                created_at_ms,
            )
            .expect("syn chunks"),
        );
        let conn_bytes = conn_bytes_override.unwrap_or_else(|| conn.to_canonical_bytes());
        chunks.extend(
            chunk_asset(
                AssetKind::ConnectivityGraph,
                conn.version,
                conn_digest,
                &conn_bytes,
                &chunker,
                created_at_ms,
            )
            .expect("conn chunks"),
        );
        chunks.sort_by(|a, b| {
            a.asset_digest
                .cmp(&b.asset_digest)
                .then_with(|| a.chunk_index.cmp(&b.chunk_index))
        });

        build_asset_bundle_with_policy(
            manifest,
            chunks,
            created_at_ms,
            BundleIdPolicy::ManifestDigestPrefix { prefix_len: 8 },
        )
    }

    fn demo_inputs() -> Vec<HypoInput> {
        vec![
            HypoInput::default(),
            HypoInput {
                isv: dbm_core::IsvSnapshot {
                    policy_pressure: LevelClass::High,
                    ..dbm_core::IsvSnapshot::default()
                },
                ..HypoInput::default()
            },
            HypoInput {
                isv: dbm_core::IsvSnapshot {
                    threat: LevelClass::High,
                    ..dbm_core::IsvSnapshot::default()
                },
                ..HypoInput::default()
            },
            HypoInput {
                isv: dbm_core::IsvSnapshot {
                    integrity: IntegrityState::Fail,
                    ..dbm_core::IsvSnapshot::default()
                },
                ..HypoInput::default()
            },
        ]
    }

    #[test]
    fn asset_build_is_deterministic() {
        let morph = hypo_morphology_assets(NEURON_COUNT);
        let chan = hypo_channel_params_assets(&morph);
        let syn = hypo_synapse_params_assets();
        let conn = hypo_connectivity_assets();
        let bundle = build_asset_bundle(&morph, &chan, &syn, &conn, None, None);
        let rehydrator = AssetRehydrator::new();
        let inputs = demo_inputs();

        let mut a =
            HypothalamusL4Microcircuit::new_from_asset_bundle(&bundle, &rehydrator).expect("asset");
        let mut b =
            HypothalamusL4Microcircuit::new_from_asset_bundle(&bundle, &rehydrator).expect("asset");

        let outputs_a = inputs
            .iter()
            .map(|input| {
                let output = a.step(input, 0);
                (
                    output.profile_state,
                    output.overlays.clone(),
                    a.config_digest(),
                )
            })
            .collect::<Vec<_>>();
        let outputs_b = inputs
            .iter()
            .map(|input| {
                let output = b.step(input, 0);
                (
                    output.profile_state,
                    output.overlays.clone(),
                    b.config_digest(),
                )
            })
            .collect::<Vec<_>>();

        assert_eq!(outputs_a, outputs_b);
    }

    #[test]
    fn asset_pool_mapping_uses_labels() {
        let mut morph = hypo_morphology_assets(NEURON_COUNT);
        for neuron in &mut morph.neurons {
            for label in &mut neuron.labels {
                if label.k == "pool" {
                    label.v = match label.v.as_str() {
                        "P0" => "P1".to_string(),
                        "P1" => "P0".to_string(),
                        other => other.to_string(),
                    };
                }
            }
        }
        let chan = hypo_channel_params_assets(&morph);
        let syn = hypo_synapse_params_assets();
        let conn = hypo_empty_connectivity_assets();

        let circuit = <HypothalamusL4Microcircuit as CircuitBuilderFromAssets>::build_from_assets(
            &morph, &chan, &syn, &conn,
        )
        .expect("asset build");
        let pool_map = pool_map_from_labels(&morph).expect("pool map");
        assert_eq!(circuit.asset_pool_mapping_version, ASSET_POOL_CONVENTION_V2);
        assert_eq!(
            circuit.asset_pool_mapping_digest,
            pool_map_digest(&pool_map)
        );
    }

    #[test]
    fn asset_pool_mapping_v1_fallback_accepts_unlabeled_bundle() {
        let mut morph = hypo_morphology_assets(NEURON_COUNT);
        morph.version = 1;
        for neuron in &mut morph.neurons {
            neuron.labels.clear();
        }
        let chan = hypo_channel_params_assets(&morph);
        let syn = hypo_synapse_params_assets();
        let conn = hypo_connectivity_assets();
        let bundle = build_asset_bundle(&morph, &chan, &syn, &conn, None, None);
        let rehydrator = AssetRehydrator::new();

        let circuit =
            HypothalamusL4Microcircuit::new_from_asset_bundle(&bundle, &rehydrator).expect("asset");
        let pool_map = pool_map_from_ranges(&morph).expect("pool map");
        assert_eq!(circuit.asset_pool_mapping_version, ASSET_POOL_CONVENTION_V1);
        assert_eq!(
            circuit.asset_pool_mapping_digest,
            pool_map_digest(&pool_map)
        );
    }

    #[test]
    fn asset_pool_mapping_rejects_missing_pool_label() {
        let mut morph = hypo_morphology_assets(NEURON_COUNT);
        if let Some(neuron) = morph.neurons.get_mut(0) {
            neuron.labels.retain(|label| label.k != "pool");
        }
        let chan = hypo_channel_params_assets(&morph);
        let syn = hypo_synapse_params_assets();
        let conn = hypo_empty_connectivity_assets();

        let err = <HypothalamusL4Microcircuit as CircuitBuilderFromAssets>::build_from_assets(
            &morph, &chan, &syn, &conn,
        )
        .expect_err("missing pool label should fail");
        assert!(matches!(err, AssetBuildError::InvalidAssetData { .. }));
    }

    #[test]
    fn asset_pool_mapping_rejects_duplicate_pool_labels() {
        let mut morph = hypo_morphology_assets(NEURON_COUNT);
        if let Some(neuron) = morph.neurons.get_mut(0) {
            neuron.labels.push(LabelKV {
                k: "pool".to_string(),
                v: "P0".to_string(),
            });
        }
        let chan = hypo_channel_params_assets(&morph);
        let syn = hypo_synapse_params_assets();
        let conn = hypo_empty_connectivity_assets();

        let err = <HypothalamusL4Microcircuit as CircuitBuilderFromAssets>::build_from_assets(
            &morph, &chan, &syn, &conn,
        )
        .expect_err("duplicate pool label should fail");
        assert!(matches!(err, AssetBuildError::InvalidAssetData { .. }));
    }

    #[test]
    fn asset_backed_obeys_demo_outputs() {
        let morph = hypo_morphology_assets(NEURON_COUNT);
        let chan = hypo_channel_params_assets(&morph);
        let syn = hypo_synapse_params_assets();
        let conn = hypo_connectivity_assets();
        let bundle = build_asset_bundle(&morph, &chan, &syn, &conn, None, None);
        let rehydrator = AssetRehydrator::new();
        let inputs = demo_inputs();

        let expected_floor = [
            ProfileState::M0,
            ProfileState::M1,
            ProfileState::M2,
            ProfileState::M3,
        ];

        for (idx, (input, expected_profile)) in inputs.iter().zip(expected_floor.iter()).enumerate()
        {
            let mut hardcoded = HypothalamusL4Microcircuit::new(CircuitConfig::default());
            let mut asset_backed =
                HypothalamusL4Microcircuit::new_from_asset_bundle(&bundle, &rehydrator)
                    .expect("asset");
            let hardcoded_output = (0..20)
                .map(|_| hardcoded.step(input, 0))
                .last()
                .expect("hardcoded output");
            let asset_output = (0..20)
                .map(|_| asset_backed.step(input, 0))
                .last()
                .expect("asset output");
            let floor = HypothalamusL4Microcircuit::rules_floor(input);
            assert_eq!(floor, *expected_profile, "rules floor mismatch at {idx}");
            assert_eq!(
                hardcoded_output.profile_state, asset_output.profile_state,
                "asset/hardcoded divergence at {idx}"
            );
            assert!(
                profile_rank(hardcoded_output.profile_state) >= profile_rank(*expected_profile),
                "hardcoded below floor at {idx}"
            );
            assert!(
                profile_rank(asset_output.profile_state) >= profile_rank(*expected_profile),
                "asset below floor at {idx}"
            );
        }
    }

    #[test]
    fn reject_wrong_neuron_count() {
        let morph = hypo_morphology_assets(NEURON_COUNT - 1);
        let chan = hypo_channel_params_assets(&morph);
        let syn = hypo_synapse_params_assets();
        let conn = hypo_connectivity_assets();
        let bundle = build_asset_bundle(&morph, &chan, &syn, &conn, None, None);
        let rehydrator = AssetRehydrator::new();

        let err = HypothalamusL4Microcircuit::new_from_asset_bundle(&bundle, &rehydrator)
            .expect_err("expected neuron count mismatch");
        assert!(matches!(err, AssetBuildError::InvalidAssetData { .. }));
    }

    #[test]
    fn manifest_digest_binding_rejects_mismatch() {
        let morph = hypo_morphology_assets(NEURON_COUNT);
        let chan = hypo_channel_params_assets(&morph);
        let syn = hypo_synapse_params_assets();
        let conn = hypo_connectivity_assets();
        let mut mutated_conn = conn.clone();
        if let Some(edge) = mutated_conn.edges.get_mut(0) {
            edge.delay_steps = edge.delay_steps.saturating_add(1);
        }
        let mutated = mutated_conn.to_canonical_bytes();
        let bundle = build_asset_bundle(
            &morph,
            &chan,
            &syn,
            &conn,
            Some(conn.digest()),
            Some(mutated),
        );
        let rehydrator = AssetRehydrator::new();

        let err = HypothalamusL4Microcircuit::new_from_asset_bundle(&bundle, &rehydrator)
            .expect_err("expected digest mismatch");
        match err {
            AssetBuildError::AssetDigestMismatch { .. } => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn boundedness_rejects_large_connectivity() {
        let morph = hypo_morphology_assets(NEURON_COUNT);
        let chan = hypo_channel_params_assets(&morph);
        let syn = hypo_synapse_params_assets();
        let conn = ConnectivityGraph {
            version: 1,
            edges: (0..(MAX_ASSET_EDGES as u32 + 1))
                .map(|idx| ConnEdge {
                    pre: 0,
                    post: 0,
                    syn_type: SynType::Exc,
                    delay_steps: 1,
                    syn_param_id: idx % syn.params.len() as u32,
                })
                .collect(),
        };
        let bundle = build_asset_bundle(&morph, &chan, &syn, &conn, None, None);
        let rehydrator = AssetRehydrator::new();

        let err = HypothalamusL4Microcircuit::new_from_asset_bundle(&bundle, &rehydrator)
            .expect_err("expected bounds error");
        assert!(matches!(err, AssetBuildError::BoundsExceeded { .. }));
    }
}
