#![forbid(unsafe_code)]

#[cfg(feature = "biophys-l4-ca")]
use biophys_channels::CaLike;
use biophys_channels::{Leak, NaK};
use biophys_compartmental_solver::{CompartmentChannels, L4Solver, L4State};
use biophys_core::{CompartmentId, ModChannel, ModLevel, ModulatorField, NeuronId};
use biophys_event_queue_l4::{QueueLimits, RuntimeHealth, SpikeEventQueueL4};
use biophys_homeostasis_l4::{
    homeostasis_tick, scale_g_max_fixed, HomeoMode, HomeostasisConfig, HomeostasisState,
};
use biophys_morphology::{Compartment, CompartmentKind, NeuronMorphology};
use biophys_plasticity_l4::{plasticity_snapshot_digest, LearningMode, StdpConfig, StdpTrace};
#[cfg(feature = "biophys-l4-ca-feedback")]
use biophys_synapses_l4::{apply_inhibitory_boost, INHIBITORY_BOOST_SCALE_Q};
use biophys_synapses_l4::{
    apply_stdp_updates, decay_k, f32_to_fixed_u32, max_synapse_g_fixed, NmdaVDepMode, StpParamsL4,
    StpStateL4, SynKind, SynapseAccumulator, SynapseL4, SynapseState,
};
use biophys_targeting_l4::{select_post_compartment, EdgeKey, TargetRule, TargetingPolicy};
use dbm_core::{
    DbmModule, DwmMode, IntegrityState, LevelClass, ReasonSet, SalienceItem, SalienceSource,
};
use microcircuit_core::{CircuitConfig, MicrocircuitBackend};
use microcircuit_sn_stub::{SnInput, SnOutput};
#[cfg(feature = "biophys-l4-sn-assets")]
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

#[cfg(feature = "biophys-l4-ca-feedback")]
mod ca_feedback;

#[cfg(feature = "biophys-l4-ca-feedback")]
use ca_feedback::CaFeedbackPolicy;

const POOL_COUNT: usize = 4;
const POOL_SIZE: usize = 3;
const EXCITATORY_COUNT: usize = POOL_COUNT * POOL_SIZE;
const INHIBITORY_COUNT: usize = 2;
const NEURON_COUNT: usize = EXCITATORY_COUNT + INHIBITORY_COUNT;

const IDX_EXEC_PLAN: usize = 0;
const IDX_SIMULATE: usize = 1;
const IDX_STABILIZE: usize = 2;
const IDX_REPORT: usize = 3;

#[cfg(feature = "biophys-l4-sn-assets")]
const LABEL_KEY_POOL: &str = "pool";
#[cfg(feature = "biophys-l4-sn-assets")]
const LABEL_KEY_ROLE: &str = "role";
#[cfg(feature = "biophys-l4-sn-assets")]
const ROLE_EXCITATORY: &str = "E";
#[cfg(feature = "biophys-l4-sn-assets")]
const ROLE_INHIBITORY: &str = "I";
#[cfg(feature = "biophys-l4-sn-assets")]
const POOL_LABELS: [&str; 5] = ["EXEC", "SIM", "STAB", "REPORT", "INH"];

const COMPARTMENT_COUNT: usize = 3;
const SUBSTEPS: usize = 10;
const DT_MS: f32 = 0.1;
const CLAMP_MIN: f32 = -120.0;
const CLAMP_MAX: f32 = 60.0;
const THRESHOLD_MV: f32 = -20.0;

const CURRENT_EXEC: f32 = 4.0;
const CURRENT_HIGH: f32 = 12.0;
const CURRENT_REPLAY: f32 = 5.0;

const AMPA_G_MAX: f32 = 4.0;
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
const HYSTERESIS_TICKS: u8 = 3;
const CONFIG_VERSION: u32 = 1;
#[cfg(feature = "biophys-l4-sn-assets")]
const MAX_ASSET_NEURONS: usize = 200;
#[cfg(feature = "biophys-l4-sn-assets")]
const MAX_ASSET_EDGES: usize = 50_000;
#[cfg(feature = "biophys-l4-sn-assets")]
const ASSET_POOL_CONVENTION_V1: u32 = 1;
#[cfg(feature = "biophys-l4-sn-assets")]
const ASSET_POOL_CONVENTION_V2: u32 = 2;
#[cfg(feature = "biophys-l4-sn-assets")]
const LEAK_G_SCALE: f32 = 0.1;

#[derive(Debug, Clone)]
struct L4Neuron {
    solver: L4Solver,
    state: L4State,
    last_soma_v: f32,
}

#[derive(Debug, Clone)]
struct SnL4State {
    tick_count: u64,
    step_count: u64,
    pool_acc: [i32; POOL_COUNT],
    last_pool_spikes: [usize; POOL_COUNT],
    last_spike_count_total: usize,
    last_spike_counts: [u16; NEURON_COUNT],
    winner: usize,
    hysteresis_count: u8,
    pending_winner: Option<usize>,
    #[cfg(feature = "biophys-l4-ca-feedback")]
    ca_hold_counter: u8,
    #[cfg(feature = "biophys-l4-ca")]
    ca_spike_neurons: Vec<u32>,
    last_queue_health: RuntimeHealth,
}

impl Default for SnL4State {
    fn default() -> Self {
        Self {
            tick_count: 0,
            step_count: 0,
            pool_acc: [0; POOL_COUNT],
            last_pool_spikes: [0; POOL_COUNT],
            last_spike_count_total: 0,
            last_spike_counts: [0; NEURON_COUNT],
            winner: IDX_EXEC_PLAN,
            hysteresis_count: 0,
            pending_winner: None,
            #[cfg(feature = "biophys-l4-ca-feedback")]
            ca_hold_counter: 0,
            #[cfg(feature = "biophys-l4-ca")]
            ca_spike_neurons: Vec::new(),
            last_queue_health: RuntimeHealth::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SnL4Microcircuit {
    _config: CircuitConfig,
    neurons: Vec<L4Neuron>,
    synapses: Vec<SynapseL4>,
    syn_states: Vec<SynapseState>,
    syn_g_max_eff: Vec<u32>,
    syn_decay: Vec<u16>,
    syn_stp_params_eff: Vec<StpParamsL4>,
    pre_index: Vec<Vec<usize>>,
    queue: SpikeEventQueueL4,
    state: SnL4State,
    current_modulators: ModulatorField,
    stdp_config: StdpConfig,
    stdp_traces: Vec<StdpTrace>,
    stdp_spike_flags: Vec<bool>,
    learning_enabled: bool,
    in_replay_mode: bool,
    homeostasis_config: HomeostasisConfig,
    homeostasis_state: HomeostasisState,
    #[cfg(feature = "biophys-l4-ca-feedback")]
    ca_feedback_policy: CaFeedbackPolicy,
    #[cfg(feature = "biophys-l4-ca-feedback")]
    inhibitory_boost_q: u16,
    #[cfg(feature = "biophys-l4-sn-assets")]
    asset_manifest_digest: Option<[u8; 32]>,
    #[cfg(feature = "biophys-l4-sn-assets")]
    asset_pool_mapping_version: u32,
    #[cfg(feature = "biophys-l4-sn-assets")]
    asset_pool_mapping_digest: [u8; 32],
}

impl SnL4Microcircuit {
    pub fn new(config: CircuitConfig) -> Self {
        let neurons = (0..NEURON_COUNT)
            .map(|idx| build_neuron(idx as u32))
            .collect::<Vec<_>>();
        let synapses = build_synapses();
        Self::build_from_parts(config, neurons, synapses, None, 0, [0u8; 32])
    }

    #[cfg(feature = "biophys-l4-sn-assets")]
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

        if morph.neurons.len() > MAX_ASSET_NEURONS {
            return Err(AssetBuildError::BoundsExceeded {
                label: "neurons",
                count: morph.neurons.len(),
                max: MAX_ASSET_NEURONS,
            });
        }
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
        #[cfg(not(feature = "biophys-l4-sn-assets"))]
        let _ = asset_manifest_digest;
        #[cfg(not(feature = "biophys-l4-sn-assets"))]
        let _ = asset_pool_mapping_version;
        #[cfg(not(feature = "biophys-l4-sn-assets"))]
        let _ = asset_pool_mapping_digest;
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
            state: SnL4State::default(),
            current_modulators,
            stdp_config,
            stdp_traces,
            stdp_spike_flags,
            learning_enabled: false,
            in_replay_mode: false,
            homeostasis_config,
            homeostasis_state,
            #[cfg(feature = "biophys-l4-ca-feedback")]
            ca_feedback_policy: CaFeedbackPolicy::default().normalized(),
            #[cfg(feature = "biophys-l4-ca-feedback")]
            inhibitory_boost_q: INHIBITORY_BOOST_SCALE_Q,
            #[cfg(feature = "biophys-l4-sn-assets")]
            asset_manifest_digest,
            #[cfg(feature = "biophys-l4-sn-assets")]
            asset_pool_mapping_version,
            #[cfg(feature = "biophys-l4-sn-assets")]
            asset_pool_mapping_digest,
        }
    }

    fn pool_bounds(pool: usize) -> (usize, usize) {
        let start = pool * POOL_SIZE;
        (start, start + POOL_SIZE)
    }

    fn severity_index(mode: DwmMode) -> u8 {
        match mode {
            DwmMode::ExecPlan => 0,
            DwmMode::Simulate => 1,
            DwmMode::Stabilize => 2,
            DwmMode::Report => 3,
        }
    }

    fn rules_floor(input: &SnInput) -> DwmMode {
        if input.isv.integrity == IntegrityState::Fail {
            DwmMode::Report
        } else if input.isv.threat == LevelClass::High {
            DwmMode::Stabilize
        } else if input.isv.policy_pressure == LevelClass::High
            || input.isv.arousal == LevelClass::High
            || input.replay_hint
        {
            DwmMode::Simulate
        } else {
            DwmMode::ExecPlan
        }
    }

    fn mode_from_pool(pool: usize) -> DwmMode {
        match pool {
            IDX_REPORT => DwmMode::Report,
            IDX_STABILIZE => DwmMode::Stabilize,
            IDX_SIMULATE => DwmMode::Simulate,
            IDX_EXEC_PLAN => DwmMode::ExecPlan,
            _ => DwmMode::ExecPlan,
        }
    }

    pub fn last_spike_counts(&self) -> &[u16; NEURON_COUNT] {
        &self.state.last_spike_counts
    }

    pub fn pool_accumulators(&self) -> &[i32; POOL_COUNT] {
        &self.state.pool_acc
    }

    fn pool_for_mode(mode: DwmMode) -> usize {
        match mode {
            DwmMode::ExecPlan => IDX_EXEC_PLAN,
            DwmMode::Simulate => IDX_SIMULATE,
            DwmMode::Stabilize => IDX_STABILIZE,
            DwmMode::Report => IDX_REPORT,
        }
    }

    fn build_inputs(input: &SnInput) -> [f32; NEURON_COUNT] {
        let mut currents = [0.0_f32; NEURON_COUNT];
        let mut pool_drive = [0.0_f32; POOL_COUNT];

        if input.isv.integrity == IntegrityState::Fail {
            pool_drive[IDX_REPORT] = CURRENT_HIGH;
        } else if input.isv.threat == LevelClass::High {
            pool_drive[IDX_STABILIZE] = CURRENT_HIGH;
        } else if input.isv.policy_pressure == LevelClass::High
            || input.isv.arousal == LevelClass::High
        {
            pool_drive[IDX_SIMULATE] = CURRENT_HIGH;
        } else {
            pool_drive[IDX_EXEC_PLAN] = CURRENT_EXEC;
        }

        if input.replay_hint {
            pool_drive[IDX_SIMULATE] += CURRENT_REPLAY;
        }

        for (pool, drive) in pool_drive.iter().enumerate().take(POOL_COUNT) {
            let (start, end) = Self::pool_bounds(pool);
            for current in currents.iter_mut().take(end).skip(start) {
                *current += *drive;
            }
        }

        currents
    }

    fn update_pool_accumulators(&mut self, spike_counts: &[usize; NEURON_COUNT]) {
        let mut pool_counts = [0usize; POOL_COUNT];
        for (idx, count) in spike_counts.iter().enumerate().take(EXCITATORY_COUNT) {
            let pool = idx / POOL_SIZE;
            pool_counts[pool] += *count;
        }

        self.state.last_spike_count_total = spike_counts.iter().sum();
        self.state.last_pool_spikes = pool_counts;
        for (slot, count) in self
            .state
            .last_spike_counts
            .iter_mut()
            .zip(spike_counts.iter())
        {
            *slot = (*count).min(u16::MAX as usize) as u16;
        }

        for (acc, &count) in self.state.pool_acc.iter_mut().zip(pool_counts.iter()) {
            let delta = (count as i32).saturating_mul(ACCUMULATOR_GAIN);
            *acc = (*acc + delta - ACCUMULATOR_DECAY).clamp(0, ACCUMULATOR_MAX);
        }
    }

    fn select_winner(&mut self) -> (usize, bool) {
        let max_value = self.state.pool_acc.iter().copied().max().unwrap_or(0);
        let strict_order = [IDX_REPORT, IDX_STABILIZE, IDX_SIMULATE, IDX_EXEC_PLAN];
        let mut best_pool = IDX_EXEC_PLAN;
        for &pool in &strict_order {
            if self.state.pool_acc[pool] == max_value {
                best_pool = pool;
                break;
            }
        }

        let current = self.state.winner;
        let mut changed = false;
        if best_pool == current {
            self.state.hysteresis_count = 0;
            self.state.pending_winner = None;
        } else if Self::severity_index(Self::mode_from_pool(best_pool))
            > Self::severity_index(Self::mode_from_pool(current))
        {
            self.state.winner = best_pool;
            self.state.hysteresis_count = 0;
            self.state.pending_winner = None;
            changed = true;
        } else {
            if self.state.pending_winner == Some(best_pool) {
                self.state.hysteresis_count = self.state.hysteresis_count.saturating_add(1);
            } else {
                self.state.pending_winner = Some(best_pool);
                self.state.hysteresis_count = 1;
            }

            if self.state.hysteresis_count >= HYSTERESIS_TICKS {
                self.state.winner = best_pool;
                self.state.hysteresis_count = 0;
                self.state.pending_winner = None;
                changed = true;
            }
        }

        (self.state.winner, changed)
    }

    fn build_salience(input: &SnInput, sources: &[SalienceSource]) -> Vec<SalienceItem> {
        let mut items: Vec<SalienceItem> = sources
            .iter()
            .map(|source| {
                SalienceItem::new(
                    *source,
                    LevelClass::High,
                    input.isv.dominant_reason_codes.codes.clone(),
                )
            })
            .collect();

        items.sort_by(|a, b| (a.source as u8).cmp(&(b.source as u8)));
        if items.len() > 8 {
            items.truncate(8);
        }

        items
    }

    fn mod_level_from_class(level: LevelClass) -> ModLevel {
        match level {
            LevelClass::Low => ModLevel::Low,
            LevelClass::Med => ModLevel::Med,
            LevelClass::High => ModLevel::High,
        }
    }

    fn modulators_from_input(input: &SnInput) -> ModulatorField {
        let da_level = if input.reward_block {
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

    fn update_modulators(&mut self, input: &SnInput) {
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
        injected_currents: &[f32; NEURON_COUNT],
        ca_spike_flags: &mut [bool; NEURON_COUNT],
    ) -> Vec<usize> {
        for (idx, state) in self.syn_states.iter_mut().enumerate() {
            let synapse = &self.synapses[idx];
            let decay = self.syn_decay[idx];
            state.decay(synapse.kind, decay, synapse.tau_decay_nmda_steps);
        }

        let events = self.queue.drain_current(self.state.step_count);
        for event in events {
            let synapse = &self.synapses[event.synapse_index];
            let g_max_eff = {
                let base = self.syn_g_max_eff[event.synapse_index];
                #[cfg(feature = "biophys-l4-ca-feedback")]
                {
                    apply_inhibitory_boost(base, synapse.kind, self.inhibitory_boost_q)
                }
                #[cfg(not(feature = "biophys-l4-ca-feedback"))]
                {
                    base
                }
            };
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
            let mut input = [0.0_f32; COMPARTMENT_COUNT];
            input[0] = injected_currents[idx];
            let syn_input = &accumulators[idx];
            let output =
                neuron
                    .solver
                    .step_with_synapses_output(&mut neuron.state, &input, syn_input);
            sanitize_voltages(&mut neuron.state);
            let v = neuron.state.comp_v[0];
            if neuron.last_soma_v < THRESHOLD_MV && v >= THRESHOLD_MV {
                spikes.push(idx);
            }
            if output.ca_spike {
                ca_spike_flags[idx] = true;
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

    #[cfg(feature = "biophys-l4-ca-feedback")]
    pub fn set_ca_feedback_policy(&mut self, policy: CaFeedbackPolicy) {
        self.ca_feedback_policy = policy.normalized();
    }

    #[cfg(test)]
    #[allow(dead_code)]
    fn rebuild_synapse_index(&mut self) {
        self.syn_states = vec![SynapseState::default(); self.synapses.len()];
        self.refresh_syn_g_max_eff();
        self.syn_decay = self
            .synapses
            .iter()
            .map(|synapse| decay_k(DT_MS, synapse.tau_decay_ms))
            .collect::<Vec<_>>();
        self.syn_stp_params_eff = self
            .synapses
            .iter()
            .map(|synapse| synapse.stp_effective_params(self.current_modulators))
            .collect::<Vec<_>>();
        self.pre_index = build_pre_index(NEURON_COUNT, &self.synapses);
        let max_delay = self
            .synapses
            .iter()
            .map(|synapse| synapse.delay_steps)
            .max()
            .unwrap_or(0);
        let limits = QueueLimits::new(
            MAX_EVENTS_PER_STEP.saturating_mul(max_delay as usize + 1),
            MAX_EVENTS_PER_STEP,
        );
        self.queue = SpikeEventQueueL4::new(max_delay, limits);
    }
}

impl MicrocircuitBackend<SnInput, SnOutput> for SnL4Microcircuit {
    fn step(&mut self, input: &SnInput, _now_ms: u64) -> SnOutput {
        self.state.tick_count = self.state.tick_count.saturating_add(1);
        self.update_modulators(input);
        self.set_learning_context(
            input.replay_hint,
            self.current_modulators,
            input.reward_block,
        );
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
        let mut ca_spike_flags = [false; NEURON_COUNT];
        #[cfg(feature = "biophys-l4-ca-feedback")]
        {
            let policy = self.ca_feedback_policy.normalized();
            if policy.enabled && self.state.ca_hold_counter > 0 {
                self.inhibitory_boost_q = policy.gaba_boost_q;
                self.state.ca_hold_counter = self.state.ca_hold_counter.saturating_sub(1);
            } else {
                self.inhibitory_boost_q = INHIBITORY_BOOST_SCALE_Q;
            }
        }
        for _ in 0..SUBSTEPS {
            let spikes = self.substep(&currents, &mut ca_spike_flags);
            for spike in spikes {
                spike_counts[spike] = spike_counts[spike].saturating_add(1);
            }
        }

        #[cfg(feature = "biophys-l4-ca")]
        {
            let mut neurons = ca_spike_flags
                .iter()
                .enumerate()
                .filter_map(|(idx, &flag)| flag.then_some(idx as u32))
                .collect::<Vec<_>>();
            neurons.sort_unstable();
            if neurons.len() > NEURON_COUNT {
                neurons.truncate(NEURON_COUNT);
            }
            self.state.ca_spike_neurons = neurons;
        }

        #[cfg(feature = "biophys-l4-ca-feedback")]
        {
            let policy = self.ca_feedback_policy.normalized();
            if policy.enabled
                && ca_spike_flags
                    .iter()
                    .take(EXCITATORY_COUNT)
                    .any(|&flag| flag)
            {
                self.state.ca_hold_counter = policy.hold_bias_steps;
            }
        }

        self.state.last_queue_health = self.queue.finish_tick();

        self.update_homeostasis(&spike_counts);
        self.update_pool_accumulators(&spike_counts);
        let (winner_pool, mut switched) = self.select_winner();
        let mut dwm = Self::mode_from_pool(winner_pool);

        let floor = Self::rules_floor(input);
        if Self::severity_index(dwm) < Self::severity_index(floor) {
            dwm = floor;
            let floor_pool = Self::pool_for_mode(floor);
            if self.state.winner != floor_pool {
                self.state.winner = floor_pool;
                self.state.hysteresis_count = 0;
                self.state.pending_winner = None;
                switched = true;
            }
        }

        let mut reason_codes = ReasonSet::default();
        if switched {
            reason_codes.insert("RC.GV.DWM.SWITCHED");
        }
        reason_codes.insert(match dwm {
            DwmMode::Report => "RC.GV.DWM.REPORT",
            DwmMode::Stabilize => "RC.GV.DWM.STABILIZE",
            DwmMode::Simulate => "RC.GV.DWM.SIMULATE",
            DwmMode::ExecPlan => "RC.GV.DWM.EXEC_PLAN",
        });
        if self.state.last_queue_health.overflowed {
            reason_codes.insert("RC.GV.BIO.QUEUE_OVERFLOW");
        }
        if self.state.last_queue_health.dropped_events > 0 {
            reason_codes.insert("RC.GV.BIO.EVENTS_DROPPED");
        }
        if self.state.last_queue_health.compacted {
            reason_codes.insert("RC.GV.BIO.QUEUE_COMPACTED");
        }

        let mut sources = Vec::new();
        if input.isv.integrity == IntegrityState::Fail {
            sources.push(SalienceSource::Integrity);
        }
        if input.isv.threat == LevelClass::High {
            sources.push(SalienceSource::Threat);
        }
        if input.isv.policy_pressure == LevelClass::High {
            sources.push(SalienceSource::PolicyPressure);
        }
        if input.replay_hint {
            sources.push(SalienceSource::Replay);
        }

        let salience_items = Self::build_salience(input, &sources);

        SnOutput {
            dwm,
            salience_items,
            reason_codes,
        }
    }

    fn snapshot_digest(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"UCF:BIO:L4:SN:SNAP");
        update_u64(&mut hasher, self.state.tick_count);
        update_u64(&mut hasher, self.state.step_count);
        update_u32(&mut hasher, self.state.winner as u32);
        update_u32(&mut hasher, self.state.hysteresis_count as u32);
        #[cfg(feature = "biophys-l4-ca-feedback")]
        update_u32(&mut hasher, self.state.ca_hold_counter as u32);
        update_u32(
            &mut hasher,
            self.state
                .pending_winner
                .map(|value| value as u32 + 1)
                .unwrap_or(0),
        );
        #[cfg(feature = "biophys-l4-ca")]
        {
            update_u32(&mut hasher, self.state.ca_spike_neurons.len() as u32);
            for neuron_id in &self.state.ca_spike_neurons {
                update_u32(&mut hasher, *neuron_id);
            }
        }
        for value in self.state.pool_acc {
            update_i32(&mut hasher, value);
        }
        for value in self.state.last_pool_spikes {
            update_u32(&mut hasher, value as u32);
        }
        update_u32(&mut hasher, self.state.last_spike_count_total as u32);
        for neuron in &self.neurons {
            update_u64(&mut hasher, neuron.solver.step_count());
            update_u32(&mut hasher, neuron.state.comp_v.len() as u32);
            for idx in 0..neuron.state.comp_v.len() {
                update_i32(&mut hasher, quantize_f32(neuron.state.comp_v[idx], 100.0));
                update_i32(&mut hasher, neuron.state.m_q[idx] as i32);
                update_i32(&mut hasher, neuron.state.h_q[idx] as i32);
                update_i32(&mut hasher, neuron.state.n_q[idx] as i32);
            }
        }
        for state in &self.syn_states {
            update_u32(&mut hasher, state.g_ampa_q);
            update_u32(&mut hasher, state.g_nmda_q);
            update_u32(&mut hasher, state.g_gaba_q);
        }
        update_u32(&mut hasher, mod_level_code(self.current_modulators.na));
        update_u32(&mut hasher, mod_level_code(self.current_modulators.da));
        update_u32(&mut hasher, mod_level_code(self.current_modulators.ht));
        update_u64(&mut hasher, self.queue.dropped_event_count);
        *hasher.finalize().as_bytes()
    }

    fn config_digest(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"UCF:BIO:L4:SN:CFG");
        update_u32(&mut hasher, CONFIG_VERSION);
        update_f32(&mut hasher, DT_MS);
        update_u32(&mut hasher, SUBSTEPS as u32);
        update_f32(&mut hasher, CLAMP_MIN);
        update_f32(&mut hasher, CLAMP_MAX);
        #[cfg(feature = "biophys-l4-sn-assets")]
        {
            let digest = self.asset_manifest_digest.unwrap_or([0u8; 32]);
            hasher.update(&digest);
            update_u32(&mut hasher, self.asset_pool_mapping_version);
            hasher.update(&self.asset_pool_mapping_digest);
        }
        #[cfg(feature = "biophys-l4-ca-feedback")]
        {
            let policy = self.ca_feedback_policy.normalized();
            update_u32(&mut hasher, policy.enabled as u32);
            update_u32(&mut hasher, policy.gaba_boost_q as u32);
            update_u32(&mut hasher, policy.hold_bias_steps as u32);
        }
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
            update_u32(&mut hasher, synapse.g_nmda_base_q);
            update_f32(&mut hasher, synapse.e_rev);
            update_f32(&mut hasher, synapse.tau_rise_ms);
            update_f32(&mut hasher, synapse.tau_decay_ms);
            update_u32(&mut hasher, synapse.tau_decay_nmda_steps as u32);
            update_u32(&mut hasher, synapse.nmda_vdep_mode as u32);
            update_u32(&mut hasher, synapse.delay_steps as u32);
        }
        *hasher.finalize().as_bytes()
    }

    fn plasticity_snapshot_digest_opt(&self) -> Option<[u8; 32]> {
        SnL4Microcircuit::plasticity_snapshot_digest_opt(self)
    }
}

impl DbmModule for SnL4Microcircuit {
    type Input = SnInput;
    type Output = SnOutput;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        self.step(input, 0)
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
        seed_digest: *blake3::hash(b"UCF:L4:TARGETING:SN").as_bytes(),
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
    #[cfg(feature = "biophys-l4-ca")]
    let ca = CaLike {
        g_ca: 0.8,
        e_ca: 120.0,
    };

    let channels = vec![
        CompartmentChannels {
            leak,
            nak: Some(nak),
            #[cfg(feature = "biophys-l4-ca")]
            ca: Some(ca),
        },
        CompartmentChannels {
            leak,
            nak: None,
            #[cfg(feature = "biophys-l4-ca")]
            ca: Some(ca),
        },
        CompartmentChannels {
            leak,
            nak: None,
            #[cfg(feature = "biophys-l4-ca")]
            ca: Some(ca),
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
    struct SynapseSpec {
        pre: u32,
        post: u32,
        kind: SynKind,
        mod_channel: ModChannel,
        g_max_base_q: u32,
        e_rev: f32,
        tau_rise_ms: f32,
        tau_decay_ms: f32,
        delay_steps: u16,
        stp_params: StpParamsL4,
        stp_state: StpStateL4,
        stdp_enabled: bool,
    }

    let mut specs = Vec::new();
    let policy = default_targeting_policy();

    for pool in 0..POOL_COUNT {
        let (start, end) = SnL4Microcircuit::pool_bounds(pool);
        for pre in start..end {
            for post in start..end {
                if pre == post {
                    continue;
                }
                let delay = if (pre + post) % 2 == 0 { 1 } else { 2 };
                let (stp_params, stp_state) = disabled_stp();
                specs.push(SynapseSpec {
                    pre: pre as u32,
                    post: post as u32,
                    kind: SynKind::AMPA,
                    mod_channel: ModChannel::NaDa,
                    g_max_base_q: f32_to_fixed_u32(AMPA_G_MAX),
                    e_rev: AMPA_E_REV,
                    tau_rise_ms: AMPA_TAU_RISE_MS,
                    tau_decay_ms: AMPA_TAU_DECAY_MS,
                    delay_steps: delay,
                    stp_params,
                    stp_state,
                    stdp_enabled: true,
                });
            }
        }
    }

    for pre in 0..EXCITATORY_COUNT {
        for inh in 0..INHIBITORY_COUNT {
            let post = EXCITATORY_COUNT + inh;
            let (stp_params, stp_state) = disabled_stp();
            specs.push(SynapseSpec {
                pre: pre as u32,
                post: post as u32,
                kind: SynKind::AMPA,
                mod_channel: ModChannel::Na,
                g_max_base_q: f32_to_fixed_u32(AMPA_G_MAX),
                e_rev: AMPA_E_REV,
                tau_rise_ms: AMPA_TAU_RISE_MS,
                tau_decay_ms: AMPA_TAU_DECAY_MS,
                delay_steps: 1,
                stp_params,
                stp_state,
                stdp_enabled: true,
            });
        }
    }

    for inh in 0..INHIBITORY_COUNT {
        let pre = EXCITATORY_COUNT + inh;
        for post in 0..EXCITATORY_COUNT {
            let (stp_params, stp_state) = disabled_stp();
            specs.push(SynapseSpec {
                pre: pre as u32,
                post: post as u32,
                kind: SynKind::GABA,
                mod_channel: ModChannel::Ht,
                g_max_base_q: f32_to_fixed_u32(GABA_G_MAX),
                e_rev: GABA_E_REV,
                tau_rise_ms: GABA_TAU_RISE_MS,
                tau_decay_ms: GABA_TAU_DECAY_MS,
                delay_steps: 1,
                stp_params,
                stp_state,
                stdp_enabled: false,
            });
        }
    }

    specs.sort_by_key(|spec| {
        (
            spec.pre,
            spec.post,
            spec.kind as u8,
            spec.delay_steps,
            spec.mod_channel as u8,
        )
    });

    let mut synapses = Vec::with_capacity(specs.len());
    for (idx, spec) in specs.into_iter().enumerate() {
        let edge_key = EdgeKey {
            pre_neuron_id: NeuronId(spec.pre),
            post_neuron_id: NeuronId(spec.post),
            synapse_index: idx as u32,
        };
        let post_compartment =
            select_post_compartment(&build_morphology(spec.post), spec.kind, &policy, edge_key).0;
        synapses.push(SynapseL4 {
            pre_neuron: spec.pre,
            post_neuron: spec.post,
            post_compartment,
            kind: spec.kind,
            mod_channel: spec.mod_channel,
            g_max_base_q: spec.g_max_base_q,
            g_nmda_base_q: 0,
            g_max_min_q: 0,
            g_max_max_q: max_synapse_g_fixed(),
            e_rev: spec.e_rev,
            tau_rise_ms: spec.tau_rise_ms,
            tau_decay_ms: spec.tau_decay_ms,
            tau_decay_nmda_steps: 100,
            nmda_vdep_mode: NmdaVDepMode::PiecewiseLinear,
            delay_steps: spec.delay_steps,
            stp_params: spec.stp_params,
            stp_state: spec.stp_state,
            stdp_enabled: spec.stdp_enabled,
            stdp_trace: StdpTrace::default(),
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

fn update_i32(hasher: &mut blake3::Hasher, value: i32) {
    hasher.update(&value.to_le_bytes());
}

fn update_f32(hasher: &mut blake3::Hasher, value: f32) {
    hasher.update(&value.to_bits().to_le_bytes());
}

#[cfg(feature = "biophys-l4-sn-assets")]
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

#[cfg(feature = "biophys-l4-sn-assets")]
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

#[cfg(feature = "biophys-l4-sn-assets")]
fn kind_name(kind: AssetKind) -> &'static str {
    match kind {
        AssetKind::MorphologySet => "morphology",
        AssetKind::ChannelParamsSet => "channel_params",
        AssetKind::SynapseParamsSet => "synapse_params",
        AssetKind::ConnectivityGraph => "connectivity",
        AssetKind::Unknown => "unknown",
    }
}

#[cfg(feature = "biophys-l4-sn-assets")]
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

#[cfg(feature = "biophys-l4-sn-assets")]
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

#[cfg(feature = "biophys-l4-sn-assets")]
fn compartment_capacitance(kind: biophys_assets::CompartmentKind, length_um: u16) -> f32 {
    match kind {
        biophys_assets::CompartmentKind::Soma => 1.0,
        _ => 1.0 + (length_um as f32) / 100.0,
    }
}

#[cfg(feature = "biophys-l4-sn-assets")]
fn compartment_axial_resistance(length_um: u16) -> f32 {
    100.0 + (length_um as f32) * 5.0
}

#[cfg(feature = "biophys-l4-sn-assets")]
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
        let comp_id = comp.comp_id;
        compute_depth_for_comp(comp_id, &parent_map, &mut depths, &mut visiting)?;
    }
    Ok(depths)
}

#[cfg(feature = "biophys-l4-sn-assets")]
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

#[cfg(feature = "biophys-l4-sn-assets")]
type PoolMap = std::collections::BTreeMap<String, Vec<u32>>;

#[cfg(feature = "biophys-l4-sn-assets")]
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

#[cfg(feature = "biophys-l4-sn-assets")]
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
    pool_map.insert("EXEC".to_string(), (0..POOL_SIZE as u32).collect());
    pool_map.insert(
        "SIM".to_string(),
        ((POOL_SIZE as u32)..(2 * POOL_SIZE) as u32).collect(),
    );
    pool_map.insert(
        "STAB".to_string(),
        ((2 * POOL_SIZE) as u32..(3 * POOL_SIZE) as u32).collect(),
    );
    pool_map.insert(
        "REPORT".to_string(),
        ((3 * POOL_SIZE) as u32..(4 * POOL_SIZE) as u32).collect(),
    );
    pool_map.insert(
        "INH".to_string(),
        (EXCITATORY_COUNT as u32..NEURON_COUNT as u32).collect(),
    );
    Ok(pool_map)
}

#[cfg(feature = "biophys-l4-sn-assets")]
fn ordered_neurons_from_pool_map(pool_map: &PoolMap) -> Vec<u32> {
    let mut ordered = Vec::with_capacity(NEURON_COUNT);
    for pool in POOL_LABELS {
        if let Some(neurons) = pool_map.get(pool) {
            ordered.extend_from_slice(neurons);
        }
    }
    ordered
}

#[cfg(feature = "biophys-l4-sn-assets")]
fn pool_map_digest(pool_map: &PoolMap) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"UCF:BIO:L4:SN:POOLMAP");
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

#[cfg(feature = "biophys-l4-sn-assets")]
impl CircuitBuilderFromAssets for SnL4Microcircuit {
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
                        capacitance: compartment_capacitance(comp.kind, comp.length_um),
                        axial_resistance: compartment_axial_resistance(comp.length_um),
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
                    #[cfg(feature = "biophys-l4-ca")]
                    let ca = CaLike {
                        g_ca: 0.8,
                        e_ca: 120.0,
                    };
                    Ok(CompartmentChannels {
                        leak,
                        nak,
                        #[cfg(feature = "biophys-l4-ca")]
                        ca: Some(ca),
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
                let params = StpParamsL4::disabled();
                let state = StpStateL4::new(params);
                (params, state)
            } else {
                let params = StpParamsL4 {
                    mode: biophys_synapses_l4::StpMode::STP_TM,
                    u_base_q: syn_params.stp_u.min(biophys_core::STP_SCALE),
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

        Ok(SnL4Microcircuit::build_from_parts(
            CircuitConfig::default(),
            neurons,
            synapses,
            None,
            pool_mapping_version,
            pool_mapping_digest,
        ))
    }
}

fn mod_level_code(level: ModLevel) -> u32 {
    match level {
        ModLevel::Low => 0,
        ModLevel::Med => 1,
        ModLevel::High => 2,
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

fn quantize_f32(value: f32, scale: f32) -> i32 {
    (value * scale).round() as i32
}

#[cfg(all(
    test,
    feature = "biophys",
    feature = "biophys-l4",
    feature = "biophys-l4-synapses",
    feature = "biophys-l4-modulation",
    feature = "biophys-l4-sn"
))]
mod tests {
    #[cfg(all(feature = "biophys-l4-ca", feature = "biophys-l4-ca-feedback"))]
    use super::ca_feedback::MAX_HOLD_BIAS_STEPS;
    use super::*;
    #[cfg(all(feature = "biophys-l4-ca", feature = "biophys-l4-ca-feedback"))]
    use biophys_synapses_l4::INHIBITORY_BOOST_Q_MAX;

    fn severity(mode: DwmMode) -> u8 {
        match mode {
            DwmMode::ExecPlan => 0,
            DwmMode::Simulate => 1,
            DwmMode::Stabilize => 2,
            DwmMode::Report => 3,
        }
    }

    #[test]
    fn determinism_sequence_and_digests() {
        let inputs = (0..30)
            .map(|idx| SnInput {
                isv: dbm_core::IsvSnapshot {
                    integrity: if idx % 11 == 0 {
                        IntegrityState::Fail
                    } else {
                        IntegrityState::Ok
                    },
                    threat: if idx % 7 == 0 {
                        LevelClass::High
                    } else {
                        LevelClass::Low
                    },
                    stability: if idx % 6 == 0 {
                        LevelClass::High
                    } else {
                        LevelClass::Low
                    },
                    policy_pressure: if idx % 5 == 0 {
                        LevelClass::High
                    } else {
                        LevelClass::Low
                    },
                    arousal: if idx % 9 == 0 {
                        LevelClass::High
                    } else {
                        LevelClass::Low
                    },
                    progress: if idx % 4 == 0 {
                        LevelClass::High
                    } else {
                        LevelClass::Low
                    },
                    ..dbm_core::IsvSnapshot::default()
                },
                replay_hint: idx % 4 == 0,
                ..Default::default()
            })
            .collect::<Vec<_>>();

        let run_sequence = |inputs: &[SnInput]| -> Vec<(DwmMode, [u8; 32])> {
            let mut circuit = SnL4Microcircuit::new(CircuitConfig::default());
            inputs
                .iter()
                .map(|input| {
                    let output = circuit.step(input, 0);
                    let digest = circuit.snapshot_digest();
                    (output.dwm, digest)
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
        let mut circuit = SnL4Microcircuit::new(CircuitConfig::default());
        let integrity = SnInput {
            isv: dbm_core::IsvSnapshot {
                integrity: IntegrityState::Fail,
                ..dbm_core::IsvSnapshot::default()
            },
            ..Default::default()
        };
        let threat = SnInput {
            isv: dbm_core::IsvSnapshot {
                threat: LevelClass::High,
                ..dbm_core::IsvSnapshot::default()
            },
            ..Default::default()
        };
        let policy = SnInput {
            isv: dbm_core::IsvSnapshot {
                policy_pressure: LevelClass::High,
                ..dbm_core::IsvSnapshot::default()
            },
            ..Default::default()
        };

        let integrity_out = circuit.step(&integrity, 0);
        assert_eq!(integrity_out.dwm, DwmMode::Report);

        let threat_out = circuit.step(&threat, 0);
        assert!(severity(threat_out.dwm) >= severity(DwmMode::Stabilize));

        let policy_out = circuit.step(&policy, 0);
        assert!(severity(policy_out.dwm) >= severity(DwmMode::Simulate));
    }

    #[test]
    fn within_pool_excitation_increases_spikes() {
        let mut circuit = SnL4Microcircuit::new(CircuitConfig::default());
        let mut control = SnL4Microcircuit::new(CircuitConfig::default());
        control.synapses.retain(|synapse| {
            if synapse.kind != SynKind::AMPA {
                return true;
            }
            let pre = synapse.pre_neuron as usize;
            let post = synapse.post_neuron as usize;
            if pre >= EXCITATORY_COUNT || post >= EXCITATORY_COUNT {
                return true;
            }
            let pre_pool = pre / POOL_SIZE;
            let post_pool = post / POOL_SIZE;
            pre_pool != post_pool
        });
        control.rebuild_synapse_index();

        let input = SnInput {
            isv: dbm_core::IsvSnapshot {
                policy_pressure: LevelClass::High,
                ..dbm_core::IsvSnapshot::default()
            },
            ..Default::default()
        };

        let mut spike_within = 0usize;
        let mut spike_control = 0usize;
        for _ in 0..5 {
            circuit.step(&input, 0);
            control.step(&input, 0);
            spike_within += circuit.state.last_pool_spikes[IDX_SIMULATE];
            spike_control += control.state.last_pool_spikes[IDX_SIMULATE];
        }

        assert!(
            spike_within >= spike_control,
            "within-pool excitation should not reduce spikes"
        );
    }

    #[test]
    fn bounded_state_values() {
        let mut circuit = SnL4Microcircuit::new(CircuitConfig::default());
        let input = SnInput {
            isv: dbm_core::IsvSnapshot {
                integrity: IntegrityState::Fail,
                threat: LevelClass::High,
                policy_pressure: LevelClass::High,
                arousal: LevelClass::High,
                ..dbm_core::IsvSnapshot::default()
            },
            replay_hint: true,
            ..Default::default()
        };

        circuit.step(&input, 0);

        assert!(circuit
            .state
            .pool_acc
            .iter()
            .all(|&acc| (0..=ACCUMULATOR_MAX).contains(&acc)));
        assert!(circuit.state.last_spike_count_total <= NEURON_COUNT * SUBSTEPS);
    }

    #[test]
    fn na_high_increases_pool_accumulation() {
        let mut low = SnL4Microcircuit::new(CircuitConfig::default());
        let mut high = SnL4Microcircuit::new(CircuitConfig::default());

        let base_isv = dbm_core::IsvSnapshot {
            policy_pressure: LevelClass::High,
            ..dbm_core::IsvSnapshot::default()
        };

        for _ in 0..6 {
            let low_input = SnInput {
                isv: dbm_core::IsvSnapshot {
                    arousal: LevelClass::Low,
                    ..base_isv.clone()
                },
                ..Default::default()
            };
            let high_input = SnInput {
                isv: dbm_core::IsvSnapshot {
                    arousal: LevelClass::High,
                    ..base_isv.clone()
                },
                ..Default::default()
            };
            low.step(&low_input, 0);
            high.step(&high_input, 0);
        }

        assert!(
            high.state.pool_acc[IDX_SIMULATE] >= low.state.pool_acc[IDX_SIMULATE],
            "NA high should not reduce pool accumulation"
        );
    }

    #[test]
    fn ht_high_reduces_spike_counts() {
        let mut low = SnL4Microcircuit::new(CircuitConfig::default());
        let mut high = SnL4Microcircuit::new(CircuitConfig::default());

        let base_isv = dbm_core::IsvSnapshot {
            policy_pressure: LevelClass::High,
            ..dbm_core::IsvSnapshot::default()
        };

        let mut low_spikes = 0usize;
        let mut high_spikes = 0usize;
        for _ in 0..3 {
            let low_input = SnInput {
                isv: dbm_core::IsvSnapshot {
                    stability: LevelClass::Low,
                    ..base_isv.clone()
                },
                ..Default::default()
            };
            let high_input = SnInput {
                isv: dbm_core::IsvSnapshot {
                    stability: LevelClass::High,
                    ..base_isv.clone()
                },
                ..Default::default()
            };
            low.step(&low_input, 0);
            high.step(&high_input, 0);
            low_spikes += low.state.last_pool_spikes.iter().sum::<usize>();
            high_spikes += high.state.last_pool_spikes.iter().sum::<usize>();
        }

        assert!(
            high_spikes <= low_spikes,
            "HT high should not increase overall spike counts"
        );
    }

    #[test]
    fn reward_block_forces_da_low() {
        let mut circuit = SnL4Microcircuit::new(CircuitConfig::default());
        let input = SnInput {
            isv: dbm_core::IsvSnapshot {
                arousal: LevelClass::Med,
                progress: LevelClass::High,
                ..dbm_core::IsvSnapshot::default()
            },
            reward_block: true,
            ..Default::default()
        };

        circuit.step(&input, 0);

        assert_eq!(circuit.current_modulators.da, ModLevel::Low);

        let (idx, synapse) = circuit
            .synapses
            .iter()
            .enumerate()
            .find(|(_, synapse)| synapse.mod_channel == ModChannel::NaDa)
            .expect("expected a NA/DA synapse");
        let expected = (synapse.g_max_base_q as u64 * 90) / 100;
        assert_eq!(circuit.syn_g_max_eff[idx] as u64, expected);
    }

    #[cfg(all(feature = "biophys-l4-ca", feature = "biophys-l4-ca-feedback"))]
    #[test]
    fn ca_feedback_determinism() {
        let input = SnInput {
            isv: dbm_core::IsvSnapshot {
                policy_pressure: LevelClass::High,
                ..dbm_core::IsvSnapshot::default()
            },
            replay_hint: true,
            ..Default::default()
        };
        let inputs = vec![input; 8];

        let run = |inputs: &[SnInput]| -> Vec<(DwmMode, Vec<u32>)> {
            let mut circuit = SnL4Microcircuit::new(CircuitConfig::default());
            inputs
                .iter()
                .map(|input| {
                    let output = circuit.step(input, 0);
                    (output.dwm, circuit.state.ca_spike_neurons.clone())
                })
                .collect()
        };

        let outputs_a = run(&inputs);
        let outputs_b = run(&inputs);

        assert_eq!(outputs_a, outputs_b);
        assert!(
            outputs_a.iter().any(|(_, spikes)| !spikes.is_empty()),
            "expected at least one ca-spike event"
        );
    }

    #[cfg(all(feature = "biophys-l4-ca", feature = "biophys-l4-ca-feedback"))]
    #[test]
    fn ca_feedback_reduces_spikes_and_switches() {
        let mut enabled = SnL4Microcircuit::new(CircuitConfig::default());
        let mut disabled = SnL4Microcircuit::new(CircuitConfig::default());
        disabled.set_ca_feedback_policy(CaFeedbackPolicy::disabled());

        let inputs = (0..10)
            .map(|idx| SnInput {
                isv: dbm_core::IsvSnapshot {
                    policy_pressure: LevelClass::High,
                    arousal: if idx % 2 == 0 {
                        LevelClass::High
                    } else {
                        LevelClass::Low
                    },
                    ..dbm_core::IsvSnapshot::default()
                },
                replay_hint: idx % 3 == 0,
                ..Default::default()
            })
            .collect::<Vec<_>>();

        let mut enabled_spikes = 0usize;
        let mut disabled_spikes = 0usize;
        let mut enabled_switches = 0usize;
        let mut disabled_switches = 0usize;
        let mut enabled_any_ca = false;
        let mut disabled_any_ca = false;

        let mut prev_enabled = enabled.state.winner;
        let mut prev_disabled = disabled.state.winner;

        for input in &inputs {
            enabled.step(input, 0);
            disabled.step(input, 0);

            enabled_spikes += enabled.state.last_spike_count_total;
            disabled_spikes += disabled.state.last_spike_count_total;

            if enabled.state.winner != prev_enabled {
                enabled_switches = enabled_switches.saturating_add(1);
                prev_enabled = enabled.state.winner;
            }
            if disabled.state.winner != prev_disabled {
                disabled_switches = disabled_switches.saturating_add(1);
                prev_disabled = disabled.state.winner;
            }

            enabled_any_ca |= !enabled.state.ca_spike_neurons.is_empty();
            disabled_any_ca |= !disabled.state.ca_spike_neurons.is_empty();
        }

        assert!(enabled_any_ca && disabled_any_ca);
        assert!(
            enabled_spikes <= disabled_spikes,
            "ca feedback should not increase excitatory spikes"
        );
        assert!(
            enabled_switches <= disabled_switches,
            "ca feedback should not increase winner switches"
        );
    }

    #[cfg(all(feature = "biophys-l4-ca", feature = "biophys-l4-ca-feedback"))]
    #[test]
    fn ca_feedback_is_bounded() {
        let mut circuit = SnL4Microcircuit::new(CircuitConfig::default());
        circuit.set_ca_feedback_policy(CaFeedbackPolicy {
            enabled: true,
            gaba_boost_q: 4000,
            hold_bias_steps: 100,
        });

        let input = SnInput {
            isv: dbm_core::IsvSnapshot {
                policy_pressure: LevelClass::High,
                ..dbm_core::IsvSnapshot::default()
            },
            replay_hint: true,
            ..Default::default()
        };

        circuit.step(&input, 0);

        assert!(circuit.ca_feedback_policy.gaba_boost_q <= INHIBITORY_BOOST_Q_MAX);
        assert!(circuit.ca_feedback_policy.hold_bias_steps <= MAX_HOLD_BIAS_STEPS);
        assert!(circuit.state.ca_hold_counter <= MAX_HOLD_BIAS_STEPS);
    }
}

#[cfg(all(
    test,
    feature = "biophys",
    feature = "biophys-l4",
    feature = "biophys-l4-synapses",
    feature = "biophys-l4-modulation",
    feature = "biophys-l4-sn",
    feature = "biophys-l4-sn-assets"
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

    fn sn_l4_morphology_assets() -> MorphologySet {
        let neurons = (0..NEURON_COUNT as u32)
            .map(|neuron_id| {
                let pool = match neuron_id as usize {
                    0..=2 => "EXEC",
                    3..=5 => "SIM",
                    6..=8 => "STAB",
                    9..=11 => "REPORT",
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

    fn sn_l4_channel_params_assets(morph: &MorphologySet) -> ChannelParamsSet {
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

    fn sn_l4_synapse_params_assets() -> SynapseParamsSet {
        SynapseParamsSet {
            version: 1,
            params: vec![
                SynapseParams {
                    syn_type: SynType::Exc,
                    weight_base: AMPA_G_MAX as i32,
                    stp_u: 0,
                    tau_rec: 1,
                    tau_fac: 1,
                    mod_channel: ModChannel::A,
                },
                SynapseParams {
                    syn_type: SynType::Exc,
                    weight_base: AMPA_G_MAX as i32,
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

    fn sn_l4_connectivity_assets() -> ConnectivityGraph {
        let mut edges = Vec::new();
        for pool in 0..POOL_COUNT {
            let (start, end) = SnL4Microcircuit::pool_bounds(pool);
            for pre in start..end {
                for post in start..end {
                    if pre == post {
                        continue;
                    }
                    let delay = if (pre + post) % 2 == 0 { 1 } else { 2 };
                    edges.push(ConnEdge {
                        pre: pre as u32,
                        post: post as u32,
                        syn_type: SynType::Exc,
                        delay_steps: delay,
                        syn_param_id: 0,
                    });
                }
            }
        }

        for pre in 0..EXCITATORY_COUNT {
            for inh in 0..INHIBITORY_COUNT {
                let post = EXCITATORY_COUNT + inh;
                edges.push(ConnEdge {
                    pre: pre as u32,
                    post: post as u32,
                    syn_type: SynType::Exc,
                    delay_steps: 1,
                    syn_param_id: 1,
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
                    syn_param_id: 2,
                });
            }
        }

        ConnectivityGraph { version: 1, edges }
    }

    fn sn_l4_empty_connectivity_assets() -> ConnectivityGraph {
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
    ) -> AssetBundle {
        let created_at_ms = 10;
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
                    conn.digest(),
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
            max_chunks_total: 4096,
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
        chunks.extend(
            chunk_asset(
                AssetKind::ConnectivityGraph,
                conn.version,
                conn.digest(),
                &conn.to_canonical_bytes(),
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

    fn demo_inputs() -> Vec<SnInput> {
        (0..20)
            .map(|idx| SnInput {
                isv: dbm_core::IsvSnapshot {
                    integrity: if idx % 7 == 0 {
                        IntegrityState::Fail
                    } else {
                        IntegrityState::Ok
                    },
                    threat: if idx % 5 == 0 {
                        LevelClass::High
                    } else {
                        LevelClass::Low
                    },
                    policy_pressure: if idx % 4 == 0 {
                        LevelClass::High
                    } else {
                        LevelClass::Low
                    },
                    stability: if idx % 3 == 0 {
                        LevelClass::High
                    } else {
                        LevelClass::Low
                    },
                    ..dbm_core::IsvSnapshot::default()
                },
                replay_hint: idx % 6 == 0,
                ..Default::default()
            })
            .collect()
    }

    #[test]
    fn asset_backed_matches_hardcoded_outputs() {
        let morph = sn_l4_morphology_assets();
        let chan = sn_l4_channel_params_assets(&morph);
        let syn = sn_l4_synapse_params_assets();
        let conn = sn_l4_connectivity_assets();
        let bundle = build_asset_bundle(&morph, &chan, &syn, &conn);
        let rehydrator = AssetRehydrator::new();

        let inputs = demo_inputs();
        let mut hardcoded = SnL4Microcircuit::new(CircuitConfig::default());
        let mut asset_backed =
            SnL4Microcircuit::new_from_asset_bundle(&bundle, &rehydrator).expect("asset build");

        let outputs_hardcoded = inputs
            .iter()
            .map(|input| hardcoded.step(input, 0).dwm)
            .collect::<Vec<_>>();
        let outputs_asset = inputs
            .iter()
            .map(|input| asset_backed.step(input, 0).dwm)
            .collect::<Vec<_>>();

        assert_eq!(outputs_asset, outputs_hardcoded);
    }

    #[test]
    fn asset_build_is_deterministic() {
        let morph = sn_l4_morphology_assets();
        let chan = sn_l4_channel_params_assets(&morph);
        let syn = sn_l4_synapse_params_assets();
        let conn = sn_l4_connectivity_assets();
        let bundle = build_asset_bundle(&morph, &chan, &syn, &conn);
        let rehydrator = AssetRehydrator::new();
        let inputs = demo_inputs();

        let mut a =
            SnL4Microcircuit::new_from_asset_bundle(&bundle, &rehydrator).expect("asset build");
        let mut b =
            SnL4Microcircuit::new_from_asset_bundle(&bundle, &rehydrator).expect("asset build");

        let outputs_a = inputs
            .iter()
            .map(|input| (a.step(input, 0).dwm, a.config_digest()))
            .collect::<Vec<_>>();
        let outputs_b = inputs
            .iter()
            .map(|input| (b.step(input, 0).dwm, b.config_digest()))
            .collect::<Vec<_>>();

        assert_eq!(outputs_a, outputs_b);
    }

    #[test]
    fn asset_pool_mapping_uses_labels() {
        let mut morph = sn_l4_morphology_assets();
        for neuron in &mut morph.neurons {
            for label in &mut neuron.labels {
                if label.k == "pool" {
                    label.v = match label.v.as_str() {
                        "EXEC" => "SIM".to_string(),
                        "SIM" => "EXEC".to_string(),
                        other => other.to_string(),
                    };
                }
            }
        }
        let chan = sn_l4_channel_params_assets(&morph);
        let syn = sn_l4_synapse_params_assets();
        let conn = sn_l4_empty_connectivity_assets();

        let circuit = <SnL4Microcircuit as CircuitBuilderFromAssets>::build_from_assets(
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
        let mut morph = sn_l4_morphology_assets();
        morph.version = 1;
        for neuron in &mut morph.neurons {
            neuron.labels.clear();
        }
        let chan = sn_l4_channel_params_assets(&morph);
        let syn = sn_l4_synapse_params_assets();
        let conn = sn_l4_connectivity_assets();
        let bundle = build_asset_bundle(&morph, &chan, &syn, &conn);
        let rehydrator = AssetRehydrator::new();

        let circuit = SnL4Microcircuit::new_from_asset_bundle(&bundle, &rehydrator).expect("asset");
        let pool_map = pool_map_from_ranges(&morph).expect("pool map");
        assert_eq!(circuit.asset_pool_mapping_version, ASSET_POOL_CONVENTION_V1);
        assert_eq!(
            circuit.asset_pool_mapping_digest,
            pool_map_digest(&pool_map)
        );
    }

    #[test]
    fn asset_pool_mapping_rejects_missing_pool_label() {
        let mut morph = sn_l4_morphology_assets();
        if let Some(neuron) = morph.neurons.get_mut(0) {
            neuron.labels.retain(|label| label.k != "pool");
        }
        let chan = sn_l4_channel_params_assets(&morph);
        let syn = sn_l4_synapse_params_assets();
        let conn = sn_l4_empty_connectivity_assets();

        let err = <SnL4Microcircuit as CircuitBuilderFromAssets>::build_from_assets(
            &morph, &chan, &syn, &conn,
        )
        .expect_err("missing pool label should fail");
        assert!(matches!(err, AssetBuildError::InvalidAssetData { .. }));
    }

    #[test]
    fn asset_pool_mapping_rejects_duplicate_pool_labels() {
        let mut morph = sn_l4_morphology_assets();
        if let Some(neuron) = morph.neurons.get_mut(0) {
            neuron.labels.push(LabelKV {
                k: "pool".to_string(),
                v: "EXEC".to_string(),
            });
        }
        let chan = sn_l4_channel_params_assets(&morph);
        let syn = sn_l4_synapse_params_assets();
        let conn = sn_l4_empty_connectivity_assets();

        let err = <SnL4Microcircuit as CircuitBuilderFromAssets>::build_from_assets(
            &morph, &chan, &syn, &conn,
        )
        .expect_err("duplicate pool label should fail");
        assert!(matches!(err, AssetBuildError::InvalidAssetData { .. }));
    }

    #[test]
    fn manifest_digest_binding_rejects_mismatch() {
        let morph = sn_l4_morphology_assets();
        let chan = sn_l4_channel_params_assets(&morph);
        let syn = sn_l4_synapse_params_assets();
        let mut mutated_syn = syn.clone();
        mutated_syn.params[0].weight_base = (AMPA_G_MAX as i32) + 1;
        let conn = sn_l4_connectivity_assets();
        let created_at_ms = 10;
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
                    conn.digest(),
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
            max_chunks_total: 4096,
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
                &mutated_syn.to_canonical_bytes(),
                &chunker,
                created_at_ms,
            )
            .expect("syn chunks"),
        );
        chunks.extend(
            chunk_asset(
                AssetKind::ConnectivityGraph,
                conn.version,
                conn.digest(),
                &conn.to_canonical_bytes(),
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
        let bundle = build_asset_bundle_with_policy(
            manifest,
            chunks,
            created_at_ms,
            BundleIdPolicy::ManifestDigestPrefix { prefix_len: 8 },
        );

        let rehydrator = AssetRehydrator::new();
        let err = SnL4Microcircuit::new_from_asset_bundle(&bundle, &rehydrator)
            .expect_err("expected digest mismatch");
        match err {
            AssetBuildError::AssetDigestMismatch { .. } => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn boundedness_rejects_large_assets() {
        let morph = MorphologySet {
            version: 1,
            neurons: (0..(MAX_ASSET_NEURONS as u32 + 1))
                .map(|neuron_id| MorphNeuron {
                    neuron_id,
                    compartments: vec![AssetCompartment {
                        comp_id: 0,
                        parent: None,
                        kind: AssetCompartmentKind::Soma,
                        length_um: 10,
                        diameter_um: 8,
                    }],
                    labels: Vec::new(),
                })
                .collect(),
        };
        let chan = sn_l4_channel_params_assets(&morph);
        let syn = sn_l4_synapse_params_assets();
        let conn = sn_l4_connectivity_assets();
        let bundle = build_asset_bundle(&morph, &chan, &syn, &conn);
        let rehydrator = AssetRehydrator::new();

        let err = SnL4Microcircuit::new_from_asset_bundle(&bundle, &rehydrator)
            .expect_err("expected bounds error");
        assert!(matches!(err, AssetBuildError::BoundsExceeded { .. }));

        let morph = sn_l4_morphology_assets();
        let chan = sn_l4_channel_params_assets(&morph);
        let syn = sn_l4_synapse_params_assets();
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
        let bundle = build_asset_bundle(&morph, &chan, &syn, &conn);
        let err = SnL4Microcircuit::new_from_asset_bundle(&bundle, &rehydrator)
            .expect_err("expected bounds error");
        assert!(matches!(err, AssetBuildError::BoundsExceeded { .. }));
    }
}
