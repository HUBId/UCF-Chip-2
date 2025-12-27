#![forbid(unsafe_code)]

use biophys_channels::{Leak, NaK};
use biophys_compartmental_solver::{CompartmentChannels, L4Solver, L4State};
use biophys_core::{CompartmentId, ModChannel, ModLevel, ModulatorField, NeuronId};
use biophys_event_queue_l4::{QueueLimits, RuntimeHealth, SpikeEventQueueL4};
use biophys_homeostasis_l4::{
    homeostasis_tick, scale_g_max_fixed, HomeoMode, HomeostasisConfig, HomeostasisState,
};
use biophys_morphology::{Compartment, CompartmentKind, NeuronMorphology};
use biophys_plasticity_l4::{plasticity_snapshot_digest, LearningMode, StdpConfig, StdpTrace};
use biophys_synapses_l4::{
    apply_stdp_updates, decay_k, f32_to_fixed_u32, max_synapse_g_fixed, NmdaVDepMode, StpParamsL4,
    StpStateL4, SynKind, SynapseAccumulator, SynapseL4, SynapseState,
};
use biophys_targeting_l4::{select_post_compartment, EdgeKey, TargetRule, TargetingPolicy};
use dbm_core::{IntegrityState, LevelClass, ReasonSet, ThreatVector};
use microcircuit_amygdala_stub::{AmyInput, AmyOutput};
use microcircuit_core::{CircuitConfig, MicrocircuitBackend};
#[cfg(feature = "biophys-l4-amygdala-assets")]
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

#[cfg(feature = "biophys-l4-amygdala-assets")]
const MAX_ASSET_NEURONS: usize = 100;
#[cfg(feature = "biophys-l4-amygdala-assets")]
const MAX_ASSET_EDGES: usize = 25_000;
#[cfg(feature = "biophys-l4-amygdala-assets")]
const ASSET_POOL_CONVENTION_V1: u32 = 1;
#[cfg(feature = "biophys-l4-amygdala-assets")]
const ASSET_POOL_CONVENTION_V2: u32 = 2;
#[cfg(feature = "biophys-l4-amygdala-assets")]
const LEAK_G_SCALE: f32 = 0.1;

#[cfg(feature = "biophys-l4-amygdala-assets")]
const LABEL_KEY_POOL: &str = "pool";
#[cfg(feature = "biophys-l4-amygdala-assets")]
const LABEL_KEY_ROLE: &str = "role";
#[cfg(feature = "biophys-l4-amygdala-assets")]
const ROLE_EXCITATORY: &str = "E";
#[cfg(feature = "biophys-l4-amygdala-assets")]
const ROLE_INHIBITORY: &str = "I";
#[cfg(feature = "biophys-l4-amygdala-assets")]
const POOL_LABELS: [&str; 5] = ["INTEGRITY", "EXFIL", "PROBING", "TOOLSE", "INH"];

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
    last_queue_health: RuntimeHealth,
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
            last_queue_health: RuntimeHealth::default(),
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
    stdp_config: StdpConfig,
    stdp_traces: Vec<StdpTrace>,
    stdp_spike_flags: Vec<bool>,
    learning_enabled: bool,
    in_replay_mode: bool,
    homeostasis_config: HomeostasisConfig,
    homeostasis_state: HomeostasisState,
    #[cfg(feature = "biophys-l4-amygdala-assets")]
    asset_manifest_digest: Option<[u8; 32]>,
    #[cfg(feature = "biophys-l4-amygdala-assets")]
    asset_pool_mapping_version: u32,
    #[cfg(feature = "biophys-l4-amygdala-assets")]
    asset_pool_mapping_digest: [u8; 32],
}

impl AmygdalaL4Microcircuit {
    pub fn new(config: CircuitConfig) -> Self {
        let neurons = (0..NEURON_COUNT)
            .map(|idx| build_neuron(idx as u32))
            .collect::<Vec<_>>();
        let synapses = build_synapses();
        Self::build_from_parts(config, neurons, synapses, None, 0, [0u8; 32])
    }

    #[cfg(feature = "biophys-l4-amygdala-assets")]
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
        #[cfg(not(feature = "biophys-l4-amygdala-assets"))]
        let _ = asset_manifest_digest;
        #[cfg(not(feature = "biophys-l4-amygdala-assets"))]
        let _ = asset_pool_mapping_version;
        #[cfg(not(feature = "biophys-l4-amygdala-assets"))]
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
            state: AmyL4State::default(),
            current_modulators,
            stdp_config,
            stdp_traces,
            stdp_spike_flags,
            learning_enabled: false,
            in_replay_mode: false,
            homeostasis_config,
            homeostasis_state,
            #[cfg(feature = "biophys-l4-amygdala-assets")]
            asset_manifest_digest,
            #[cfg(feature = "biophys-l4-amygdala-assets")]
            asset_pool_mapping_version,
            #[cfg(feature = "biophys-l4-amygdala-assets")]
            asset_pool_mapping_digest,
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
            self.syn_g_max_eff[idx] = Self::scaled_g_max_fixed(
                synapse,
                self.current_modulators,
                self.homeostasis_state.scale_q,
            );
            self.syn_stp_params_eff[idx] = synapse.stp_effective_params(self.current_modulators);
        }
    }

    #[cfg(not(feature = "biophys-l4-modulation"))]
    fn update_modulators(&mut self, _input: &AmyInput) {
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
            let input = injected_currents[idx];
            let syn_input = &accumulators[idx];
            neuron
                .solver
                .step_with_synapses(&mut neuron.state, &input, syn_input);
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

impl MicrocircuitBackend<AmyInput, AmyOutput> for AmygdalaL4Microcircuit {
    fn step(&mut self, input: &AmyInput, _now_ms: u64) -> AmyOutput {
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
        let currents = Self::build_inputs(input);

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
        if self.state.last_queue_health.overflowed {
            reason_codes.insert("RC.GV.BIO.QUEUE_OVERFLOW");
        }
        if self.state.last_queue_health.dropped_events > 0 {
            reason_codes.insert("RC.GV.BIO.EVENTS_DROPPED");
        }
        if self.state.last_queue_health.compacted {
            reason_codes.insert("RC.GV.BIO.QUEUE_COMPACTED");
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
        *hasher.finalize().as_bytes()
    }

    fn config_digest(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"UCF:BIO:L4:AMY:CFG");
        update_f32(&mut hasher, DT_MS);
        update_u32(&mut hasher, SUBSTEPS as u32);
        update_f32(&mut hasher, CLAMP_MIN);
        update_f32(&mut hasher, CLAMP_MAX);
        #[cfg(feature = "biophys-l4-amygdala-assets")]
        {
            let digest = self.asset_manifest_digest.unwrap_or([0u8; 32]);
            hasher.update(&digest);
            update_u32(&mut hasher, self.asset_pool_mapping_version);
            hasher.update(&self.asset_pool_mapping_digest);
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
        AmygdalaL4Microcircuit::plasticity_snapshot_digest_opt(self)
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
        seed_digest: *blake3::hash(b"UCF:L4:TARGETING:AMYGDALA").as_bytes(),
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
        let (start, end) = AmygdalaL4Microcircuit::pool_bounds(pool);
        for pre in start..end {
            for (post, morphology) in morphologies.iter().enumerate().take(end).skip(start) {
                if pre == post {
                    continue;
                }
                let (stp_params, stp_state) = disabled_stp();
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
                    mod_channel: ModChannel::NaDa,
                    g_max_base_q: f32_to_fixed_u32(AMPA_G_MAX),
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

    let (integrity_start, integrity_end) = AmygdalaL4Microcircuit::pool_bounds(IDX_INTEGRITY);
    let (exfil_start, exfil_end) = AmygdalaL4Microcircuit::pool_bounds(IDX_EXFIL);
    for pre in integrity_start..integrity_end {
        for (post, morphology) in morphologies
            .iter()
            .enumerate()
            .take(exfil_end)
            .skip(exfil_start)
        {
            let (stp_params, stp_state) = disabled_stp();
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
                mod_channel: ModChannel::NaDa,
                g_max_base_q: f32_to_fixed_u32(AMPA_G_MAX_WEAK),
                g_nmda_base_q: 0,
                g_max_min_q: 0,
                g_max_max_q: max_synapse_g_fixed(),
                e_rev: AMPA_E_REV,
                tau_rise_ms: AMPA_TAU_RISE_MS,
                tau_decay_ms: AMPA_TAU_DECAY_MS,
                tau_decay_nmda_steps: 100,
                nmda_vdep_mode: NmdaVDepMode::PiecewiseLinear,
                delay_steps: 2,
                stp_params,
                stp_state,
                stdp_enabled: true,
                stdp_trace: StdpTrace::default(),
            });
        }
    }

    for pre in 0..EXCITATORY_COUNT {
        let post = EXCITATORY_COUNT;
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
            g_max_base_q: f32_to_fixed_u32(AMPA_G_MAX),
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

    let pre = EXCITATORY_COUNT;
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

    synapses
}

#[cfg(feature = "biophys-l4-amygdala-assets")]
type PoolMap = std::collections::BTreeMap<String, Vec<u32>>;

#[cfg(feature = "biophys-l4-amygdala-assets")]
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

#[cfg(feature = "biophys-l4-amygdala-assets")]
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
    pool_map.insert("INTEGRITY".to_string(), (0..POOL_SIZE as u32).collect());
    pool_map.insert(
        "EXFIL".to_string(),
        ((POOL_SIZE as u32)..(2 * POOL_SIZE) as u32).collect(),
    );
    pool_map.insert(
        "PROBING".to_string(),
        ((2 * POOL_SIZE) as u32..(3 * POOL_SIZE) as u32).collect(),
    );
    pool_map.insert(
        "TOOLSE".to_string(),
        ((3 * POOL_SIZE) as u32..(4 * POOL_SIZE) as u32).collect(),
    );
    pool_map.insert(
        "INH".to_string(),
        (EXCITATORY_COUNT as u32..NEURON_COUNT as u32).collect(),
    );
    Ok(pool_map)
}

#[cfg(feature = "biophys-l4-amygdala-assets")]
fn ordered_neurons_from_pool_map(pool_map: &PoolMap) -> Vec<u32> {
    let mut ordered = Vec::with_capacity(NEURON_COUNT);
    for pool in POOL_LABELS {
        if let Some(neurons) = pool_map.get(pool) {
            ordered.extend_from_slice(neurons);
        }
    }
    ordered
}

#[cfg(feature = "biophys-l4-amygdala-assets")]
fn pool_map_digest(pool_map: &PoolMap) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"UCF:BIO:L4:AMY:POOLMAP");
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

#[cfg(feature = "biophys-l4-amygdala-assets")]
fn asset_compartment_capacitance(kind: biophys_assets::CompartmentKind) -> f32 {
    match kind {
        biophys_assets::CompartmentKind::Soma => 1.0,
        biophys_assets::CompartmentKind::Dendrite => 1.2,
        biophys_assets::CompartmentKind::Axon => 1.0,
    }
}

#[cfg(feature = "biophys-l4-amygdala-assets")]
fn asset_compartment_axial_resistance(kind: biophys_assets::CompartmentKind) -> f32 {
    match kind {
        biophys_assets::CompartmentKind::Soma => 150.0,
        biophys_assets::CompartmentKind::Dendrite => 200.0,
        biophys_assets::CompartmentKind::Axon => 150.0,
    }
}

#[cfg(feature = "biophys-l4-amygdala-assets")]
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

#[cfg(feature = "biophys-l4-amygdala-assets")]
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

#[cfg(feature = "biophys-l4-amygdala-assets")]
impl CircuitBuilderFromAssets for AmygdalaL4Microcircuit {
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

        Ok(AmygdalaL4Microcircuit::build_from_parts(
            CircuitConfig::default(),
            neurons,
            synapses,
            None,
            pool_mapping_version,
            pool_mapping_digest,
        ))
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

#[cfg(feature = "biophys-l4-amygdala-assets")]
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

#[cfg(feature = "biophys-l4-amygdala-assets")]
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

#[cfg(feature = "biophys-l4-amygdala-assets")]
fn kind_name(kind: AssetKind) -> &'static str {
    match kind {
        AssetKind::MorphologySet => "morphology",
        AssetKind::ChannelParamsSet => "channel_params",
        AssetKind::SynapseParamsSet => "synapse_params",
        AssetKind::ConnectivityGraph => "connectivity",
        AssetKind::Unknown => "unknown",
    }
}

#[cfg(feature = "biophys-l4-amygdala-assets")]
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

#[cfg(feature = "biophys-l4-amygdala-assets")]
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
            trace_fail_present: false,
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
            .all(|&acc| (0..=ACCUMULATOR_MAX).contains(&acc)));
        assert!(circuit
            .state
            .latch_steps
            .iter()
            .all(|&latch| latch <= LATCH_STEPS_MAX));
        assert!(circuit.state.last_spike_count_total <= NEURON_COUNT * SUBSTEPS);
    }
}

#[cfg(all(
    test,
    feature = "biophys",
    feature = "biophys-l4",
    feature = "biophys-l4-synapses",
    feature = "biophys-l4-amygdala",
    feature = "biophys-l4-amygdala-assets"
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

    fn amygdala_morphology_assets() -> MorphologySet {
        let neurons = (0..NEURON_COUNT as u32)
            .map(|neuron_id| {
                let pool = match neuron_id as usize {
                    0..=1 => "INTEGRITY",
                    2..=3 => "EXFIL",
                    4..=5 => "PROBING",
                    6..=7 => "TOOLSE",
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

    fn amygdala_channel_params_assets(morph: &MorphologySet) -> ChannelParamsSet {
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

    fn amygdala_synapse_params_assets() -> SynapseParamsSet {
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
                    weight_base: AMPA_G_MAX_WEAK as i32,
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

    fn amygdala_connectivity_assets() -> ConnectivityGraph {
        let mut edges = Vec::new();
        for pool in 0..POOL_COUNT {
            let (start, end) = AmygdalaL4Microcircuit::pool_bounds(pool);
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

        let (integrity_start, integrity_end) = AmygdalaL4Microcircuit::pool_bounds(IDX_INTEGRITY);
        let (exfil_start, exfil_end) = AmygdalaL4Microcircuit::pool_bounds(IDX_EXFIL);
        for pre in integrity_start..integrity_end {
            for post in exfil_start..exfil_end {
                edges.push(ConnEdge {
                    pre: pre as u32,
                    post: post as u32,
                    syn_type: SynType::Exc,
                    delay_steps: 2,
                    syn_param_id: 1,
                });
            }
        }

        for pre in 0..EXCITATORY_COUNT {
            let post = EXCITATORY_COUNT;
            edges.push(ConnEdge {
                pre: pre as u32,
                post: post as u32,
                syn_type: SynType::Exc,
                delay_steps: 1,
                syn_param_id: 2,
            });
        }

        let pre = EXCITATORY_COUNT;
        for post in 0..EXCITATORY_COUNT {
            edges.push(ConnEdge {
                pre: pre as u32,
                post: post as u32,
                syn_type: SynType::Inh,
                delay_steps: 1,
                syn_param_id: 3,
            });
        }

        ConnectivityGraph { version: 1, edges }
    }

    fn amygdala_empty_connectivity_assets() -> ConnectivityGraph {
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
        syn_digest_override: Option<[u8; 32]>,
        syn_bytes_override: Option<Vec<u8>>,
    ) -> AssetBundle {
        let created_at_ms = 10;
        let syn_digest = syn_digest_override.unwrap_or_else(|| syn.digest());
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
                    syn_digest,
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
        let syn_bytes = syn_bytes_override.unwrap_or_else(|| syn.to_canonical_bytes());
        chunks.extend(
            chunk_asset(
                AssetKind::SynapseParamsSet,
                syn.version,
                syn_digest,
                &syn_bytes,
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

    fn demo_inputs() -> Vec<AmyInput> {
        vec![
            AmyInput {
                dlp_secret_present: true,
                ..AmyInput::default()
            },
            AmyInput {
                integrity: IntegrityState::Fail,
                ..AmyInput::default()
            },
            AmyInput {
                policy_pressure: LevelClass::High,
                ..AmyInput::default()
            },
            AmyInput {
                tool_anomaly_present: true,
                ..AmyInput::default()
            },
        ]
    }

    #[test]
    fn asset_build_is_deterministic() {
        let morph = amygdala_morphology_assets();
        let chan = amygdala_channel_params_assets(&morph);
        let syn = amygdala_synapse_params_assets();
        let conn = amygdala_connectivity_assets();
        let bundle = build_asset_bundle(&morph, &chan, &syn, &conn, None, None);
        let rehydrator = AssetRehydrator::new();
        let inputs = demo_inputs();

        let mut a =
            AmygdalaL4Microcircuit::new_from_asset_bundle(&bundle, &rehydrator).expect("asset");
        let mut b =
            AmygdalaL4Microcircuit::new_from_asset_bundle(&bundle, &rehydrator).expect("asset");

        let outputs_a = inputs
            .iter()
            .map(|input| {
                let output = a.step(input, 0);
                (output.threat, output.vectors.clone(), a.config_digest())
            })
            .collect::<Vec<_>>();
        let outputs_b = inputs
            .iter()
            .map(|input| {
                let output = b.step(input, 0);
                (output.threat, output.vectors.clone(), b.config_digest())
            })
            .collect::<Vec<_>>();

        assert_eq!(outputs_a, outputs_b);
    }

    #[test]
    fn asset_pool_mapping_uses_labels() {
        let mut morph = amygdala_morphology_assets();
        for neuron in &mut morph.neurons {
            for label in &mut neuron.labels {
                if label.k == "pool" {
                    label.v = match label.v.as_str() {
                        "INTEGRITY" => "EXFIL".to_string(),
                        "EXFIL" => "INTEGRITY".to_string(),
                        other => other.to_string(),
                    };
                }
            }
        }
        let chan = amygdala_channel_params_assets(&morph);
        let syn = amygdala_synapse_params_assets();
        let conn = amygdala_empty_connectivity_assets();

        let circuit = <AmygdalaL4Microcircuit as CircuitBuilderFromAssets>::build_from_assets(
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
        let mut morph = amygdala_morphology_assets();
        morph.version = 1;
        for neuron in &mut morph.neurons {
            neuron.labels.clear();
        }
        let chan = amygdala_channel_params_assets(&morph);
        let syn = amygdala_synapse_params_assets();
        let conn = amygdala_connectivity_assets();
        let bundle = build_asset_bundle(&morph, &chan, &syn, &conn, None, None);
        let rehydrator = AssetRehydrator::new();

        let circuit =
            AmygdalaL4Microcircuit::new_from_asset_bundle(&bundle, &rehydrator).expect("asset");
        let pool_map = pool_map_from_ranges(&morph).expect("pool map");
        assert_eq!(circuit.asset_pool_mapping_version, ASSET_POOL_CONVENTION_V1);
        assert_eq!(
            circuit.asset_pool_mapping_digest,
            pool_map_digest(&pool_map)
        );
    }

    #[test]
    fn asset_pool_mapping_rejects_missing_pool_label() {
        let mut morph = amygdala_morphology_assets();
        if let Some(neuron) = morph.neurons.get_mut(0) {
            neuron.labels.retain(|label| label.k != "pool");
        }
        let chan = amygdala_channel_params_assets(&morph);
        let syn = amygdala_synapse_params_assets();
        let conn = amygdala_empty_connectivity_assets();

        let err = <AmygdalaL4Microcircuit as CircuitBuilderFromAssets>::build_from_assets(
            &morph, &chan, &syn, &conn,
        )
        .expect_err("missing pool label should fail");
        assert!(matches!(err, AssetBuildError::InvalidAssetData { .. }));
    }

    #[test]
    fn asset_pool_mapping_rejects_duplicate_pool_labels() {
        let mut morph = amygdala_morphology_assets();
        if let Some(neuron) = morph.neurons.get_mut(0) {
            neuron.labels.push(LabelKV {
                k: "pool".to_string(),
                v: "INTEGRITY".to_string(),
            });
        }
        let chan = amygdala_channel_params_assets(&morph);
        let syn = amygdala_synapse_params_assets();
        let conn = amygdala_empty_connectivity_assets();

        let err = <AmygdalaL4Microcircuit as CircuitBuilderFromAssets>::build_from_assets(
            &morph, &chan, &syn, &conn,
        )
        .expect_err("duplicate pool label should fail");
        assert!(matches!(err, AssetBuildError::InvalidAssetData { .. }));
    }

    #[test]
    fn asset_backed_matches_hardcoded_outputs() {
        let morph = amygdala_morphology_assets();
        let chan = amygdala_channel_params_assets(&morph);
        let syn = amygdala_synapse_params_assets();
        let conn = amygdala_connectivity_assets();
        let bundle = build_asset_bundle(&morph, &chan, &syn, &conn, None, None);
        let rehydrator = AssetRehydrator::new();
        let inputs = demo_inputs();

        let mut hardcoded = AmygdalaL4Microcircuit::new(CircuitConfig::default());
        let mut asset_backed =
            AmygdalaL4Microcircuit::new_from_asset_bundle(&bundle, &rehydrator).expect("asset");

        let outputs_hardcoded = inputs
            .iter()
            .map(|input| hardcoded.step(input, 0))
            .collect::<Vec<_>>();
        let outputs_asset = inputs
            .iter()
            .map(|input| asset_backed.step(input, 0))
            .collect::<Vec<_>>();

        assert_eq!(outputs_asset, outputs_hardcoded);
        assert!(outputs_asset[0].vectors.contains(&ThreatVector::Exfil));
        assert!(outputs_asset[1]
            .vectors
            .contains(&ThreatVector::IntegrityCompromise));
    }

    #[test]
    fn manifest_digest_binding_rejects_mismatch() {
        let morph = amygdala_morphology_assets();
        let chan = amygdala_channel_params_assets(&morph);
        let syn = amygdala_synapse_params_assets();
        let conn = amygdala_connectivity_assets();
        let mut mutated_syn = syn.clone();
        mutated_syn.params[0].weight_base = (AMPA_G_MAX as i32) + 1;
        let bundle = build_asset_bundle(
            &morph,
            &chan,
            &syn,
            &conn,
            Some(syn.digest()),
            Some(mutated_syn.to_canonical_bytes()),
        );
        let rehydrator = AssetRehydrator::new();

        let err = AmygdalaL4Microcircuit::new_from_asset_bundle(&bundle, &rehydrator)
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
        let chan = amygdala_channel_params_assets(&morph);
        let syn = amygdala_synapse_params_assets();
        let conn = amygdala_connectivity_assets();
        let bundle = build_asset_bundle(&morph, &chan, &syn, &conn, None, None);
        let rehydrator = AssetRehydrator::new();

        let err = AmygdalaL4Microcircuit::new_from_asset_bundle(&bundle, &rehydrator)
            .expect_err("expected bounds error");
        assert!(matches!(err, AssetBuildError::BoundsExceeded { .. }));

        let morph = amygdala_morphology_assets();
        let chan = amygdala_channel_params_assets(&morph);
        let syn = amygdala_synapse_params_assets();
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
        let err = AmygdalaL4Microcircuit::new_from_asset_bundle(&bundle, &rehydrator)
            .expect_err("expected bounds error");
        assert!(matches!(err, AssetBuildError::BoundsExceeded { .. }));
    }
}
