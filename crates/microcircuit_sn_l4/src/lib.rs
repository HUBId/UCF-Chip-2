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
use dbm_core::{
    DbmModule, DwmMode, IntegrityState, LevelClass, ReasonSet, SalienceItem, SalienceSource,
};
use microcircuit_core::{CircuitConfig, MicrocircuitBackend};
use microcircuit_sn_stub::{SnInput, SnOutput};

const POOL_COUNT: usize = 4;
const POOL_SIZE: usize = 3;
const EXCITATORY_COUNT: usize = POOL_COUNT * POOL_SIZE;
const INHIBITORY_COUNT: usize = 2;
const NEURON_COUNT: usize = EXCITATORY_COUNT + INHIBITORY_COUNT;

const IDX_EXEC_PLAN: usize = 0;
const IDX_SIMULATE: usize = 1;
const IDX_STABILIZE: usize = 2;
const IDX_REPORT: usize = 3;

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
    winner: usize,
    hysteresis_count: u8,
    pending_winner: Option<usize>,
}

impl Default for SnL4State {
    fn default() -> Self {
        Self {
            tick_count: 0,
            step_count: 0,
            pool_acc: [0; POOL_COUNT],
            last_pool_spikes: [0; POOL_COUNT],
            last_spike_count_total: 0,
            winner: IDX_EXEC_PLAN,
            hysteresis_count: 0,
            pending_winner: None,
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
}

impl SnL4Microcircuit {
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
            state: SnL4State::default(),
            current_modulators,
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
            self.syn_g_max_eff[idx] = synapse.effective_g_max_fixed(mods);
            self.syn_stp_params_eff[idx] = synapse.stp_effective_params(mods);
        }
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

        self.state.step_count = self.state.step_count.saturating_add(1);
        spikes
    }

    #[cfg(test)]
    #[allow(dead_code)]
    fn rebuild_synapse_index(&mut self) {
        self.syn_states = vec![SynapseState::default(); self.synapses.len()];
        self.syn_g_max_eff = self
            .synapses
            .iter()
            .map(|synapse| synapse.effective_g_max_fixed(self.current_modulators))
            .collect();
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
        self.queue = SpikeEventQueueL4::new(max_delay, MAX_EVENTS_PER_STEP);
    }
}

impl MicrocircuitBackend<SnInput, SnOutput> for SnL4Microcircuit {
    fn step(&mut self, input: &SnInput, _now_ms: u64) -> SnOutput {
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
        update_u32(
            &mut hasher,
            self.state
                .pending_winner
                .map(|value| value as u32 + 1)
                .unwrap_or(0),
        );
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
        hasher.update(b"UCF:BIO:L4:SN:CFG");
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

impl DbmModule for SnL4Microcircuit {
    type Input = SnInput;
    type Output = SnOutput;

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
        let (start, end) = SnL4Microcircuit::pool_bounds(pool);
        for pre in start..end {
            for post in start..end {
                if pre == post {
                    continue;
                }
                let delay = if (pre + post) % 2 == 0 { 1 } else { 2 };
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
                    delay_steps: delay,
                    stp_params,
                    stp_state,
                });
            }
        }
    }

    for pre in 0..EXCITATORY_COUNT {
        for inh in 0..INHIBITORY_COUNT {
            let post = EXCITATORY_COUNT + inh;
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
                e_rev: GABA_E_REV,
                tau_rise_ms: GABA_TAU_RISE_MS,
                tau_decay_ms: GABA_TAU_DECAY_MS,
                delay_steps: 1,
                stp_params,
                stp_state,
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
    use super::*;

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
            .all(|&acc| acc >= 0 && acc <= ACCUMULATOR_MAX));
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
}
