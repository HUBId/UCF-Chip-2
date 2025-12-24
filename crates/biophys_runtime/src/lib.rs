#![forbid(unsafe_code)]

use biophys_core::{
    clamp_i32, clamp_usize, level_mul, LifParams, LifState, ModChannel, ModLevel, ModulatorField,
    NeuronId, PopCode, StpParams, SynapseEdge, STP_SCALE,
};
use biophys_solver::{LifSolver, StepSolver};

const MIN_EFFECTIVE_WEIGHT: i32 = -1000;
const MAX_EFFECTIVE_WEIGHT: i32 = 1000;
const MAX_TAU_REC_STEPS: u16 = 1000;

pub trait BiophysCircuit<I: ?Sized, O> {
    fn step(&mut self, input: &I) -> O;
    fn config_digest(&self) -> [u8; 32];
}

#[derive(Debug, Clone)]
pub struct BiophysRuntime {
    pub params: Vec<LifParams>,
    pub states: Vec<LifState>,
    pub dt_ms: u16,
    pub step_count: u64,
    pub max_spikes_per_step: usize,
    pub current_modulators: ModulatorField,
    pub edges: Vec<SynapseEdge>,
    pub stp_params: Vec<StpParams>,
    pub pre_index: Vec<Vec<usize>>,
    pub event_queue: Vec<Vec<SpikeEvent>>,
    pub max_events_per_step: usize,
    pub dropped_event_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpikeEvent {
    pub deliver_at_step: u64,
    pub post: NeuronId,
    pub current: i32,
}

impl BiophysRuntime {
    pub fn new(
        params: Vec<LifParams>,
        states: Vec<LifState>,
        dt_ms: u16,
        max_spikes_per_step: usize,
    ) -> Self {
        Self::new_with_synapses(
            params,
            states,
            dt_ms,
            max_spikes_per_step,
            Vec::new(),
            Vec::new(),
            50_000,
        )
    }

    pub fn new_with_synapses(
        params: Vec<LifParams>,
        states: Vec<LifState>,
        dt_ms: u16,
        max_spikes_per_step: usize,
        edges: Vec<SynapseEdge>,
        stp_params: Vec<StpParams>,
        max_events_per_step: usize,
    ) -> Self {
        assert!(dt_ms > 0, "dt_ms must be non-zero");
        assert_eq!(params.len(), states.len(), "params/state mismatch");
        assert!(
            edges.is_empty() || edges.len() == stp_params.len(),
            "edges/stp_params mismatch"
        );
        let mut pre_index = vec![Vec::new(); states.len()];
        let mut max_delay = 0u16;
        for (idx, edge) in edges.iter().enumerate() {
            let pre_idx = edge.pre.0 as usize;
            let post_idx = edge.post.0 as usize;
            assert!(pre_idx < states.len(), "edge pre out of range");
            assert!(post_idx < states.len(), "edge post out of range");
            pre_index[pre_idx].push(idx);
            max_delay = max_delay.max(edge.delay_steps);
        }
        let queue_len = max_delay as usize + 1;
        Self {
            params,
            states,
            dt_ms,
            step_count: 0,
            max_spikes_per_step,
            current_modulators: ModulatorField::default(),
            edges,
            stp_params,
            pre_index,
            event_queue: vec![Vec::new(); queue_len.max(1)],
            max_events_per_step,
            dropped_event_count: 0,
        }
    }

    pub fn set_modulators(&mut self, mods: ModulatorField) {
        self.current_modulators = mods;
    }

    pub fn step(&mut self, inputs: &[i32]) -> PopCode {
        assert_eq!(inputs.len(), self.states.len(), "input length mismatch");
        let mods = self.current_modulators;
        if !self.edges.is_empty() {
            for (edge, params) in self.edges.iter_mut().zip(self.stp_params.iter().copied()) {
                edge.weight_effective =
                    Self::compute_effective_weight(mods, edge.weight_base, edge.mod_channel);
                let effective_params = Self::modulated_stp_params(mods, edge, params);
                edge.stp.update_between_spikes(effective_params);
            }
        }

        let mut syn_inputs = vec![0i32; self.states.len()];
        if !self.event_queue.is_empty() {
            let bucket = (self.step_count as usize) % self.event_queue.len();
            let mut events = std::mem::take(&mut self.event_queue[bucket]);
            events.sort_by_key(|event| event.post.0);
            for event in events {
                let post_idx = event.post.0 as usize;
                if post_idx < syn_inputs.len() {
                    syn_inputs[post_idx] = syn_inputs[post_idx].saturating_add(event.current);
                }
            }
        }
        let mut spikes = Vec::new();
        for (idx, (state, params)) in self.states.iter_mut().zip(self.params.iter()).enumerate() {
            let mut solver = LifSolver::new(*params, self.dt_ms);
            let total_input = inputs[idx].saturating_add(syn_inputs[idx]);
            if solver.step(state, &total_input) {
                spikes.push(NeuronId(idx as u32));
            }
        }
        spikes.sort_by_key(|id| id.0);
        let max_spikes = clamp_usize(spikes.len(), self.max_spikes_per_step);
        spikes.truncate(max_spikes);

        if !self.edges.is_empty() {
            for spike in &spikes {
                let pre_idx = spike.0 as usize;
                if pre_idx >= self.pre_index.len() {
                    continue;
                }
                for &edge_idx in &self.pre_index[pre_idx] {
                    let edge = &mut self.edges[edge_idx];
                    let params = self.stp_params[edge_idx];
                    let effective_params = Self::modulated_stp_params(mods, edge, params);
                    let released = edge.stp.on_spike(effective_params);
                    let current =
                        (edge.weight_effective as i64 * released as i64 / 1000) as i32;
                    let deliver_at_step = self.step_count.saturating_add(edge.delay_steps as u64);
                    let bucket = (deliver_at_step as usize) % self.event_queue.len();
                    if self.event_queue[bucket].len() >= self.max_events_per_step {
                        self.dropped_event_count = self.dropped_event_count.saturating_add(1);
                        continue;
                    }
                    self.event_queue[bucket].push(SpikeEvent {
                        deliver_at_step,
                        post: edge.post,
                        current,
                    });
                }
            }
        }
        self.step_count = self.step_count.saturating_add(1);
        PopCode { spikes }
    }

    pub fn config_digest(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"UCF:BIO:CFG");
        update_u16(&mut hasher, self.dt_ms);
        update_u32(&mut hasher, self.params.len() as u32);
        for params in &self.params {
            update_u16(&mut hasher, params.tau_ms);
            update_i32(&mut hasher, params.v_rest);
            update_i32(&mut hasher, params.v_reset);
            update_i32(&mut hasher, params.v_threshold);
        }
        update_u32(&mut hasher, self.edges.len() as u32);
        for (edge, params) in self.edges.iter().zip(self.stp_params.iter()) {
            update_u32(&mut hasher, edge.pre.0);
            update_u32(&mut hasher, edge.post.0);
            update_i32(&mut hasher, edge.weight_base);
            update_u16(&mut hasher, edge.delay_steps);
            update_u8(&mut hasher, mod_channel_code(edge.mod_channel));
            update_u16(&mut hasher, params.u);
            update_u16(&mut hasher, params.tau_rec_steps);
            update_u16(&mut hasher, params.tau_fac_steps);
            update_u8(&mut hasher, mod_channel_code(params.mod_channel.unwrap_or_default()));
        }
        update_u32(&mut hasher, self.max_events_per_step as u32);
        *hasher.finalize().as_bytes()
    }

    pub fn snapshot_digest(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"UCF:BIO:SNAP");
        update_u64(&mut hasher, self.step_count);
        update_u32(&mut hasher, self.states.len() as u32);
        for state in &self.states {
            update_i32(&mut hasher, state.v);
            update_u16(&mut hasher, state.refractory_steps);
        }
        update_u32(&mut hasher, self.edges.len() as u32);
        for edge in &self.edges {
            update_u16(&mut hasher, edge.stp.x);
            update_u16(&mut hasher, edge.stp.u);
        }
        update_u8(&mut hasher, mod_level_code(self.current_modulators.na));
        update_u8(&mut hasher, mod_level_code(self.current_modulators.da));
        update_u8(&mut hasher, mod_level_code(self.current_modulators.ht));
        update_u64(&mut hasher, self.dropped_event_count);
        *hasher.finalize().as_bytes()
    }

    fn compute_effective_weight(
        mods: ModulatorField,
        weight_base: i32,
        channel: ModChannel,
    ) -> i32 {
        let mult = match channel {
            ModChannel::None => 100,
            ModChannel::Na => level_mul(mods.na),
            ModChannel::Da => level_mul(mods.da),
            ModChannel::Ht => level_mul(mods.ht),
            ModChannel::NaDa => {
                let na = level_mul(mods.na) as i64;
                let da = level_mul(mods.da) as i64;
                ((na * da) / 100) as i32
            }
        };
        let scaled = (weight_base as i64 * mult as i64) / 100;
        clamp_i32(scaled as i32, MIN_EFFECTIVE_WEIGHT, MAX_EFFECTIVE_WEIGHT)
    }

    fn modulated_stp_params(
        mods: ModulatorField,
        edge: &SynapseEdge,
        base: StpParams,
    ) -> StpParams {
        let channel = base.mod_channel.unwrap_or(edge.mod_channel);
        match channel {
            ModChannel::Da | ModChannel::NaDa => {
                if mods.da == ModLevel::High {
                    let u = (base.u as i32 + 50).clamp(0, STP_SCALE as i32) as u16;
                    StpParams { u, ..base }
                } else {
                    base
                }
            }
            ModChannel::Ht => {
                if mods.ht == ModLevel::High {
                    let tau_rec_steps =
                        (base.tau_rec_steps as u32 + 5).min(MAX_TAU_REC_STEPS as u32) as u16;
                    StpParams {
                        tau_rec_steps,
                        ..base
                    }
                } else {
                    base
                }
            }
            _ => base,
        }
    }
}

impl BiophysCircuit<[i32], PopCode> for BiophysRuntime {
    fn step(&mut self, input: &[i32]) -> PopCode {
        BiophysRuntime::step(self, input)
    }

    fn config_digest(&self) -> [u8; 32] {
        BiophysRuntime::config_digest(self)
    }
}

fn update_u16(hasher: &mut blake3::Hasher, value: u16) {
    hasher.update(&value.to_le_bytes());
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

fn update_u8(hasher: &mut blake3::Hasher, value: u8) {
    hasher.update(&[value]);
}

fn mod_channel_code(channel: ModChannel) -> u8 {
    match channel {
        ModChannel::None => 0,
        ModChannel::Na => 1,
        ModChannel::Da => 2,
        ModChannel::Ht => 3,
        ModChannel::NaDa => 4,
    }
}

fn mod_level_code(level: ModLevel) -> u8 {
    match level {
        ModLevel::Low => 0,
        ModLevel::Med => 1,
        ModLevel::High => 2,
    }
}
