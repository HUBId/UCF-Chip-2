#![forbid(unsafe_code)]

use biophys_core::{
    clamp_i32, clamp_usize, level_mul, LifParams, LifState, ModChannel, ModLevel, ModulatorField,
    NeuronId, PartitionPlan, PopCode, StpParams, SynapseEdge, STP_SCALE,
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
    counters: RuntimeCounters,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpikeEvent {
    pub deliver_at_step: u64,
    pub post: NeuronId,
    pub current: i32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RuntimeCounters {
    pub steps_executed: u64,
    pub spikes_total: u64,
    pub events_pushed: u64,
    pub events_delivered: u64,
    pub events_dropped: u64,
    pub max_bucket_depth_seen: u32,
    pub compactions_run: u32,
    pub asset_bytes_decoded: u64,
}

pub struct SynapseConfig {
    pub edges: Vec<SynapseEdge>,
    pub stp_params: Vec<StpParams>,
    pub max_events_per_step: usize,
}

fn compute_effective_weight(mods: ModulatorField, weight_base: i32, channel: ModChannel) -> i32 {
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

fn modulated_stp_params(mods: ModulatorField, edge: &SynapseEdge, base: StpParams) -> StpParams {
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

fn deliver_events(
    step_count: u64,
    event_queue: &mut [Vec<SpikeEvent>],
    syn_inputs: &mut [i32],
    counters: &mut RuntimeCounters,
) {
    if event_queue.is_empty() {
        return;
    }
    let bucket = (step_count as usize) % event_queue.len();
    let mut events = std::mem::take(&mut event_queue[bucket]);
    counters.events_delivered = counters
        .events_delivered
        .saturating_add(events.len() as u64);
    events.sort_by_key(|event| event.post.0);
    for event in events {
        let post_idx = event.post.0 as usize;
        if post_idx < syn_inputs.len() {
            syn_inputs[post_idx] = syn_inputs[post_idx].saturating_add(event.current);
        }
    }
}

struct ScheduleContext<'a> {
    pre_index: &'a [Vec<usize>],
    edges: &'a mut [SynapseEdge],
    stp_params: &'a [StpParams],
    mods: ModulatorField,
    event_queue: &'a mut [Vec<SpikeEvent>],
    max_events_per_step: usize,
    dropped_event_count: &'a mut u64,
    counters: &'a mut RuntimeCounters,
}

fn schedule_spikes(spikes: &[NeuronId], step_count: u64, context: &mut ScheduleContext<'_>) {
    if context.event_queue.is_empty() || context.edges.is_empty() {
        return;
    }
    for spike in spikes {
        let pre_idx = spike.0 as usize;
        if pre_idx >= context.pre_index.len() {
            continue;
        }
        for &edge_idx in &context.pre_index[pre_idx] {
            let edge = &mut context.edges[edge_idx];
            let params = context.stp_params[edge_idx];
            let effective_params = modulated_stp_params(context.mods, edge, params);
            let released = edge.stp.on_spike(effective_params);
            let current = (edge.weight_effective as i64 * released as i64 / 1000) as i32;
            let deliver_at_step = step_count.saturating_add(edge.delay_steps as u64);
            let bucket = (deliver_at_step as usize) % context.event_queue.len();
            if context.event_queue[bucket].len() >= context.max_events_per_step {
                *context.dropped_event_count = context.dropped_event_count.saturating_add(1);
                context.counters.events_dropped = context.counters.events_dropped.saturating_add(1);
                continue;
            }
            context.event_queue[bucket].push(SpikeEvent {
                deliver_at_step,
                post: edge.post,
                current,
            });
            context.counters.events_pushed = context.counters.events_pushed.saturating_add(1);
            let depth = context.event_queue[bucket].len() as u32;
            if depth > context.counters.max_bucket_depth_seen {
                context.counters.max_bucket_depth_seen = depth;
            }
        }
    }
}

struct PartitionStepContext<'a> {
    dt_ms: u16,
    params: &'a [LifParams],
    states: &'a mut [LifState],
    inputs: &'a [i32],
    syn_inputs: &'a [i32],
    spike_buffer: &'a mut Vec<NeuronId>,
    neuron_start: u32,
    max_spikes: usize,
}

fn step_partition(context: &mut PartitionStepContext<'_>) {
    context.spike_buffer.clear();
    for idx in 0..context.states.len() {
        let mut solver = LifSolver::new(context.params[idx], context.dt_ms);
        let total_input = context.inputs[idx].saturating_add(context.syn_inputs[idx]);
        if solver.step(&mut context.states[idx], &total_input) {
            context
                .spike_buffer
                .push(NeuronId(context.neuron_start + idx as u32));
        }
    }
    let max_spikes = clamp_usize(context.spike_buffer.len(), context.max_spikes);
    context.spike_buffer.truncate(max_spikes);
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
            counters: RuntimeCounters::default(),
        }
    }

    pub fn set_modulators(&mut self, mods: ModulatorField) {
        self.current_modulators = mods;
    }

    pub fn step(&mut self, inputs: &[i32]) -> PopCode {
        assert_eq!(inputs.len(), self.states.len(), "input length mismatch");
        self.counters.steps_executed = self.counters.steps_executed.saturating_add(1);
        let mods = self.current_modulators;
        if !self.edges.is_empty() {
            for (edge, params) in self.edges.iter_mut().zip(self.stp_params.iter().copied()) {
                edge.weight_effective =
                    compute_effective_weight(mods, edge.weight_base, edge.mod_channel);
                let effective_params = modulated_stp_params(mods, edge, params);
                edge.stp.update_between_spikes(effective_params);
            }
        }

        let mut syn_inputs = vec![0i32; self.states.len()];
        deliver_events(
            self.step_count,
            &mut self.event_queue,
            &mut syn_inputs,
            &mut self.counters,
        );
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
        self.counters.spikes_total = self
            .counters
            .spikes_total
            .saturating_add(spikes.len() as u64);

        let mut schedule_context = ScheduleContext {
            pre_index: &self.pre_index,
            edges: &mut self.edges,
            stp_params: &self.stp_params,
            mods,
            event_queue: &mut self.event_queue,
            max_events_per_step: self.max_events_per_step,
            dropped_event_count: &mut self.dropped_event_count,
            counters: &mut self.counters,
        };
        schedule_spikes(&spikes, self.step_count, &mut schedule_context);
        self.step_count = self.step_count.saturating_add(1);
        PopCode { spikes }
    }

    pub fn counters_snapshot(&self) -> RuntimeCounters {
        self.counters
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
            update_u8(
                &mut hasher,
                mod_channel_code(params.mod_channel.unwrap_or_default()),
            );
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
}

#[derive(Debug, Clone)]
pub struct PartitionedRuntime {
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
    counters: RuntimeCounters,
    pub partition_plan: PartitionPlan,
    partition_spike_buffers: Vec<Vec<NeuronId>>,
    partition_input_buffers: Vec<Vec<i32>>,
    partition_merge_order: Vec<usize>,
}

impl PartitionedRuntime {
    pub fn new(
        params: Vec<LifParams>,
        states: Vec<LifState>,
        dt_ms: u16,
        max_spikes_per_step: usize,
        partition_plan: PartitionPlan,
    ) -> Self {
        Self::new_with_synapses(
            params,
            states,
            dt_ms,
            max_spikes_per_step,
            partition_plan,
            SynapseConfig {
                edges: Vec::new(),
                stp_params: Vec::new(),
                max_events_per_step: 50_000,
            },
        )
    }

    pub fn new_with_synapses(
        params: Vec<LifParams>,
        states: Vec<LifState>,
        dt_ms: u16,
        max_spikes_per_step: usize,
        partition_plan: PartitionPlan,
        synapses: SynapseConfig,
    ) -> Self {
        assert!(dt_ms > 0, "dt_ms must be non-zero");
        assert_eq!(params.len(), states.len(), "params/state mismatch");
        assert!(
            synapses.edges.is_empty() || synapses.edges.len() == synapses.stp_params.len(),
            "edges/stp_params mismatch"
        );
        partition_plan
            .validate(states.len() as u32)
            .expect("invalid partition plan");
        let mut pre_index = vec![Vec::new(); states.len()];
        let mut max_delay = 0u16;
        for (idx, edge) in synapses.edges.iter().enumerate() {
            let pre_idx = edge.pre.0 as usize;
            let post_idx = edge.post.0 as usize;
            assert!(pre_idx < states.len(), "edge pre out of range");
            assert!(post_idx < states.len(), "edge post out of range");
            pre_index[pre_idx].push(idx);
            max_delay = max_delay.max(edge.delay_steps);
        }
        let queue_len = max_delay as usize + 1;
        let partition_spike_buffers = vec![Vec::new(); partition_plan.partitions.len()];
        let partition_input_buffers = partition_plan
            .partitions
            .iter()
            .map(|partition| vec![0i32; partition.len() as usize])
            .collect();
        let mut partition_merge_order: Vec<usize> = (0..partition_plan.partitions.len()).collect();
        partition_merge_order.sort_by_key(|idx| partition_plan.partitions[*idx].id);
        Self {
            params,
            states,
            dt_ms,
            step_count: 0,
            max_spikes_per_step,
            current_modulators: ModulatorField::default(),
            edges: synapses.edges,
            stp_params: synapses.stp_params,
            pre_index,
            event_queue: vec![Vec::new(); queue_len.max(1)],
            max_events_per_step: synapses.max_events_per_step,
            dropped_event_count: 0,
            counters: RuntimeCounters::default(),
            partition_plan,
            partition_spike_buffers,
            partition_input_buffers,
            partition_merge_order,
        }
    }

    pub fn set_modulators(&mut self, mods: ModulatorField) {
        self.current_modulators = mods;
    }

    pub fn step(&mut self, inputs: &[i32]) -> PopCode {
        #[cfg(feature = "biophys-parallel")]
        {
            self.step_parallel(inputs)
        }
        #[cfg(not(feature = "biophys-parallel"))]
        {
            self.step_serial(inputs)
        }
    }

    pub fn step_serial(&mut self, inputs: &[i32]) -> PopCode {
        self.step_inner(inputs, false)
    }

    #[cfg(feature = "biophys-parallel")]
    pub fn step_parallel(&mut self, inputs: &[i32]) -> PopCode {
        self.step_inner(inputs, true)
    }

    pub fn partition_spike_buffer_sizes(&self) -> Vec<usize> {
        self.partition_spike_buffers
            .iter()
            .map(|buffer| buffer.len())
            .collect()
    }

    pub fn counters_snapshot(&self) -> RuntimeCounters {
        self.counters
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
            update_u8(
                &mut hasher,
                mod_channel_code(params.mod_channel.unwrap_or_default()),
            );
        }
        update_u32(&mut hasher, self.max_events_per_step as u32);
        update_u32(&mut hasher, self.partition_plan.partitions.len() as u32);
        for partition in &self.partition_plan.partitions {
            update_u16(&mut hasher, partition.id);
            update_u32(&mut hasher, partition.neuron_start);
            update_u32(&mut hasher, partition.neuron_end);
        }
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

    fn step_inner(&mut self, inputs: &[i32], parallel: bool) -> PopCode {
        assert_eq!(inputs.len(), self.states.len(), "input length mismatch");
        self.counters.steps_executed = self.counters.steps_executed.saturating_add(1);
        let mods = self.current_modulators;
        if !self.edges.is_empty() {
            for (edge, params) in self.edges.iter_mut().zip(self.stp_params.iter().copied()) {
                edge.weight_effective =
                    compute_effective_weight(mods, edge.weight_base, edge.mod_channel);
                let effective_params = modulated_stp_params(mods, edge, params);
                edge.stp.update_between_spikes(effective_params);
            }
        }

        let mut syn_inputs = vec![0i32; self.states.len()];
        deliver_events(
            self.step_count,
            &mut self.event_queue,
            &mut syn_inputs,
            &mut self.counters,
        );
        self.fill_partition_inputs(&syn_inputs);

        #[cfg(feature = "biophys-parallel")]
        if parallel {
            self.step_partitions_parallel(inputs);
        } else {
            self.step_partitions_serial(inputs);
        }
        #[cfg(not(feature = "biophys-parallel"))]
        {
            let _ = parallel;
            self.step_partitions_serial(inputs);
        }

        let mut merged_spikes = Vec::new();
        for &partition_idx in &self.partition_merge_order {
            merged_spikes.extend_from_slice(&self.partition_spike_buffers[partition_idx]);
        }
        merged_spikes.sort_by_key(|id| id.0);
        let max_spikes = clamp_usize(merged_spikes.len(), self.max_spikes_per_step);
        merged_spikes.truncate(max_spikes);
        self.counters.spikes_total = self
            .counters
            .spikes_total
            .saturating_add(merged_spikes.len() as u64);

        let mut schedule_context = ScheduleContext {
            pre_index: &self.pre_index,
            edges: &mut self.edges,
            stp_params: &self.stp_params,
            mods,
            event_queue: &mut self.event_queue,
            max_events_per_step: self.max_events_per_step,
            dropped_event_count: &mut self.dropped_event_count,
            counters: &mut self.counters,
        };
        schedule_spikes(&merged_spikes, self.step_count, &mut schedule_context);
        self.step_count = self.step_count.saturating_add(1);
        PopCode {
            spikes: merged_spikes,
        }
    }

    fn fill_partition_inputs(&mut self, syn_inputs: &[i32]) {
        for (partition, buffer) in self
            .partition_plan
            .partitions
            .iter()
            .zip(self.partition_input_buffers.iter_mut())
        {
            let start = partition.neuron_start as usize;
            let end = partition.neuron_end as usize;
            buffer.clear();
            buffer.extend_from_slice(&syn_inputs[start..end]);
        }
    }

    fn step_partitions_serial(&mut self, inputs: &[i32]) {
        for (partition_idx, partition) in self.partition_plan.partitions.iter().enumerate() {
            let start = partition.neuron_start as usize;
            let end = partition.neuron_end as usize;
            let params = &self.params[start..end];
            let states = &mut self.states[start..end];
            let inputs = &inputs[start..end];
            let syn_inputs = &self.partition_input_buffers[partition_idx];
            let buffer = &mut self.partition_spike_buffers[partition_idx];
            let mut context = PartitionStepContext {
                dt_ms: self.dt_ms,
                params,
                states,
                inputs,
                syn_inputs,
                spike_buffer: buffer,
                neuron_start: partition.neuron_start,
                max_spikes: self.max_spikes_per_step,
            };
            step_partition(&mut context);
        }
    }

    #[cfg(feature = "biophys-parallel")]
    fn step_partitions_parallel(&mut self, inputs: &[i32]) {
        let mut offset = 0usize;
        let mut params_slice = self.params.as_slice();
        let mut states_slice = self.states.as_mut_slice();
        let mut inputs_slice = inputs;
        let mut syn_inputs_slice = self.partition_input_buffers.as_slice();
        let mut spike_buffers_slice = self.partition_spike_buffers.as_mut_slice();

        std::thread::scope(|scope| {
            for partition in &self.partition_plan.partitions {
                let partition_len = partition.len() as usize;
                let start = partition.neuron_start as usize;
                assert_eq!(start, offset, "partition order mismatch");
                let (params_chunk, params_rest) = params_slice.split_at(partition_len);
                let (states_chunk, states_rest) = states_slice.split_at_mut(partition_len);
                let (inputs_chunk, inputs_rest) = inputs_slice.split_at(partition_len);
                let (syn_inputs_chunk, syn_inputs_rest) = syn_inputs_slice.split_at(1);
                let (spike_buffer_chunk, spike_buffer_rest) = spike_buffers_slice.split_at_mut(1);

                let syn_inputs_chunk = &syn_inputs_chunk[0];
                let spike_buffer_chunk = &mut spike_buffer_chunk[0];
                let dt_ms = self.dt_ms;
                let max_spikes = self.max_spikes_per_step;
                let neuron_start = partition.neuron_start;

                scope.spawn(move || {
                    let mut context = PartitionStepContext {
                        dt_ms,
                        params: params_chunk,
                        states: states_chunk,
                        inputs: inputs_chunk,
                        syn_inputs: syn_inputs_chunk,
                        spike_buffer: spike_buffer_chunk,
                        neuron_start,
                        max_spikes,
                    };
                    step_partition(&mut context);
                });

                offset = offset.saturating_add(partition_len);
                params_slice = params_rest;
                states_slice = states_rest;
                inputs_slice = inputs_rest;
                syn_inputs_slice = syn_inputs_rest;
                spike_buffers_slice = spike_buffer_rest;
            }
        });
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

impl BiophysCircuit<[i32], PopCode> for PartitionedRuntime {
    fn step(&mut self, input: &[i32]) -> PopCode {
        PartitionedRuntime::step(self, input)
    }

    fn config_digest(&self) -> [u8; 32] {
        PartitionedRuntime::config_digest(self)
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
