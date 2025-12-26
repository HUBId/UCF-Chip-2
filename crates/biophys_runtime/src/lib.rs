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
    pub v: Vec<i32>,
    pub refractory: Vec<u16>,
    pub dt_ms: u16,
    pub step_count: u64,
    pub max_spikes_per_step: usize,
    pub current_modulators: ModulatorField,
    pub edges: Vec<SynapseEdge>,
    pub stp_params: Vec<StpParams>,
    pub pre_index: CsrAdjacency,
    pub event_queue: Vec<Vec<SpikeEvent>>,
    pub max_events_per_step: usize,
    pub dropped_event_count: u64,
    counters: RuntimeCounters,
    syn_inputs: Vec<i32>,
    spike_list: Vec<NeuronId>,
    spike_output: Vec<NeuronId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpikeEvent {
    pub deliver_at_step: u64,
    pub post: NeuronId,
    pub current: i32,
    pub synapse_index: usize,
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

#[derive(Debug, Clone)]
pub struct CsrAdjacency {
    pub row_offsets: Vec<usize>,
    pub col_indices: Vec<usize>,
}

impl CsrAdjacency {
    pub fn build(neuron_count: usize, edges: &[SynapseEdge]) -> Self {
        let mut row_offsets = vec![0usize; neuron_count.saturating_add(1)];
        let mut col_indices = Vec::with_capacity(edges.len());
        let mut edge_idx = 0usize;
        for (neuron, row_offset) in row_offsets.iter_mut().enumerate().take(neuron_count) {
            *row_offset = edge_idx;
            while edge_idx < edges.len() && edges[edge_idx].pre.0 as usize == neuron {
                col_indices.push(edge_idx);
                edge_idx = edge_idx.saturating_add(1);
            }
        }
        if neuron_count > 0 {
            row_offsets[neuron_count] = edge_idx;
        }
        debug_assert_eq!(
            edge_idx,
            edges.len(),
            "edges must be sorted by pre for CSR build"
        );
        Self {
            row_offsets,
            col_indices,
        }
    }

    pub fn outgoing(&self, neuron_idx: usize) -> &[usize] {
        if neuron_idx >= self.neuron_count() {
            return &[];
        }
        let start = self.row_offsets[neuron_idx];
        let end = self.row_offsets[neuron_idx.saturating_add(1)];
        &self.col_indices[start..end]
    }

    pub fn neuron_count(&self) -> usize {
        self.row_offsets.len().saturating_sub(1)
    }
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
    let events = &mut event_queue[bucket];
    counters.events_delivered = counters
        .events_delivered
        .saturating_add(events.len() as u64);
    events.sort_by_key(|event| (event.post.0, event.synapse_index));
    for event in events.iter() {
        let post_idx = event.post.0 as usize;
        if post_idx < syn_inputs.len() {
            syn_inputs[post_idx] = syn_inputs[post_idx].saturating_add(event.current);
        }
    }
    events.clear();
}

fn deliver_partitioned_events(
    step_count: u64,
    event_queue: &mut [Vec<Vec<SpikeEvent>>],
    partition_plan: &PartitionPlan,
    partition_input_buffers: &mut [Vec<i32>],
    counters: &mut RuntimeCounters,
) {
    if event_queue.is_empty() {
        return;
    }
    let bucket = (step_count as usize) % event_queue.len();
    for (partition_idx, partition) in partition_plan.partitions.iter().enumerate() {
        let buffer = &mut partition_input_buffers[partition_idx];
        buffer.fill(0);
        let events = &mut event_queue[bucket][partition_idx];
        counters.events_delivered = counters
            .events_delivered
            .saturating_add(events.len() as u64);
        if events.is_empty() {
            continue;
        }
        events.sort_by_key(|event| (event.post.0, event.synapse_index));
        for event in events.iter() {
            let local_idx = (event.post.0 - partition.neuron_start) as usize;
            if local_idx < buffer.len() {
                buffer[local_idx] = buffer[local_idx].saturating_add(event.current);
            }
        }
        events.clear();
    }
}

fn push_event_with_limit(
    bucket: &mut Vec<SpikeEvent>,
    event: SpikeEvent,
    max_events_per_step: usize,
    dropped_event_count: &mut u64,
    counters: &mut RuntimeCounters,
) -> bool {
    if bucket.len() >= max_events_per_step {
        if let Some((max_idx, max_value)) = bucket
            .iter()
            .enumerate()
            .max_by_key(|(_, queued)| queued.synapse_index)
            .map(|(idx, queued)| (idx, queued.synapse_index))
        {
            if event.synapse_index < max_value {
                bucket[max_idx] = event;
                counters.events_pushed = counters.events_pushed.saturating_add(1);
            }
        }
        *dropped_event_count = dropped_event_count.saturating_add(1);
        counters.events_dropped = counters.events_dropped.saturating_add(1);
        return false;
    }
    bucket.push(event);
    counters.events_pushed = counters.events_pushed.saturating_add(1);
    let depth = bucket.len() as u32;
    if depth > counters.max_bucket_depth_seen {
        counters.max_bucket_depth_seen = depth;
    }
    true
}

struct ScheduleContext<'a> {
    pre_index: &'a CsrAdjacency,
    edges: &'a mut [SynapseEdge],
    stp_params: &'a [StpParams],
    mods: ModulatorField,
    event_queue: &'a mut [Vec<SpikeEvent>],
    max_events_per_step: usize,
    dropped_event_count: &'a mut u64,
    counters: &'a mut RuntimeCounters,
}

struct PartitionedScheduleContext<'a> {
    pre_index: &'a CsrAdjacency,
    edges: &'a mut [SynapseEdge],
    stp_params: &'a [StpParams],
    mods: ModulatorField,
    event_queue: &'a mut [Vec<Vec<SpikeEvent>>],
    neuron_to_partition: &'a [usize],
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
        if pre_idx >= context.pre_index.neuron_count() {
            continue;
        }
        for &edge_idx in context.pre_index.outgoing(pre_idx) {
            let edge = &mut context.edges[edge_idx];
            let params = context.stp_params[edge_idx];
            let effective_params = modulated_stp_params(context.mods, edge, params);
            let released = edge.stp.on_spike(effective_params);
            let current = (edge.weight_effective as i64 * released as i64 / 1000) as i32;
            let deliver_at_step = step_count.saturating_add(edge.delay_steps as u64);
            let bucket = (deliver_at_step as usize) % context.event_queue.len();
            let event = SpikeEvent {
                deliver_at_step,
                post: edge.post,
                current,
                synapse_index: edge_idx,
            };
            let _ = push_event_with_limit(
                &mut context.event_queue[bucket],
                event,
                context.max_events_per_step,
                context.dropped_event_count,
                context.counters,
            );
        }
    }
}

fn schedule_spikes_partitioned(
    spikes: &[NeuronId],
    step_count: u64,
    context: &mut PartitionedScheduleContext<'_>,
) {
    if context.event_queue.is_empty() || context.edges.is_empty() {
        return;
    }
    for spike in spikes {
        let pre_idx = spike.0 as usize;
        if pre_idx >= context.pre_index.neuron_count() {
            continue;
        }
        for &edge_idx in context.pre_index.outgoing(pre_idx) {
            let edge = &mut context.edges[edge_idx];
            let params = context.stp_params[edge_idx];
            let effective_params = modulated_stp_params(context.mods, edge, params);
            let released = edge.stp.on_spike(effective_params);
            let current = (edge.weight_effective as i64 * released as i64 / 1000) as i32;
            let deliver_at_step = step_count.saturating_add(edge.delay_steps as u64);
            let bucket = (deliver_at_step as usize) % context.event_queue.len();
            let partition_idx = context.neuron_to_partition[edge.post.0 as usize];
            let event = SpikeEvent {
                deliver_at_step,
                post: edge.post,
                current,
                synapse_index: edge_idx,
            };
            let _ = push_event_with_limit(
                &mut context.event_queue[bucket][partition_idx],
                event,
                context.max_events_per_step,
                context.dropped_event_count,
                context.counters,
            );
        }
    }
}

struct PartitionStepContext<'a> {
    dt_ms: u16,
    params: &'a [LifParams],
    v: &'a mut [i32],
    refractory: &'a mut [u16],
    inputs: &'a [i32],
    syn_inputs: &'a [i32],
    spike_buffer: &'a mut Vec<NeuronId>,
    neuron_start: u32,
    max_spikes: usize,
}

fn step_partition(context: &mut PartitionStepContext<'_>) {
    context.spike_buffer.clear();
    for idx in 0..context.v.len() {
        let mut solver = LifSolver::new(context.params[idx], context.dt_ms);
        let total_input = context.inputs[idx].saturating_add(context.syn_inputs[idx]);
        let mut state = LifState {
            v: context.v[idx],
            refractory_steps: context.refractory[idx],
        };
        let did_spike = solver.step(&mut state, &total_input);
        context.v[idx] = state.v;
        context.refractory[idx] = state.refractory_steps;
        if did_spike {
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
        let neuron_count = params.len();
        let (v, refractory) = split_states(states);
        let (edges, stp_params) = sort_synapses(edges, stp_params);
        let mut max_delay = 0u16;
        for edge in &edges {
            let pre_idx = edge.pre.0 as usize;
            let post_idx = edge.post.0 as usize;
            assert!(pre_idx < v.len(), "edge pre out of range");
            assert!(post_idx < v.len(), "edge post out of range");
            max_delay = max_delay.max(edge.delay_steps);
        }
        let pre_index = CsrAdjacency::build(neuron_count, &edges);
        let queue_len = max_delay as usize + 1;
        let event_queue = (0..queue_len.max(1))
            .map(|_| Vec::with_capacity(max_events_per_step))
            .collect();
        Self {
            params,
            v,
            refractory,
            dt_ms,
            step_count: 0,
            max_spikes_per_step,
            current_modulators: ModulatorField::default(),
            edges,
            stp_params,
            pre_index,
            event_queue,
            max_events_per_step,
            dropped_event_count: 0,
            counters: RuntimeCounters::default(),
            syn_inputs: vec![0i32; neuron_count],
            spike_list: Vec::with_capacity(max_spikes_per_step),
            spike_output: Vec::with_capacity(max_spikes_per_step),
        }
    }

    pub fn set_modulators(&mut self, mods: ModulatorField) {
        self.current_modulators = mods;
    }

    pub fn step(&mut self, inputs: &[i32]) -> PopCode {
        assert_eq!(inputs.len(), self.v.len(), "input length mismatch");
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

        self.syn_inputs.fill(0);
        deliver_events(
            self.step_count,
            &mut self.event_queue,
            &mut self.syn_inputs,
            &mut self.counters,
        );
        self.spike_list.clear();
        for (idx, params) in self.params.iter().enumerate() {
            let mut solver = LifSolver::new(*params, self.dt_ms);
            let total_input = inputs[idx].saturating_add(self.syn_inputs[idx]);
            let mut state = LifState {
                v: self.v[idx],
                refractory_steps: self.refractory[idx],
            };
            if solver.step(&mut state, &total_input) {
                self.spike_list.push(NeuronId(idx as u32));
            }
            self.v[idx] = state.v;
            self.refractory[idx] = state.refractory_steps;
        }
        self.spike_list.sort_by_key(|id| id.0);
        let max_spikes = clamp_usize(self.spike_list.len(), self.max_spikes_per_step);
        self.spike_list.truncate(max_spikes);
        self.counters.spikes_total = self
            .counters
            .spikes_total
            .saturating_add(self.spike_list.len() as u64);

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
        schedule_spikes(&self.spike_list, self.step_count, &mut schedule_context);
        self.step_count = self.step_count.saturating_add(1);
        self.spike_output.clear();
        self.spike_output.extend_from_slice(&self.spike_list);
        PopCode {
            spikes: self.spike_output.clone(),
        }
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
        update_u32(&mut hasher, self.v.len() as u32);
        for (v, refractory) in self.v.iter().zip(self.refractory.iter()) {
            update_i32(&mut hasher, *v);
            update_u16(&mut hasher, *refractory);
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
    pub v: Vec<i32>,
    pub refractory: Vec<u16>,
    pub dt_ms: u16,
    pub step_count: u64,
    pub max_spikes_per_step: usize,
    pub current_modulators: ModulatorField,
    pub edges: Vec<SynapseEdge>,
    pub stp_params: Vec<StpParams>,
    pub pre_index: CsrAdjacency,
    pub event_queue: Vec<Vec<Vec<SpikeEvent>>>,
    pub max_events_per_step: usize,
    pub dropped_event_count: u64,
    counters: RuntimeCounters,
    pub partition_plan: PartitionPlan,
    neuron_to_partition: Vec<usize>,
    partition_spike_buffers: Vec<Vec<NeuronId>>,
    partition_input_buffers: Vec<Vec<i32>>,
    partition_merge_order: Vec<usize>,
    merged_spikes: Vec<NeuronId>,
    merged_spikes_output: Vec<NeuronId>,
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
        let (v, refractory) = split_states(states);
        partition_plan
            .validate(v.len() as u32)
            .expect("invalid partition plan");
        let (edges, stp_params) = sort_synapses(synapses.edges, synapses.stp_params);
        let mut max_delay = 0u16;
        for edge in &edges {
            let pre_idx = edge.pre.0 as usize;
            let post_idx = edge.post.0 as usize;
            assert!(pre_idx < v.len(), "edge pre out of range");
            assert!(post_idx < v.len(), "edge post out of range");
            max_delay = max_delay.max(edge.delay_steps);
        }
        let pre_index = CsrAdjacency::build(v.len(), &edges);
        let queue_len = max_delay as usize + 1;
        let partition_count = partition_plan.partitions.len();
        let partition_spike_buffers = partition_plan
            .partitions
            .iter()
            .map(|partition| Vec::with_capacity(partition.len() as usize))
            .collect();
        let partition_input_buffers = partition_plan
            .partitions
            .iter()
            .map(|partition| vec![0i32; partition.len() as usize])
            .collect();
        let mut partition_merge_order: Vec<usize> = (0..partition_plan.partitions.len()).collect();
        partition_merge_order.sort_by_key(|idx| partition_plan.partitions[*idx].id);
        let mut neuron_to_partition = vec![0usize; v.len()];
        for (partition_idx, partition) in partition_plan.partitions.iter().enumerate() {
            for neuron in partition.neuron_start..partition.neuron_end {
                neuron_to_partition[neuron as usize] = partition_idx;
            }
        }
        let event_queue = (0..queue_len.max(1))
            .map(|_| {
                (0..partition_count)
                    .map(|_| Vec::with_capacity(synapses.max_events_per_step))
                    .collect()
            })
            .collect();
        Self {
            params,
            v,
            refractory,
            dt_ms,
            step_count: 0,
            max_spikes_per_step,
            current_modulators: ModulatorField::default(),
            edges,
            stp_params,
            pre_index,
            event_queue,
            max_events_per_step: synapses.max_events_per_step,
            dropped_event_count: 0,
            counters: RuntimeCounters::default(),
            partition_plan,
            neuron_to_partition,
            partition_spike_buffers,
            partition_input_buffers,
            partition_merge_order,
            merged_spikes: Vec::with_capacity(max_spikes_per_step),
            merged_spikes_output: Vec::with_capacity(max_spikes_per_step),
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
        update_u32(&mut hasher, self.v.len() as u32);
        for (v, refractory) in self.v.iter().zip(self.refractory.iter()) {
            update_i32(&mut hasher, *v);
            update_u16(&mut hasher, *refractory);
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
        assert_eq!(inputs.len(), self.v.len(), "input length mismatch");
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

        deliver_partitioned_events(
            self.step_count,
            &mut self.event_queue,
            &self.partition_plan,
            &mut self.partition_input_buffers,
            &mut self.counters,
        );

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

        self.merged_spikes.clear();
        for &partition_idx in &self.partition_merge_order {
            for spike in &self.partition_spike_buffers[partition_idx] {
                if self.merged_spikes.len() >= self.max_spikes_per_step {
                    break;
                }
                self.merged_spikes.push(*spike);
            }
        }
        self.merged_spikes.sort_by_key(|id| id.0);
        let max_spikes = clamp_usize(self.merged_spikes.len(), self.max_spikes_per_step);
        self.merged_spikes.truncate(max_spikes);
        self.counters.spikes_total = self
            .counters
            .spikes_total
            .saturating_add(self.merged_spikes.len() as u64);

        let mut schedule_context = PartitionedScheduleContext {
            pre_index: &self.pre_index,
            edges: &mut self.edges,
            stp_params: &self.stp_params,
            mods,
            event_queue: &mut self.event_queue,
            neuron_to_partition: &self.neuron_to_partition,
            max_events_per_step: self.max_events_per_step,
            dropped_event_count: &mut self.dropped_event_count,
            counters: &mut self.counters,
        };
        schedule_spikes_partitioned(&self.merged_spikes, self.step_count, &mut schedule_context);
        self.step_count = self.step_count.saturating_add(1);
        self.merged_spikes_output.clear();
        self.merged_spikes_output
            .extend_from_slice(&self.merged_spikes);
        PopCode {
            spikes: self.merged_spikes_output.clone(),
        }
    }

    fn step_partitions_serial(&mut self, inputs: &[i32]) {
        for (partition_idx, partition) in self.partition_plan.partitions.iter().enumerate() {
            let start = partition.neuron_start as usize;
            let end = partition.neuron_end as usize;
            let params = &self.params[start..end];
            let v = &mut self.v[start..end];
            let refractory = &mut self.refractory[start..end];
            let inputs = &inputs[start..end];
            let syn_inputs = &self.partition_input_buffers[partition_idx];
            let buffer = &mut self.partition_spike_buffers[partition_idx];
            let mut context = PartitionStepContext {
                dt_ms: self.dt_ms,
                params,
                v,
                refractory,
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
        let mut v_slice = self.v.as_mut_slice();
        let mut refractory_slice = self.refractory.as_mut_slice();
        let mut inputs_slice = inputs;
        let mut syn_inputs_slice = self.partition_input_buffers.as_slice();
        let mut spike_buffers_slice = self.partition_spike_buffers.as_mut_slice();

        std::thread::scope(|scope| {
            for partition in &self.partition_plan.partitions {
                let partition_len = partition.len() as usize;
                let start = partition.neuron_start as usize;
                assert_eq!(start, offset, "partition order mismatch");
                let (params_chunk, params_rest) = params_slice.split_at(partition_len);
                let (v_chunk, v_rest) = v_slice.split_at_mut(partition_len);
                let (refractory_chunk, refractory_rest) =
                    refractory_slice.split_at_mut(partition_len);
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
                        v: v_chunk,
                        refractory: refractory_chunk,
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
                v_slice = v_rest;
                refractory_slice = refractory_rest;
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

fn split_states(states: Vec<LifState>) -> (Vec<i32>, Vec<u16>) {
    let mut v = Vec::with_capacity(states.len());
    let mut refractory = Vec::with_capacity(states.len());
    for state in states {
        v.push(state.v);
        refractory.push(state.refractory_steps);
    }
    (v, refractory)
}

fn sort_synapses(
    edges: Vec<SynapseEdge>,
    stp_params: Vec<StpParams>,
) -> (Vec<SynapseEdge>, Vec<StpParams>) {
    if edges.is_empty() {
        return (edges, stp_params);
    }
    let mut combined: Vec<(SynapseEdge, StpParams, usize)> = edges
        .into_iter()
        .zip(stp_params)
        .enumerate()
        .map(|(idx, (edge, params))| (edge, params, idx))
        .collect();
    combined.sort_by_key(|(edge, _params, idx)| {
        (
            edge.pre.0,
            edge.post.0,
            *idx as u32,
            edge.delay_steps,
            mod_channel_code(edge.mod_channel),
        )
    });
    let mut edges = Vec::with_capacity(combined.len());
    let mut stp_params = Vec::with_capacity(combined.len());
    for (edge, params, _idx) in combined {
        edges.push(edge);
        stp_params.push(params);
    }
    (edges, stp_params)
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
