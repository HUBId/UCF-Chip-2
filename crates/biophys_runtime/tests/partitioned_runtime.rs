#![cfg(feature = "biophys")]

use biophys_core::{
    LifParams, LifState, ModChannel, NeuronId, Partition, PartitionPlan, StpParams, StpState,
    SynapseEdge,
};
use biophys_runtime::{BiophysRuntime, PartitionedRuntime, SynapseConfig};

fn lif_params() -> LifParams {
    LifParams {
        tau_ms: 1,
        v_rest: 0,
        v_reset: 0,
        v_threshold: 5,
    }
}

fn lif_state() -> LifState {
    LifState {
        v: 0,
        refractory_steps: 0,
    }
}

fn build_plan() -> PartitionPlan {
    PartitionPlan {
        partitions: vec![
            Partition {
                id: 1,
                neuron_start: 0,
                neuron_end: 3,
            },
            Partition {
                id: 0,
                neuron_start: 3,
                neuron_end: 6,
            },
        ],
    }
}

fn build_edges() -> (Vec<SynapseEdge>, Vec<StpParams>) {
    let edges = vec![
        SynapseEdge {
            pre: NeuronId(0),
            post: NeuronId(1),
            weight_base: 6,
            weight_effective: 6,
            delay_steps: 1,
            mod_channel: ModChannel::None,
            stp: StpState { x: 1000, u: 500 },
        },
        SynapseEdge {
            pre: NeuronId(1),
            post: NeuronId(2),
            weight_base: 6,
            weight_effective: 6,
            delay_steps: 1,
            mod_channel: ModChannel::None,
            stp: StpState { x: 1000, u: 500 },
        },
        SynapseEdge {
            pre: NeuronId(2),
            post: NeuronId(3),
            weight_base: 6,
            weight_effective: 6,
            delay_steps: 1,
            mod_channel: ModChannel::None,
            stp: StpState { x: 1000, u: 500 },
        },
        SynapseEdge {
            pre: NeuronId(3),
            post: NeuronId(4),
            weight_base: 6,
            weight_effective: 6,
            delay_steps: 1,
            mod_channel: ModChannel::None,
            stp: StpState { x: 1000, u: 500 },
        },
        SynapseEdge {
            pre: NeuronId(4),
            post: NeuronId(5),
            weight_base: 6,
            weight_effective: 6,
            delay_steps: 1,
            mod_channel: ModChannel::None,
            stp: StpState { x: 1000, u: 500 },
        },
    ];
    let stp = StpParams {
        u: 500,
        tau_rec_steps: 2,
        tau_fac_steps: 2,
        mod_channel: None,
    };
    let stp_params = vec![stp; edges.len()];
    (edges, stp_params)
}

#[test]
fn partitioned_matches_single_runtime() {
    let params = vec![lif_params(); 6];
    let states = vec![lif_state(); 6];
    let (edges, stp_params) = build_edges();
    let mut single = BiophysRuntime::new_with_synapses(
        params.clone(),
        states.clone(),
        1,
        10,
        edges.clone(),
        stp_params.clone(),
        1000,
    );
    let mut partitioned = PartitionedRuntime::new_with_synapses(
        params,
        states,
        1,
        10,
        build_plan(),
        SynapseConfig {
            edges,
            stp_params,
            max_events_per_step: 1000,
        },
    );

    let mut inputs = vec![0; 6];
    inputs[0] = 10;
    for _ in 0..8 {
        let spikes_single = single.step(&inputs);
        let spikes_partitioned = partitioned.step(&inputs);
        assert_eq!(spikes_single.spikes, spikes_partitioned.spikes);
    }
    assert_eq!(single.snapshot_digest(), partitioned.snapshot_digest());
}

#[test]
fn partitioned_is_deterministic() {
    let params = vec![lif_params(); 6];
    let states = vec![lif_state(); 6];
    let (edges, stp_params) = build_edges();
    let mut runtime_a = PartitionedRuntime::new_with_synapses(
        params.clone(),
        states.clone(),
        1,
        10,
        build_plan(),
        SynapseConfig {
            edges: edges.clone(),
            stp_params: stp_params.clone(),
            max_events_per_step: 1000,
        },
    );
    let mut runtime_b = PartitionedRuntime::new_with_synapses(
        params,
        states,
        1,
        10,
        build_plan(),
        SynapseConfig {
            edges,
            stp_params,
            max_events_per_step: 1000,
        },
    );

    let mut inputs = vec![0; 6];
    inputs[0] = 10;
    for _ in 0..8 {
        let spikes_a = runtime_a.step(&inputs);
        let spikes_b = runtime_b.step(&inputs);
        assert_eq!(spikes_a.spikes, spikes_b.spikes);
    }
    assert_eq!(runtime_a.snapshot_digest(), runtime_b.snapshot_digest());
}

#[test]
fn partition_buffers_are_bounded() {
    let params = vec![lif_params(); 6];
    let states = vec![lif_state(); 6];
    let mut runtime = PartitionedRuntime::new(params, states, 1, 1, build_plan());
    let inputs = vec![10; 6];
    let spikes = runtime.step(&inputs);
    assert!(spikes.spikes.len() <= 1);
    for size in runtime.partition_spike_buffer_sizes() {
        assert!(size <= 1);
    }
}

#[cfg(feature = "biophys-parallel")]
#[test]
fn parallel_partitioned_matches_serial_partitioned() {
    let params = vec![lif_params(); 6];
    let states = vec![lif_state(); 6];
    let (edges, stp_params) = build_edges();
    let mut serial = PartitionedRuntime::new_with_synapses(
        params.clone(),
        states.clone(),
        1,
        10,
        build_plan(),
        SynapseConfig {
            edges: edges.clone(),
            stp_params: stp_params.clone(),
            max_events_per_step: 1000,
        },
    );
    let mut parallel = PartitionedRuntime::new_with_synapses(
        params,
        states,
        1,
        10,
        build_plan(),
        SynapseConfig {
            edges,
            stp_params,
            max_events_per_step: 1000,
        },
    );

    let mut inputs = vec![0; 6];
    inputs[0] = 10;
    for _ in 0..8 {
        let spikes_serial = serial.step_serial(&inputs);
        let spikes_parallel = parallel.step(&inputs);
        assert_eq!(spikes_serial.spikes, spikes_parallel.spikes);
    }
    assert_eq!(serial.snapshot_digest(), parallel.snapshot_digest());
}
