#![cfg(feature = "biophys")]

use biophys_core::{LifParams, LifState, NeuronId, StpParams, StpState, SynapseEdge};
use biophys_runtime::BiophysRuntime;

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

#[test]
fn delayed_delivery_arrives_after_delay() {
    let params = vec![lif_params(), lif_params()];
    let states = vec![lif_state(), lif_state()];
    let edge = SynapseEdge {
        pre: NeuronId(0),
        post: NeuronId(1),
        weight: 10,
        delay_steps: 2,
        stp: StpState { x: 1000, u: 1000 },
    };
    let stp = StpParams {
        u: 1000,
        tau_rec_steps: 1,
        tau_fac_steps: 1,
    };
    let mut runtime = BiophysRuntime::new_with_synapses(
        params,
        states,
        1,
        10,
        vec![edge],
        vec![stp],
        10_000,
    );

    let pop0 = runtime.step(&[10, 0]);
    assert_eq!(pop0.spikes, vec![NeuronId(0)]);

    let pop1 = runtime.step(&[0, 0]);
    assert!(pop1.spikes.is_empty());

    let pop2 = runtime.step(&[0, 0]);
    assert_eq!(pop2.spikes, vec![NeuronId(1)]);
}

#[test]
fn network_is_deterministic_across_runs() {
    let params = vec![lif_params(); 5];
    let states = vec![lif_state(); 5];
    let edges = vec![
        SynapseEdge {
            pre: NeuronId(0),
            post: NeuronId(1),
            weight: 6,
            delay_steps: 1,
            stp: StpState { x: 1000, u: 500 },
        },
        SynapseEdge {
            pre: NeuronId(1),
            post: NeuronId(2),
            weight: 6,
            delay_steps: 1,
            stp: StpState { x: 1000, u: 500 },
        },
        SynapseEdge {
            pre: NeuronId(2),
            post: NeuronId(3),
            weight: 6,
            delay_steps: 1,
            stp: StpState { x: 1000, u: 500 },
        },
        SynapseEdge {
            pre: NeuronId(3),
            post: NeuronId(4),
            weight: 6,
            delay_steps: 1,
            stp: StpState { x: 1000, u: 500 },
        },
    ];
    let stp = StpParams {
        u: 500,
        tau_rec_steps: 2,
        tau_fac_steps: 2,
    };
    let stp_params = vec![stp; edges.len()];

    let mut runtime_a =
        BiophysRuntime::new_with_synapses(params.clone(), states.clone(), 1, 10, edges.clone(), stp_params.clone(), 1000);
    let mut runtime_b =
        BiophysRuntime::new_with_synapses(params, states, 1, 10, edges, stp_params, 1000);

    let mut spikes_a = Vec::new();
    let mut spikes_b = Vec::new();
    let mut inputs = vec![0; 5];
    inputs[0] = 10;

    for _ in 0..10 {
        spikes_a.push(runtime_a.step(&inputs).spikes);
        spikes_b.push(runtime_b.step(&inputs).spikes);
    }

    assert_eq!(spikes_a, spikes_b);
    assert_eq!(runtime_a.snapshot_digest(), runtime_b.snapshot_digest());
}

#[test]
fn event_queue_is_bounded_and_deterministic() {
    let params = vec![lif_params(); 6];
    let states = vec![lif_state(); 6];
    let mut edges = Vec::new();
    for post in 1..6 {
        edges.push(SynapseEdge {
            pre: NeuronId(0),
            post: NeuronId(post),
            weight: 10,
            delay_steps: 1,
            stp: StpState { x: 1000, u: 1000 },
        });
    }
    let stp = StpParams {
        u: 1000,
        tau_rec_steps: 1,
        tau_fac_steps: 1,
    };
    let stp_params = vec![stp; edges.len()];
    let mut runtime = BiophysRuntime::new_with_synapses(
        params,
        states,
        1,
        10,
        edges,
        stp_params,
        3,
    );

    let pop0 = runtime.step(&[10, 0, 0, 0, 0, 0]);
    assert_eq!(pop0.spikes, vec![NeuronId(0)]);
    assert_eq!(runtime.dropped_event_count, 2);

    let pop1 = runtime.step(&[0, 0, 0, 0, 0, 0]);
    assert_eq!(
        pop1.spikes,
        vec![NeuronId(1), NeuronId(2), NeuronId(3)]
    );
    assert_eq!(runtime.dropped_event_count, 2);
}
