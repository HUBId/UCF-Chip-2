#![cfg(feature = "biophys")]

use biophys_core::{LifParams, LifState, ModChannel, NeuronId, StpParams, StpState, SynapseEdge};
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
fn csr_adjacency_builds_expected_slices() {
    let params = vec![lif_params(); 3];
    let states = vec![lif_state(); 3];
    let edges = vec![
        SynapseEdge {
            pre: NeuronId(2),
            post: NeuronId(0),
            weight_base: 6,
            weight_effective: 6,
            delay_steps: 2,
            mod_channel: ModChannel::None,
            stp: StpState { x: 1000, u: 500 },
        },
        SynapseEdge {
            pre: NeuronId(0),
            post: NeuronId(2),
            weight_base: 6,
            weight_effective: 6,
            delay_steps: 1,
            mod_channel: ModChannel::None,
            stp: StpState { x: 1000, u: 500 },
        },
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
    ];
    let stp = StpParams {
        u: 500,
        tau_rec_steps: 2,
        tau_fac_steps: 2,
        mod_channel: None,
    };
    let runtime = BiophysRuntime::new_with_synapses(
        params,
        states,
        1,
        10,
        edges,
        vec![stp; 4],
        1000,
    );
    let sorted_pairs: Vec<(u32, u32)> = runtime
        .edges
        .iter()
        .map(|edge| (edge.pre.0, edge.post.0))
        .collect();
    assert_eq!(sorted_pairs, vec![(0, 1), (0, 2), (1, 2), (2, 0)]);
    assert_eq!(runtime.pre_index.row_offsets, vec![0, 2, 3, 4]);
    assert_eq!(runtime.pre_index.col_indices, vec![0, 1, 2, 3]);
}
