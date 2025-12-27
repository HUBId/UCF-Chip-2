#![cfg(all(
    feature = "biophys",
    feature = "biophys-l4",
    feature = "biophys-l4-synapses",
    feature = "biophys-l4-stp"
))]

use biophys_channels::Leak;
use biophys_compartmental_solver::{CompartmentChannels, L4Solver, L4State};
use biophys_core::{ModChannel, STP_SCALE};
use biophys_event_queue_l4::{QueueLimits, SpikeEventQueueL4};
use biophys_morphology::{Compartment, CompartmentKind, NeuronMorphology};
use biophys_plasticity_l4::StdpTrace;
use biophys_synapses_l4::{
    decay_k, f32_to_fixed_u32, max_synapse_g_fixed, NmdaVDepMode, StpMode, StpParamsL4, StpStateL4,
    SynKind, SynapseAccumulator, SynapseL4, SynapseState,
};

const DT_MS: f32 = 0.1;
const THRESHOLD_MV: f32 = -20.0;

#[derive(Clone)]
struct L4Neuron {
    solver: L4Solver,
    state: L4State,
    last_soma_v: f32,
}

fn build_single_compartment_neuron(neuron_id: u32) -> L4Neuron {
    let compartments = vec![Compartment {
        id: biophys_core::CompartmentId(0),
        parent: None,
        kind: CompartmentKind::Soma,
        depth: 0,
        capacitance: 1.0,
        axial_resistance: 150.0,
    }];

    let morphology = NeuronMorphology {
        neuron_id: biophys_core::NeuronId(neuron_id),
        compartments,
    };

    let leak = Leak {
        g: 0.1,
        e_rev: -65.0,
    };

    let channels = vec![CompartmentChannels {
        leak,
        nak: None,
        #[cfg(feature = "biophys-l4-ca")]
        ca: None,
    }];

    let solver =
        L4Solver::new(morphology, channels, DT_MS, -120.0, 60.0).expect("solver should init");
    let state = L4State::new(-65.0, 1);

    L4Neuron {
        solver,
        state,
        last_soma_v: -65.0,
    }
}

fn stp_params() -> StpParamsL4 {
    StpParamsL4 {
        mode: StpMode::STP_TM,
        u_base_q: 500,
        tau_rec_steps: 100,
        tau_fac_steps: 5,
    }
}

fn build_stp_synapse() -> SynapseL4 {
    let params = stp_params();
    SynapseL4 {
        pre_neuron: 0,
        post_neuron: 1,
        post_compartment: 0,
        kind: SynKind::AMPA,
        mod_channel: ModChannel::None,
        g_max_base_q: f32_to_fixed_u32(4.0),
        g_nmda_base_q: 0,
        g_max_min_q: 0,
        g_max_max_q: max_synapse_g_fixed(),
        e_rev: 0.0,
        tau_rise_ms: 0.0,
        tau_decay_ms: 8.0,
        tau_decay_nmda_steps: 100,
        nmda_vdep_mode: NmdaVDepMode::PiecewiseLinear,
        delay_steps: 0,
        stp_params: params,
        stp_state: StpStateL4::new(params),
        stdp_enabled: false,
        stdp_trace: StdpTrace::default(),
    }
}

fn build_pre_index(neuron_count: usize, synapses: &[SynapseL4]) -> Vec<Vec<usize>> {
    let mut pre_index = vec![Vec::new(); neuron_count];
    for (idx, synapse) in synapses.iter().enumerate() {
        let pre = synapse.pre_neuron as usize;
        pre_index[pre].push(idx);
    }
    pre_index
}

fn run_tick_stp(
    step: u64,
    neurons: &mut [L4Neuron],
    synapses: &mut [SynapseL4],
    syn_states: &mut [SynapseState],
    pre_index: &[Vec<usize>],
    queue: &mut SpikeEventQueueL4,
    inputs: &[f32],
) -> Vec<usize> {
    for (state, synapse) in syn_states.iter_mut().zip(synapses.iter_mut()) {
        let k = decay_k(DT_MS, synapse.tau_decay_ms);
        state.decay(synapse.kind, k, synapse.tau_decay_nmda_steps);
        synapse.stp_recover_tick(synapse.stp_params);
    }

    let events = queue.drain_current(step);
    for event in events {
        let synapse = &synapses[event.synapse_index];
        let g_max = synapse.g_max_base_fixed();
        syn_states[event.synapse_index].apply_spike(synapse.kind, g_max, event.release_gain_q);
    }

    let mut accumulators = vec![vec![SynapseAccumulator::default(); 1]; neurons.len()];
    for (idx, synapse) in synapses.iter().enumerate() {
        let g_fixed = syn_states[idx].g_fixed_for(synapse.kind);
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
    for (idx, neuron) in neurons.iter_mut().enumerate() {
        let current = inputs[idx];
        let input = [current];
        let syn_input = [accumulators[idx][0]];
        neuron
            .solver
            .step_with_synapses(&mut neuron.state, &input, &syn_input);
        let v = neuron.state.comp_v[0];
        if neuron.last_soma_v < THRESHOLD_MV && v >= THRESHOLD_MV {
            spikes.push(idx);
        }
        neuron.last_soma_v = v;
    }

    for spike_idx in &spikes {
        let indices = &pre_index[*spike_idx];
        let release_gain = synapses[0].stp_release_on_spike(synapses[0].stp_params);
        queue.schedule_spike(step, indices, |_| 0, |_| release_gain);
    }

    spikes
}

#[test]
fn stp_depression_reduces_release_and_conductance() {
    let mut synapse = build_stp_synapse();
    let params = synapse.stp_params;
    let mut state = SynapseState::default();
    let mut release_gains = Vec::new();
    let mut increments = Vec::new();

    for _ in 0..4 {
        synapse.stp_recover_tick(params);
        let release = synapse.stp_release_on_spike(params);
        release_gains.push(release);
        let before = state.g_ampa_q;
        state.apply_spike(synapse.kind, synapse.g_max_base_fixed(), release);
        increments.push(state.g_ampa_q - before);
    }

    assert!(
        release_gains.windows(2).all(|pair| pair[1] <= pair[0]),
        "release gain should depress with repeated spikes"
    );
    assert!(
        increments.windows(2).all(|pair| pair[1] <= pair[0]),
        "conductance increments should shrink with depression"
    );
}

#[test]
fn stp_recovery_restores_resources() {
    let mut synapse = build_stp_synapse();
    let params = synapse.stp_params;

    for _ in 0..3 {
        synapse.stp_recover_tick(params);
        synapse.stp_release_on_spike(params);
    }
    let depleted = synapse.stp_state.x_q;

    for _ in 0..params.tau_rec_steps {
        synapse.stp_recover_tick(params);
    }

    assert!(
        synapse.stp_state.x_q > depleted,
        "resources should recover toward full after pause"
    );
    assert!(synapse.stp_state.x_q <= STP_SCALE);
}

#[test]
fn stp_determinism_matches() {
    let run = || {
        let mut synapse = build_stp_synapse();
        let params = synapse.stp_params;
        let mut state = SynapseState::default();
        for tick in 0..6 {
            synapse.stp_recover_tick(params);
            if tick % 2 == 0 {
                let release = synapse.stp_release_on_spike(params);
                state.apply_spike(synapse.kind, synapse.g_max_base_fixed(), release);
            }
        }
        (synapse.stp_state, state)
    };

    let (state_a, g_a) = run();
    let (state_b, g_b) = run();

    assert_eq!(state_a, state_b, "stp state should be deterministic");
    assert_eq!(g_a, g_b, "conductance state should be deterministic");
}

#[test]
fn stp_bounds_and_g_clamp_hold() {
    let mut synapse = build_stp_synapse();
    let params = synapse.stp_params;
    let mut state = SynapseState::default();
    let max_fixed = max_synapse_g_fixed();

    for _ in 0..50 {
        synapse.stp_recover_tick(params);
        let release = synapse.stp_release_on_spike(params);
        state.apply_spike(SynKind::AMPA, u32::MAX, release);
        assert!(synapse.stp_state.x_q <= STP_SCALE);
        assert!(synapse.stp_state.u_q <= STP_SCALE);
        assert!(state.g_ampa_q <= max_fixed);
    }
}

#[test]
fn stp_integration_shows_depression_over_repeated_spikes() {
    let mut neurons = vec![
        build_single_compartment_neuron(0),
        build_single_compartment_neuron(1),
    ];
    let mut synapses = vec![build_stp_synapse()];
    let mut syn_states = vec![SynapseState::default()];
    let pre_index = build_pre_index(neurons.len(), &synapses);
    let mut queue = SpikeEventQueueL4::new(0, QueueLimits::new(100_000, 1000));

    let mut g_increments = Vec::new();
    for step in 0..4 {
        let inputs = vec![1000.0, 0.0];
        let before = syn_states[0].g_ampa_q;
        run_tick_stp(
            step,
            &mut neurons,
            &mut synapses,
            &mut syn_states,
            &pre_index,
            &mut queue,
            &inputs,
        );
        g_increments.push(syn_states[0].g_ampa_q - before);
    }

    assert!(
        g_increments.windows(2).all(|pair| pair[1] <= pair[0]),
        "postsynaptic response should depress over repeated firing"
    );
}
