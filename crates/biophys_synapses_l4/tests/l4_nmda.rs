#![cfg(all(
    feature = "biophys",
    feature = "biophys-l4",
    feature = "biophys-l4-synapses",
    feature = "biophys-l4-nmda"
))]

use biophys_channels::Leak;
use biophys_compartmental_solver::{CompartmentChannels, L4Solver, L4State};
use biophys_core::{ModChannel, STP_SCALE};
use biophys_event_queue_l4::{QueueLimits, SpikeEventQueueL4};
use biophys_morphology::{Compartment, CompartmentKind, NeuronMorphology};
use biophys_plasticity_l4::StdpTrace;
use biophys_synapses_l4::{
    decay_k, f32_to_fixed_u32, max_synapse_g_fixed, nmda_alpha_q, NmdaVDepMode, SynKind,
    SynapseAccumulator, SynapseL4, SynapseState,
};

const DT_MS: f32 = 0.1;
const THRESHOLD_MV: f32 = -20.0;

#[derive(Clone)]
struct L4Neuron {
    solver: L4Solver,
    state: L4State,
    last_soma_v: f32,
}

fn build_single_compartment_neuron(neuron_id: u32, initial_v: f32) -> L4Neuron {
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

    let channels = vec![CompartmentChannels { leak, nak: None }];

    let solver =
        L4Solver::new(morphology, channels, DT_MS, -120.0, 60.0).expect("solver should init");
    let state = L4State::new(initial_v, 1);

    L4Neuron {
        solver,
        state,
        last_soma_v: initial_v,
    }
}

fn build_nmda_synapse() -> SynapseL4 {
    SynapseL4 {
        pre_neuron: 0,
        post_neuron: 1,
        post_compartment: 0,
        kind: SynKind::NMDA,
        mod_channel: ModChannel::None,
        g_max_base_q: 0,
        g_nmda_base_q: f32_to_fixed_u32(4.0),
        g_max_min_q: 0,
        g_max_max_q: max_synapse_g_fixed(),
        e_rev: 0.0,
        tau_rise_ms: 0.0,
        tau_decay_ms: 0.0,
        tau_decay_nmda_steps: 100,
        nmda_vdep_mode: NmdaVDepMode::PiecewiseLinear,
        delay_steps: 1,
        stp_params: Default::default(),
        stp_state: Default::default(),
        stdp_enabled: false,
        stdp_trace: StdpTrace::default(),
    }
}

fn run_tick_nmda(
    step: u64,
    neurons: &mut [L4Neuron],
    synapses: &[SynapseL4],
    syn_states: &mut [SynapseState],
    queue: &mut SpikeEventQueueL4,
    inputs: &[f32],
) -> Vec<usize> {
    for (state, synapse) in syn_states.iter_mut().zip(synapses.iter()) {
        let k = decay_k(DT_MS, synapse.tau_decay_ms);
        state.decay(synapse.kind, k, synapse.tau_decay_nmda_steps);
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

    spikes
}

#[test]
fn nmda_lookup_table_bounds() {
    assert_eq!(nmda_alpha_q(-70.0, NmdaVDepMode::PiecewiseLinear), 200);
    assert_eq!(nmda_alpha_q(-20.0, NmdaVDepMode::PiecewiseLinear), 1000);
}

#[test]
fn nmda_voltage_dependence_increases_with_depolarization() {
    let mut accumulator = SynapseAccumulator::default();
    accumulator.add(
        SynKind::NMDA,
        f32_to_fixed_u32(1.0),
        0.0,
        NmdaVDepMode::PiecewiseLinear,
    );
    let low = accumulator.total_current(-80.0);
    let high = accumulator.total_current(-20.0);
    assert!(high > low, "NMDA current should increase when depolarized");
}

#[test]
fn nmda_state_clamps_and_decays_deterministically() {
    let max_fixed = max_synapse_g_fixed();
    let mut state = SynapseState::default();
    state.apply_spike(SynKind::NMDA, u32::MAX, STP_SCALE);
    assert!(state.g_nmda_q <= max_fixed);

    let before = state.g_nmda_q;
    state.decay(SynKind::NMDA, 0, 10);
    let expected = before.saturating_sub(before / 10);
    assert_eq!(state.g_nmda_q, expected);
}

#[test]
fn nmda_determinism_matches_between_runs() {
    let synapses = vec![build_nmda_synapse()];

    let run = || {
        let mut neurons = vec![
            build_single_compartment_neuron(0, -65.0),
            build_single_compartment_neuron(1, -65.0),
        ];
        let mut syn_states = vec![SynapseState::default(); synapses.len()];
        let mut queue = SpikeEventQueueL4::new(1, QueueLimits::new(100_000, 1000));
        let mut spikes = Vec::new();

        for step in 0..6 {
            let inputs = if step == 0 {
                vec![1000.0, 0.0]
            } else {
                vec![0.0, 0.0]
            };
            if step == 0 {
                queue.schedule_spike(step, &[0], |_| 1, |_| STP_SCALE);
            }
            let tick_spikes = run_tick_nmda(
                step,
                &mut neurons,
                &synapses,
                &mut syn_states,
                &mut queue,
                &inputs,
            );
            spikes.push(tick_spikes);
        }

        let voltages: Vec<f32> = neurons.iter().map(|n| n.state.comp_v[0]).collect();
        (spikes, voltages, syn_states)
    };

    let (spikes_a, voltages_a, states_a) = run();
    let (spikes_b, voltages_b, states_b) = run();

    assert_eq!(spikes_a, spikes_b, "spike times should match");
    assert_eq!(voltages_a, voltages_b, "final voltages should match");
    assert_eq!(states_a, states_b, "synapse states should match");
}
