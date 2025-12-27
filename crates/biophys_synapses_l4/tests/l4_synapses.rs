#![cfg(all(
    feature = "biophys",
    feature = "biophys-l4",
    feature = "biophys-l4-synapses"
))]

use biophys_channels::Leak;
use biophys_compartmental_solver::{CompartmentChannels, L4Solver, L4State};
use biophys_core::{ModChannel, STP_SCALE};
use biophys_event_queue_l4::{QueueLimits, SpikeEventQueueL4};
use biophys_morphology::{Compartment, CompartmentKind, NeuronMorphology};
use biophys_plasticity_l4::StdpTrace;
use biophys_synapses_l4::{
    decay_k, f32_to_fixed_u32, max_synapse_g_fixed, NmdaVDepMode, SynKind, SynapseAccumulator,
    SynapseL4, SynapseState,
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
        L4Solver::new(morphology, channels, DT_MS, -120.0, 60.0).expect("solver should initialize");
    let state = L4State::new(-65.0, 1);

    L4Neuron {
        solver,
        state,
        last_soma_v: -65.0,
    }
}

fn run_tick(
    step: u64,
    neurons: &mut [L4Neuron],
    synapses: &[SynapseL4],
    syn_states: &mut [SynapseState],
    pre_index: &[Vec<usize>],
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

    for spike_idx in &spikes {
        let indices = &pre_index[*spike_idx];
        queue.schedule_spike(
            step,
            indices,
            |idx| synapses[idx].delay_steps,
            |_| STP_SCALE,
        );
    }

    spikes
}

fn build_pre_index(neuron_count: usize, synapses: &[SynapseL4]) -> Vec<Vec<usize>> {
    let mut pre_index = vec![Vec::new(); neuron_count];
    for (idx, synapse) in synapses.iter().enumerate() {
        let pre = synapse.pre_neuron as usize;
        pre_index[pre].push(idx);
    }
    pre_index
}

#[test]
fn delay_is_applied_to_synaptic_conductance() {
    let mut neurons = vec![
        build_single_compartment_neuron(0),
        build_single_compartment_neuron(1),
    ];
    let synapses = vec![SynapseL4 {
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
        delay_steps: 2,
        stp_params: Default::default(),
        stp_state: Default::default(),
        stdp_enabled: false,
        stdp_trace: StdpTrace::default(),
    }];
    let mut syn_states = vec![SynapseState::default(); synapses.len()];
    let pre_index = build_pre_index(neurons.len(), &synapses);
    let mut queue = SpikeEventQueueL4::new(2, QueueLimits::new(1_000_000, 100_000));

    let mut g_history = Vec::new();
    let mut spike_step = None;
    for step in 0..5 {
        let inputs = if step == 0 {
            vec![1000.0, 0.0]
        } else {
            vec![0.0, 0.0]
        };
        let spikes = run_tick(
            step,
            &mut neurons,
            &synapses,
            &mut syn_states,
            &pre_index,
            &mut queue,
            &inputs,
        );
        if spike_step.is_none() && spikes.contains(&0) {
            spike_step = Some(step);
        }
        g_history.push(syn_states[0].g_fixed_for(SynKind::AMPA));
    }

    assert_eq!(spike_step, Some(0));
    assert_eq!(g_history[0], 0);
    assert_eq!(g_history[1], 0);
    assert!(g_history[2] > 0, "conductance should rise at delay");
}

#[test]
fn deterministic_runs_match() {
    let synapses = vec![SynapseL4 {
        pre_neuron: 0,
        post_neuron: 1,
        post_compartment: 0,
        kind: SynKind::AMPA,
        mod_channel: ModChannel::None,
        g_max_base_q: f32_to_fixed_u32(6.0),
        g_nmda_base_q: 0,
        g_max_min_q: 0,
        g_max_max_q: max_synapse_g_fixed(),
        e_rev: 0.0,
        tau_rise_ms: 0.0,
        tau_decay_ms: 10.0,
        tau_decay_nmda_steps: 100,
        nmda_vdep_mode: NmdaVDepMode::PiecewiseLinear,
        delay_steps: 1,
        stp_params: Default::default(),
        stp_state: Default::default(),
        stdp_enabled: false,
        stdp_trace: StdpTrace::default(),
    }];

    let run = || {
        let mut neurons = vec![
            build_single_compartment_neuron(0),
            build_single_compartment_neuron(1),
        ];
        let mut syn_states = vec![SynapseState::default(); synapses.len()];
        let pre_index = build_pre_index(neurons.len(), &synapses);
        let mut queue = SpikeEventQueueL4::new(1, QueueLimits::new(1_000_000, 100_000));
        let mut spikes = Vec::new();

        for step in 0..6 {
            let inputs = if step == 0 {
                vec![1000.0, 140.0]
            } else {
                vec![0.0, 140.0]
            };
            let tick_spikes = run_tick(
                step,
                &mut neurons,
                &synapses,
                &mut syn_states,
                &pre_index,
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

#[test]
fn bounded_queue_drops_highest_indices() {
    let mut queue = SpikeEventQueueL4::new(0, QueueLimits::new(100, 5));
    let synapse_indices: Vec<usize> = (0..10).collect();
    queue.schedule_spike(0, &synapse_indices, |_| 0, |_| STP_SCALE);

    let events = queue.drain_current(0);
    let delivered: Vec<usize> = events.iter().map(|event| event.synapse_index).collect();
    assert_eq!(delivered, vec![0, 1, 2, 3, 4]);
    assert_eq!(queue.dropped_event_count, 5);
}

#[test]
fn synaptic_input_advances_spike_time() {
    let synapses = vec![SynapseL4 {
        pre_neuron: 0,
        post_neuron: 1,
        post_compartment: 0,
        kind: SynKind::AMPA,
        mod_channel: ModChannel::None,
        g_max_base_q: f32_to_fixed_u32(8.0),
        g_nmda_base_q: 0,
        g_max_min_q: 0,
        g_max_max_q: max_synapse_g_fixed(),
        e_rev: 0.0,
        tau_rise_ms: 0.0,
        tau_decay_ms: 12.0,
        tau_decay_nmda_steps: 100,
        nmda_vdep_mode: NmdaVDepMode::PiecewiseLinear,
        delay_steps: 1,
        stp_params: Default::default(),
        stp_state: Default::default(),
        stdp_enabled: false,
        stdp_trace: StdpTrace::default(),
    }];

    let simulate = |enable_synapse: bool| {
        let mut neurons = vec![
            build_single_compartment_neuron(0),
            build_single_compartment_neuron(1),
        ];
        let mut syn_states = if enable_synapse {
            vec![SynapseState::default(); synapses.len()]
        } else {
            Vec::new()
        };
        let pre_index = if enable_synapse {
            build_pre_index(neurons.len(), &synapses)
        } else {
            vec![Vec::new(); neurons.len()]
        };
        let mut queue = SpikeEventQueueL4::new(1, QueueLimits::new(1_000_000, 100_000));
        let mut first_spike = None;

        for step in 0..6 {
            let inputs = if step == 0 {
                vec![1000.0, 150.0]
            } else {
                vec![0.0, 150.0]
            };
            let synapse_slice: &[SynapseL4] = if enable_synapse { &synapses } else { &[] };
            let tick_spikes = run_tick(
                step,
                &mut neurons,
                synapse_slice,
                &mut syn_states,
                &pre_index,
                &mut queue,
                &inputs,
            );

            if first_spike.is_none() && tick_spikes.contains(&1) {
                first_spike = Some(step);
            }
        }

        first_spike
    };

    let spike_with_synapse = simulate(true);
    let spike_without_synapse = simulate(false);

    assert!(spike_with_synapse.is_some());
    assert!(spike_without_synapse.is_some());
    assert!(
        spike_with_synapse.unwrap() < spike_without_synapse.unwrap(),
        "synaptic input should advance spike time"
    );
}

#[test]
fn synapse_conductance_clamps_without_overflow() {
    let max_fixed = max_synapse_g_fixed();
    let mut state = SynapseState::default();

    for _ in 0..5 {
        state.apply_spike(SynKind::AMPA, u32::MAX, STP_SCALE);
        assert!(
            state.g_ampa_q <= max_fixed,
            "conductance should remain clamped"
        );
    }
}
