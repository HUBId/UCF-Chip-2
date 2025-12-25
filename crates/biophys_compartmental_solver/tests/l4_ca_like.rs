#![cfg(all(
    feature = "biophys-l4",
    feature = "biophys-l4-morphology-multi",
    feature = "biophys-l4-ca"
))]

use biophys_channels::{CaLike, Leak, NaK};
use biophys_compartmental_solver::{CompartmentChannels, L4Solver, L4State};
use biophys_core::NeuronId;
use biophys_morphology::{morphology_tree, CompartmentKind};

const DT_MS: f32 = 0.1;
const CLAMP_MIN: f32 = -120.0;
const CLAMP_MAX: f32 = 60.0;

fn build_solver() -> (L4Solver, L4State, Vec<usize>) {
    let morphology = morphology_tree(NeuronId(2), 15);
    let max_depth = morphology
        .compartments
        .iter()
        .map(|compartment| compartment.depth)
        .max()
        .unwrap_or(0);
    let distal_indices = morphology
        .compartments
        .iter()
        .enumerate()
        .filter(|(_, compartment)| compartment.depth >= max_depth.saturating_sub(1))
        .map(|(index, _)| index)
        .collect::<Vec<_>>();

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
    let ca = CaLike {
        g_ca: 0.8,
        e_ca: 120.0,
    };

    let channels = morphology
        .compartments
        .iter()
        .map(|compartment| CompartmentChannels {
            leak,
            nak: if compartment.kind == CompartmentKind::Soma {
                Some(nak)
            } else {
                None
            },
            ca: Some(ca),
        })
        .collect::<Vec<_>>();

    let compartment_count = morphology.compartments.len();
    let solver = L4Solver::new(morphology, channels, DT_MS, CLAMP_MIN, CLAMP_MAX)
        .expect("solver should init");
    let state = L4State::new(-65.0, compartment_count);

    (solver, state, distal_indices)
}

fn run_simulation(injection: f32, steps: usize) -> (Vec<Vec<u16>>, Vec<bool>, Vec<f32>) {
    let (mut solver, mut state, distal_indices) = build_solver();
    let mut traces = Vec::with_capacity(steps);
    let mut spike_flags = Vec::with_capacity(steps);

    for _ in 0..steps {
        let mut input = vec![0.0_f32; state.voltages.len()];
        for &index in &distal_indices {
            input[index] = injection;
        }
        let output = solver.step_with_output(&mut state, &input);
        traces.push(
            distal_indices
                .iter()
                .map(|&index| state.p_ca_q[index])
                .collect::<Vec<_>>(),
        );
        spike_flags.push(output.ca_spike);
    }

    (traces, spike_flags, state.voltages)
}

#[test]
fn ca_like_determinism_and_bounds() {
    let (trace_a, spikes_a, voltages_a) = run_simulation(20.0, 120);
    let (trace_b, spikes_b, voltages_b) = run_simulation(20.0, 120);

    assert_eq!(trace_a, trace_b, "p_ca_q should be deterministic");
    assert_eq!(spikes_a, spikes_b, "ca_spike flag should be deterministic");
    assert_eq!(voltages_a, voltages_b, "voltages should be deterministic");

    for trace in trace_a {
        for p in trace {
            assert!((0..=1000).contains(&p), "p_ca_q should be bounded");
        }
    }
    for v in voltages_a {
        assert!(v.is_finite(), "voltage should be finite");
        assert!(
            (CLAMP_MIN..=CLAMP_MAX).contains(&v),
            "voltage should be clamped"
        );
    }
}

#[test]
fn ca_like_spike_marker_and_gating_behavior() {
    let (trace_strong, spikes_strong, _) = run_simulation(80.0, 200);
    let (trace_none, spikes_none, _) = run_simulation(0.0, 200);

    let first = trace_strong
        .first()
        .expect("trace")
        .first()
        .copied()
        .unwrap_or(0);
    let last = trace_strong
        .last()
        .expect("trace")
        .first()
        .copied()
        .unwrap_or(0);
    assert!(
        last > first,
        "p_ca_q should increase with strong distal injection"
    );
    assert!(
        spikes_strong.iter().any(|&flag| flag),
        "ca_spike flag should eventually be true"
    );

    for trace in trace_none {
        for p in trace {
            assert_eq!(p, 0, "p_ca_q should stay at rest without injection");
        }
    }
    assert!(
        spikes_none.iter().all(|&flag| !flag),
        "ca_spike flag should remain false without injection"
    );
}
