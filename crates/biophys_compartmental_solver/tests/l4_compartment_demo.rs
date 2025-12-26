#![cfg(feature = "biophys-l4")]

use biophys_channels::{Leak, NaK};
use biophys_compartmental_solver::{CompartmentChannels, L4Solver, L4State};
use biophys_core::{CompartmentId, NeuronId};
use biophys_morphology::{Compartment, CompartmentKind, NeuronMorphology};

fn build_solver() -> (L4Solver, L4State) {
    let compartments = vec![
        Compartment {
            id: CompartmentId(0),
            parent: None,
            kind: CompartmentKind::Soma,
            depth: 0,
            capacitance: 1.0,
            axial_resistance: 150.0,
        },
        Compartment {
            id: CompartmentId(1),
            parent: Some(CompartmentId(0)),
            kind: CompartmentKind::Dendrite,
            depth: 1,
            capacitance: 1.2,
            axial_resistance: 200.0,
        },
        Compartment {
            id: CompartmentId(2),
            parent: Some(CompartmentId(0)),
            kind: CompartmentKind::Dendrite,
            depth: 1,
            capacitance: 1.2,
            axial_resistance: 200.0,
        },
    ];

    let morphology = NeuronMorphology {
        neuron_id: NeuronId(0),
        compartments,
    };

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

    let channels = vec![
        CompartmentChannels {
            leak,
            nak: Some(nak),
            #[cfg(feature = "biophys-l4-ca")]
            ca: None,
        },
        CompartmentChannels {
            leak,
            nak: None,
            #[cfg(feature = "biophys-l4-ca")]
            ca: None,
        },
        CompartmentChannels {
            leak,
            nak: None,
            #[cfg(feature = "biophys-l4-ca")]
            ca: None,
        },
    ];

    let solver =
        L4Solver::new(morphology, channels, 0.1, -120.0, 60.0).expect("solver should initialize");
    let state = L4State::new(-65.0, 3);
    (solver, state)
}

fn run_simulation() -> (usize, Vec<f32>) {
    let (mut solver, mut state) = build_solver();
    let mut spike_count = 0;
    let mut prev_v = state.comp_v[0];

    for step in 0..1000 {
        let mut input = vec![0.0_f32; 3];
        if step < 200 {
            input[0] = 10.0;
        }
        solver.step(&mut state, &input);
        let v = state.comp_v[0];
        if prev_v < -20.0 && v >= -20.0 {
            spike_count += 1;
        }
        prev_v = v;
    }

    (spike_count, state.comp_v)
}

#[test]
fn l4_compartment_demo_is_deterministic_and_bounded() {
    let (spikes_a, voltages_a) = run_simulation();
    let (spikes_b, voltages_b) = run_simulation();

    assert_eq!(spikes_a, spikes_b, "spike count should match");
    assert_eq!(voltages_a, voltages_b, "final voltages should match");

    for v in voltages_a {
        assert!(v.is_finite(), "voltage should be finite");
        assert!((-120.0..=60.0).contains(&v), "voltage should be clamped");
    }
}
