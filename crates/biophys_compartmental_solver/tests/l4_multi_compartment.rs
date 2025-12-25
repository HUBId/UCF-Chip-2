#![cfg(all(feature = "biophys-l4", feature = "biophys-l4-morphology-multi"))]

use biophys_channels::{Leak, NaK};
use biophys_compartmental_solver::{CompartmentChannels, L4Solver, L4State};
use biophys_core::NeuronId;
use biophys_morphology::{morphology_tree, CompartmentKind};

const DT_MS: f32 = 0.1;
const CLAMP_MIN: f32 = -90.0;
const CLAMP_MAX: f32 = 50.0;

fn build_channels(morphology: &biophys_morphology::NeuronMorphology) -> Vec<CompartmentChannels> {
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

    morphology
        .compartments
        .iter()
        .map(|compartment| CompartmentChannels {
            leak,
            nak: if compartment.kind == CompartmentKind::Soma {
                Some(nak)
            } else {
                None
            },
            #[cfg(feature = "biophys-l4-ca")]
            ca: None,
        })
        .collect()
}

#[test]
fn multi_compartment_determinism_and_stability() {
    let morphology = morphology_tree(NeuronId(1), 15);
    let channels = build_channels(&morphology);
    let mut solver = L4Solver::new(
        morphology.clone(),
        channels.clone(),
        DT_MS,
        CLAMP_MIN,
        CLAMP_MAX,
    )
    .expect("solver");
    let mut state = L4State::new(-65.0, morphology.compartments.len());
    let mut input = vec![0.0_f32; morphology.compartments.len()];
    input[0] = 1.5;

    for _ in 0..100 {
        solver.step(&mut state, &input);
    }

    for v in &state.voltages {
        assert!(v.is_finite());
        assert!(*v >= CLAMP_MIN && *v <= CLAMP_MAX);
    }

    let snapshot_a = solver.snapshot_digest(&state);
    let config_a = solver.config_digest();
    let compartment_count = state.voltages.len();

    let mut solver_b =
        L4Solver::new(morphology, channels, DT_MS, CLAMP_MIN, CLAMP_MAX).expect("solver");
    let mut state_b = L4State::new(-65.0, compartment_count);
    let mut input_b = vec![0.0_f32; compartment_count];
    input_b[0] = 1.5;
    for _ in 0..100 {
        solver_b.step(&mut state_b, &input_b);
    }

    let snapshot_b = solver_b.snapshot_digest(&state_b);
    let config_b = solver_b.config_digest();

    assert_eq!(snapshot_a, snapshot_b);
    assert_eq!(config_a, config_b);
}
