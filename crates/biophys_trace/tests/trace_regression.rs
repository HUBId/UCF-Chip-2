#![cfg(feature = "biophys-trace")]

use std::fs;

use biophys_assets::{
    demo_channel_params, demo_connectivity, demo_morphology_3comp, demo_syn_params,
};
use biophys_core::{ModLevel, ModulatorField};
use biophys_trace::sn_l4::{
    build_sn_trace_step, mod_level_to_level_class, SnL4TraceInit, SnTraceInput,
};
use biophys_trace::{read_trace, run_trace, write_trace, LearningContext, TraceFile, TraceHeader};
use microcircuit_core::MicrocircuitBackend;

const NEURON_COUNT: u32 = 14;
const MAX_TRACE_BYTES: u64 = 2 * 1024 * 1024;
const EXPECTED_RUN_DIGEST: [u8; 32] = [
    0xe1, 0xac, 0x79, 0xd9, 0x8e, 0x04, 0x35, 0x9e, 0x7a, 0x04, 0x47, 0x76, 0xb5, 0x06, 0x54, 0x9f,
    0x0e, 0xf3, 0x69, 0x55, 0x70, 0x1c, 0x09, 0x62, 0xd2, 0x9b, 0xdc, 0x1a, 0xfc, 0x39, 0x9a, 0x7a,
];

fn build_demo_init() -> SnL4TraceInit {
    let morph = demo_morphology_3comp(NEURON_COUNT);
    let chan = demo_channel_params(&morph);
    let syn = demo_syn_params();
    let conn = demo_connectivity(NEURON_COUNT, &syn);
    SnL4TraceInit::new(morph, chan, syn, conn)
}

fn build_trace(init: &SnL4TraceInit, steps: u32) -> TraceFile {
    let circuit = <microcircuit_sn_l4::SnL4Microcircuit as biophys_asset_builder::CircuitBuilderFromAssets>::build_from_assets(
        &init.morph,
        &init.chan,
        &init.syn,
        &init.conn,
    )
    .expect("build circuit");
    let config_digest = circuit.config_digest();

    let mut trace_steps = Vec::with_capacity(steps as usize);
    for idx in 0..steps {
        let da_level = if idx % 2 == 0 {
            ModLevel::High
        } else {
            ModLevel::Med
        };
        let learning_context = LearningContext {
            in_replay: idx % 3 == 0,
            reward_block: idx % 4 == 0,
            da_level,
        };
        let mods = ModulatorField {
            na: ModLevel::Med,
            da: da_level,
            ht: ModLevel::Low,
        };
        let trace_input = SnTraceInput {
            arousal: mod_level_to_level_class(ModLevel::Med),
            threat: if idx % 5 == 0 {
                dbm_core::LevelClass::High
            } else {
                dbm_core::LevelClass::Low
            },
            stability: mod_level_to_level_class(ModLevel::Low),
            policy_pressure: if idx % 3 == 0 {
                dbm_core::LevelClass::High
            } else {
                dbm_core::LevelClass::Low
            },
            progress: mod_level_to_level_class(da_level),
            integrity: if idx % 7 == 0 {
                dbm_core::IntegrityState::Fail
            } else {
                dbm_core::IntegrityState::Ok
            },
            replay_hint: learning_context.in_replay,
        };
        trace_steps.push(build_sn_trace_step(trace_input, mods, learning_context));
    }

    TraceFile {
        header: TraceHeader {
            trace_version: 1,
            asset_manifest_digest: [0u8; 32],
            circuit_config_digest: config_digest,
            dt_us: 100,
            substeps: 10,
            neuron_count: NEURON_COUNT,
            steps,
        },
        steps: trace_steps,
    }
}

#[test]
fn trace_vector_matches_expected_digest() {
    let init = build_demo_init();
    let trace = build_trace(&init, 12);
    let path = std::env::temp_dir().join("trace_sn_small.bin");
    write_trace(&path, &trace).expect("write trace");

    let metadata = fs::metadata(&path).expect("trace vector exists");
    assert!(
        metadata.len() <= MAX_TRACE_BYTES,
        "trace vector too large: {} bytes",
        metadata.len()
    );

    let read_back = read_trace(&path).expect("read trace");
    let result = run_trace(&read_back, init).expect("run trace");
    assert_eq!(result.run_digest, EXPECTED_RUN_DIGEST);
}
