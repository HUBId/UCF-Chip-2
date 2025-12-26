#![cfg(feature = "biophys-trace")]

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

fn build_demo_init() -> SnL4TraceInit {
    let morph = demo_morphology_3comp(NEURON_COUNT);
    let chan = demo_channel_params(&morph);
    let syn = demo_syn_params();
    let conn = demo_connectivity(NEURON_COUNT, &syn);
    SnL4TraceInit::new(morph, chan, syn, conn)
}

fn trace_header(config_digest: [u8; 32], steps: u32) -> TraceHeader {
    TraceHeader {
        trace_version: 1,
        asset_manifest_digest: [0u8; 32],
        circuit_config_digest: config_digest,
        dt_us: 100,
        substeps: 10,
        neuron_count: NEURON_COUNT,
        steps,
    }
}

#[test]
fn sn_l4_trace_roundtrip() {
    let init = build_demo_init();
    let circuit = <microcircuit_sn_l4::SnL4Microcircuit as biophys_asset_builder::CircuitBuilderFromAssets>::build_from_assets(
        &init.morph,
        &init.chan,
        &init.syn,
        &init.conn,
    )
    .expect("build circuit");
    let config_digest = circuit.config_digest();

    let steps = 100u32;
    let mut trace_steps = Vec::with_capacity(steps as usize);
    for idx in 0..steps {
        let da_level = if idx % 3 == 0 {
            ModLevel::High
        } else if idx % 2 == 0 {
            ModLevel::Med
        } else {
            ModLevel::Low
        };
        let learning_context = LearningContext {
            in_replay: idx % 2 == 0,
            reward_block: idx % 5 == 0,
            da_level,
        };
        let mods = ModulatorField {
            na: ModLevel::Med,
            da: da_level,
            ht: ModLevel::Low,
        };
        let trace_input = SnTraceInput {
            arousal: mod_level_to_level_class(ModLevel::Med),
            threat: if idx % 7 == 0 {
                dbm_core::LevelClass::High
            } else {
                dbm_core::LevelClass::Low
            },
            stability: mod_level_to_level_class(ModLevel::Low),
            policy_pressure: if idx % 4 == 0 {
                dbm_core::LevelClass::High
            } else {
                dbm_core::LevelClass::Low
            },
            progress: mod_level_to_level_class(da_level),
            integrity: if idx % 9 == 0 {
                dbm_core::IntegrityState::Fail
            } else {
                dbm_core::IntegrityState::Ok
            },
            replay_hint: learning_context.in_replay,
        };
        trace_steps.push(build_sn_trace_step(trace_input, mods, learning_context));
    }

    let trace = TraceFile {
        header: trace_header(config_digest, steps),
        steps: trace_steps,
    };

    let temp_path = std::env::temp_dir().join("sn_l4_trace_roundtrip.bin");
    write_trace(&temp_path, &trace).expect("write trace");
    let read_back = read_trace(&temp_path).expect("read trace");

    let result_a = run_trace(&trace, build_demo_init()).expect("run trace");
    let result_b = run_trace(&read_back, build_demo_init()).expect("run trace read");

    assert_eq!(result_a.run_digest, result_b.run_digest);
}
