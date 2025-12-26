#![forbid(unsafe_code)]

use biophys_assets::{
    demo_channel_params, demo_connectivity, demo_morphology_3comp, demo_syn_params,
};
use biophys_core::{ModLevel, ModulatorField};
use biophys_trace::sn_l4::{
    build_sn_trace_step, mod_level_to_level_class, SnL4TraceInit, SnTraceInput,
};
use biophys_trace::{run_trace, write_trace, LearningContext, TraceFile, TraceHeader};
use microcircuit_core::MicrocircuitBackend;

const NEURON_COUNT: u32 = 14;
const TRACE_PATH: &str = "crates/biophys_trace/testvectors/trace_sn_small.bin";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let morph = demo_morphology_3comp(NEURON_COUNT);
    let chan = demo_channel_params(&morph);
    let syn = demo_syn_params();
    let conn = demo_connectivity(NEURON_COUNT, &syn);
    let init = SnL4TraceInit::new(morph, chan, syn, conn);

    let circuit = <microcircuit_sn_l4::SnL4Microcircuit as biophys_asset_builder::CircuitBuilderFromAssets>::build_from_assets(
        &init.morph,
        &init.chan,
        &init.syn,
        &init.conn,
    )?;
    let config_digest = circuit.config_digest();

    let steps = 12u32;
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

    let trace = TraceFile {
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
    };

    write_trace(TRACE_PATH, &trace)?;
    let result = run_trace(&trace, init)?;
    println!("run_digest: {:02x?}", result.run_digest);
    Ok(())
}
