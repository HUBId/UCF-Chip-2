#![forbid(unsafe_code)]

use crate::{
    digest_inputs, CircuitInit, LearningContext, TraceError, TraceFile, TraceObservation, TraceStep,
};
use biophys_asset_builder::CircuitBuilderFromAssets;
use biophys_assets::{ChannelParamsSet, ConnectivityGraph, MorphologySet, SynapseParamsSet};
use biophys_core::{ModLevel, ModulatorField};
use dbm_core::{IntegrityState, IsvSnapshot, LevelClass};
use microcircuit_core::MicrocircuitBackend;
use microcircuit_sn_l4::SnL4Microcircuit;
use microcircuit_sn_stub::SnInput;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnTraceInput {
    pub arousal: LevelClass,
    pub threat: LevelClass,
    pub stability: LevelClass,
    pub policy_pressure: LevelClass,
    pub progress: LevelClass,
    pub integrity: IntegrityState,
    pub replay_hint: bool,
}

impl Default for SnTraceInput {
    fn default() -> Self {
        Self {
            arousal: LevelClass::Low,
            threat: LevelClass::Low,
            stability: LevelClass::Low,
            policy_pressure: LevelClass::Low,
            progress: LevelClass::Low,
            integrity: IntegrityState::Ok,
            replay_hint: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SnL4TraceInit {
    pub morph: MorphologySet,
    pub chan: ChannelParamsSet,
    pub syn: SynapseParamsSet,
    pub conn: ConnectivityGraph,
    pub asset_manifest_digest: Option<[u8; 32]>,
}

impl SnL4TraceInit {
    pub fn new(
        morph: MorphologySet,
        chan: ChannelParamsSet,
        syn: SynapseParamsSet,
        conn: ConnectivityGraph,
    ) -> Self {
        Self {
            morph,
            chan,
            syn,
            conn,
            asset_manifest_digest: None,
        }
    }
}

impl CircuitInit for SnL4TraceInit {
    type Input = SnInput;
    type Output = microcircuit_sn_stub::SnOutput;
    type Circuit = SnL4Microcircuit;

    fn init(&self, trace: &TraceFile) -> Result<Self::Circuit, TraceError> {
        if let Some(expected) = self.asset_manifest_digest {
            if trace.header.asset_manifest_digest != expected {
                return Err(TraceError::Validation {
                    message: "asset manifest digest mismatch".to_string(),
                });
            }
        }
        let circuit = <SnL4Microcircuit as CircuitBuilderFromAssets>::build_from_assets(
            &self.morph,
            &self.chan,
            &self.syn,
            &self.conn,
        )
        .map_err(|error| TraceError::Validation {
            message: format!("asset build failed: {error:?}"),
        })?;
        if circuit.config_digest() != trace.header.circuit_config_digest {
            return Err(TraceError::Validation {
                message: "circuit config digest mismatch".to_string(),
            });
        }
        Ok(circuit)
    }

    fn decode_input(&self, step: &TraceStep) -> Result<Self::Input, TraceError> {
        let trace_input = decode_sn_trace_input(&step.inputs_bytes)?;
        if trace_input.replay_hint != step.learning_context.in_replay {
            return Err(TraceError::Validation {
                message: "replay hint mismatch with learning context".to_string(),
            });
        }
        let mut isv = IsvSnapshot::default();
        isv.arousal = trace_input.arousal;
        isv.threat = trace_input.threat;
        isv.stability = trace_input.stability;
        isv.policy_pressure = trace_input.policy_pressure;
        isv.progress = trace_input.progress;
        isv.integrity = trace_input.integrity;
        isv.replay_hint = trace_input.replay_hint;

        Ok(SnInput {
            isv,
            cooldown_class: None,
            current_dwm: None,
            replay_hint: trace_input.replay_hint,
            reward_block: step.learning_context.reward_block,
            modulators: step.mods,
        })
    }

    fn observe(
        &self,
        circuit: &Self::Circuit,
        _output: &Self::Output,
    ) -> Result<TraceObservation, TraceError> {
        let spike_counts = circuit
            .last_spike_counts()
            .iter()
            .map(|value| u32::from(*value))
            .collect::<Vec<_>>();
        let pool_acc = circuit.pool_accumulators().to_vec();
        Ok(TraceObservation {
            spike_counts,
            pool_acc,
        })
    }
}

pub fn build_sn_trace_step(
    trace_input: SnTraceInput,
    mods: ModulatorField,
    learning_context: LearningContext,
) -> TraceStep {
    let inputs_bytes = encode_sn_trace_input(trace_input);
    let inputs_digest = digest_inputs(&inputs_bytes);
    TraceStep {
        inputs_digest,
        inputs_bytes,
        mods,
        learning_context,
    }
}

pub fn encode_sn_trace_input(input: SnTraceInput) -> Vec<u8> {
    vec![
        level_class_code(input.arousal),
        level_class_code(input.threat),
        level_class_code(input.stability),
        level_class_code(input.policy_pressure),
        level_class_code(input.progress),
        integrity_code(input.integrity),
        input.replay_hint as u8,
    ]
}

pub fn decode_sn_trace_input(bytes: &[u8]) -> Result<SnTraceInput, TraceError> {
    if bytes.len() != 7 {
        return Err(TraceError::InvalidFormat {
            message: format!("invalid sn trace input length {}", bytes.len()),
        });
    }
    Ok(SnTraceInput {
        arousal: decode_level_class(bytes[0])?,
        threat: decode_level_class(bytes[1])?,
        stability: decode_level_class(bytes[2])?,
        policy_pressure: decode_level_class(bytes[3])?,
        progress: decode_level_class(bytes[4])?,
        integrity: decode_integrity(bytes[5])?,
        replay_hint: bytes[6] != 0,
    })
}

fn level_class_code(level: LevelClass) -> u8 {
    match level {
        LevelClass::Low => 0,
        LevelClass::Med => 1,
        LevelClass::High => 2,
    }
}

fn decode_level_class(code: u8) -> Result<LevelClass, TraceError> {
    match code {
        0 => Ok(LevelClass::Low),
        1 => Ok(LevelClass::Med),
        2 => Ok(LevelClass::High),
        other => Err(TraceError::InvalidFormat {
            message: format!("invalid level class {other}"),
        }),
    }
}

fn integrity_code(value: IntegrityState) -> u8 {
    match value {
        IntegrityState::Ok => 0,
        IntegrityState::Fail => 1,
        IntegrityState::Degraded => 2,
    }
}

fn decode_integrity(code: u8) -> Result<IntegrityState, TraceError> {
    match code {
        0 => Ok(IntegrityState::Ok),
        1 => Ok(IntegrityState::Fail),
        2 => Ok(IntegrityState::Degraded),
        other => Err(TraceError::InvalidFormat {
            message: format!("invalid integrity state {other}"),
        }),
    }
}

pub fn mod_level_to_level_class(level: ModLevel) -> LevelClass {
    match level {
        ModLevel::Low => LevelClass::Low,
        ModLevel::Med => LevelClass::Med,
        ModLevel::High => LevelClass::High,
    }
}
