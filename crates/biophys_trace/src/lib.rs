#![forbid(unsafe_code)]

use biophys_core::{ModLevel, ModulatorField};
use blake3::Hasher;
use microcircuit_core::MicrocircuitBackend;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use thiserror::Error;

const TRACE_INPUT_DIGEST_DOMAIN: &str = "UCF:TRACE:INPUT";
const TRACE_SPIKE_DIGEST_DOMAIN: &str = "UCF:TRACE:SPIKE_TRAIN";
const TRACE_POOL_DIGEST_DOMAIN: &str = "UCF:TRACE:POOL_ACC";
const TRACE_RUN_DIGEST_DOMAIN: &str = "UCF:TRACE:RUN";
const MAX_INPUT_BYTES: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceHeader {
    pub trace_version: u32,
    pub asset_manifest_digest: [u8; 32],
    pub circuit_config_digest: [u8; 32],
    pub dt_us: u32,
    pub substeps: u32,
    pub neuron_count: u32,
    pub steps: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LearningContext {
    pub in_replay: bool,
    pub reward_block: bool,
    pub da_level: ModLevel,
}

impl Default for LearningContext {
    fn default() -> Self {
        Self {
            in_replay: false,
            reward_block: false,
            da_level: ModLevel::Med,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceStep {
    pub inputs_digest: [u8; 32],
    pub inputs_bytes: Vec<u8>,
    pub mods: ModulatorField,
    pub learning_context: LearningContext,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceFile {
    pub header: TraceHeader,
    pub steps: Vec<TraceStep>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceObservation {
    pub spike_counts: Vec<u32>,
    pub pool_acc: Vec<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceRunResult {
    pub spike_train_digest: [u8; 32],
    pub pool_acc_digest: [u8; 32],
    pub final_snapshot_digest: [u8; 32],
    pub run_digest: [u8; 32],
}

#[derive(Debug, Error)]
pub enum TraceError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("trace encoding error: {message}")]
    InvalidFormat { message: String },
    #[error("trace input too large: {len} bytes")]
    InputTooLarge { len: usize },
    #[error("trace validation failed: {message}")]
    Validation { message: String },
}

pub trait CircuitInit {
    type Input;
    type Output;
    type Circuit: MicrocircuitBackend<Self::Input, Self::Output>;

    fn init(&self, trace: &TraceFile) -> Result<Self::Circuit, TraceError>;

    fn decode_input(&self, step: &TraceStep) -> Result<Self::Input, TraceError>;

    fn observe(
        &self,
        circuit: &Self::Circuit,
        output: &Self::Output,
    ) -> Result<TraceObservation, TraceError>;
}

pub fn digest_inputs(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(TRACE_INPUT_DIGEST_DOMAIN.as_bytes());
    hasher.update(bytes);
    *hasher.finalize().as_bytes()
}

pub fn write_trace(path: impl AsRef<Path>, trace: &TraceFile) -> Result<(), TraceError> {
    if trace.steps.len() != trace.header.steps as usize {
        return Err(TraceError::Validation {
            message: format!(
                "header steps {} does not match trace steps {}",
                trace.header.steps,
                trace.steps.len()
            ),
        });
    }

    let mut file = File::create(path)?;
    write_header(&mut file, &trace.header)?;
    for step in &trace.steps {
        write_step(&mut file, step)?;
    }
    file.flush()?;
    Ok(())
}

pub fn read_trace(path: impl AsRef<Path>) -> Result<TraceFile, TraceError> {
    let mut file = File::open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    let mut cursor = 0usize;

    let header = read_header(&bytes, &mut cursor)?;
    let mut steps = Vec::with_capacity(header.steps as usize);
    for _ in 0..header.steps {
        steps.push(read_step(&bytes, &mut cursor)?);
    }

    if cursor != bytes.len() {
        return Err(TraceError::InvalidFormat {
            message: "extra bytes at end of trace".to_string(),
        });
    }

    Ok(TraceFile { header, steps })
}

pub fn run_trace<I: CircuitInit>(trace: &TraceFile, init: I) -> Result<TraceRunResult, TraceError> {
    if trace.steps.len() != trace.header.steps as usize {
        return Err(TraceError::Validation {
            message: format!(
                "header steps {} does not match trace steps {}",
                trace.header.steps,
                trace.steps.len()
            ),
        });
    }

    let mut circuit = init.init(trace)?;
    let mut spike_hasher = Hasher::new();
    spike_hasher.update(TRACE_SPIKE_DIGEST_DOMAIN.as_bytes());
    let mut pool_hasher = Hasher::new();
    pool_hasher.update(TRACE_POOL_DIGEST_DOMAIN.as_bytes());

    for (step_idx, step) in trace.steps.iter().enumerate() {
        let computed = digest_inputs(&step.inputs_bytes);
        if computed != step.inputs_digest {
            return Err(TraceError::Validation {
                message: format!("inputs digest mismatch at step {step_idx}"),
            });
        }
        let input = init.decode_input(step)?;
        let output = circuit.step(&input, step_idx as u64);
        let observation = init.observe(&circuit, &output)?;
        if observation.spike_counts.len() != trace.header.neuron_count as usize {
            return Err(TraceError::Validation {
                message: format!(
                    "spike counts len {} does not match neuron count {}",
                    observation.spike_counts.len(),
                    trace.header.neuron_count
                ),
            });
        }
        for (neuron_id, count) in observation.spike_counts.iter().enumerate() {
            spike_hasher.update(&(neuron_id as u32).to_le_bytes());
            spike_hasher.update(&count.to_le_bytes());
        }
        for value in observation.pool_acc {
            pool_hasher.update(&value.to_le_bytes());
        }
    }

    let spike_train_digest = *spike_hasher.finalize().as_bytes();
    let pool_acc_digest = *pool_hasher.finalize().as_bytes();
    let final_snapshot_digest = circuit.snapshot_digest();
    let run_digest = digest_run(spike_train_digest, pool_acc_digest, final_snapshot_digest);

    Ok(TraceRunResult {
        spike_train_digest,
        pool_acc_digest,
        final_snapshot_digest,
        run_digest,
    })
}

fn digest_run(
    spike_train_digest: [u8; 32],
    pool_acc_digest: [u8; 32],
    final_snapshot_digest: [u8; 32],
) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(TRACE_RUN_DIGEST_DOMAIN.as_bytes());
    hasher.update(&spike_train_digest);
    hasher.update(&pool_acc_digest);
    hasher.update(&final_snapshot_digest);
    *hasher.finalize().as_bytes()
}

fn write_header(mut writer: impl Write, header: &TraceHeader) -> Result<(), TraceError> {
    writer.write_all(&header.trace_version.to_le_bytes())?;
    writer.write_all(&header.asset_manifest_digest)?;
    writer.write_all(&header.circuit_config_digest)?;
    writer.write_all(&header.dt_us.to_le_bytes())?;
    writer.write_all(&header.substeps.to_le_bytes())?;
    writer.write_all(&header.neuron_count.to_le_bytes())?;
    writer.write_all(&header.steps.to_le_bytes())?;
    Ok(())
}

fn write_step(mut writer: impl Write, step: &TraceStep) -> Result<(), TraceError> {
    if step.inputs_bytes.len() > MAX_INPUT_BYTES {
        return Err(TraceError::InputTooLarge {
            len: step.inputs_bytes.len(),
        });
    }
    writer.write_all(&step.inputs_digest)?;
    writer.write_all(&(step.inputs_bytes.len() as u32).to_le_bytes())?;
    writer.write_all(&step.inputs_bytes)?;
    write_modulators(&mut writer, step.mods)?;
    write_learning_context(&mut writer, step.learning_context)?;
    Ok(())
}

fn read_header(bytes: &[u8], cursor: &mut usize) -> Result<TraceHeader, TraceError> {
    Ok(TraceHeader {
        trace_version: read_u32(bytes, cursor)?,
        asset_manifest_digest: read_digest(bytes, cursor)?,
        circuit_config_digest: read_digest(bytes, cursor)?,
        dt_us: read_u32(bytes, cursor)?,
        substeps: read_u32(bytes, cursor)?,
        neuron_count: read_u32(bytes, cursor)?,
        steps: read_u32(bytes, cursor)?,
    })
}

fn read_step(bytes: &[u8], cursor: &mut usize) -> Result<TraceStep, TraceError> {
    let inputs_digest = read_digest(bytes, cursor)?;
    let inputs_len = read_u32(bytes, cursor)? as usize;
    if inputs_len > MAX_INPUT_BYTES {
        return Err(TraceError::InputTooLarge { len: inputs_len });
    }
    let inputs_bytes = read_vec(bytes, cursor, inputs_len)?;
    let mods = read_modulators(bytes, cursor)?;
    let learning_context = read_learning_context(bytes, cursor)?;
    Ok(TraceStep {
        inputs_digest,
        inputs_bytes,
        mods,
        learning_context,
    })
}

fn write_modulators(mut writer: impl Write, mods: ModulatorField) -> Result<(), TraceError> {
    writer.write_all(&[mod_level_code(mods.na)])?;
    writer.write_all(&[mod_level_code(mods.da)])?;
    writer.write_all(&[mod_level_code(mods.ht)])?;
    Ok(())
}

fn read_modulators(bytes: &[u8], cursor: &mut usize) -> Result<ModulatorField, TraceError> {
    let na = read_mod_level(bytes, cursor)?;
    let da = read_mod_level(bytes, cursor)?;
    let ht = read_mod_level(bytes, cursor)?;
    Ok(ModulatorField { na, da, ht })
}

fn write_learning_context(
    mut writer: impl Write,
    context: LearningContext,
) -> Result<(), TraceError> {
    writer.write_all(&[context.in_replay as u8])?;
    writer.write_all(&[context.reward_block as u8])?;
    writer.write_all(&[mod_level_code(context.da_level)])?;
    Ok(())
}

fn read_learning_context(bytes: &[u8], cursor: &mut usize) -> Result<LearningContext, TraceError> {
    let in_replay = read_u8(bytes, cursor)? != 0;
    let reward_block = read_u8(bytes, cursor)? != 0;
    let da_level = read_mod_level(bytes, cursor)?;
    Ok(LearningContext {
        in_replay,
        reward_block,
        da_level,
    })
}

fn mod_level_code(level: ModLevel) -> u8 {
    match level {
        ModLevel::Low => 0,
        ModLevel::Med => 1,
        ModLevel::High => 2,
    }
}

fn read_mod_level(bytes: &[u8], cursor: &mut usize) -> Result<ModLevel, TraceError> {
    match read_u8(bytes, cursor)? {
        0 => Ok(ModLevel::Low),
        1 => Ok(ModLevel::Med),
        2 => Ok(ModLevel::High),
        other => Err(TraceError::InvalidFormat {
            message: format!("invalid mod level {other}"),
        }),
    }
}

fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32, TraceError> {
    let end = cursor
        .checked_add(4)
        .ok_or_else(|| TraceError::InvalidFormat {
            message: "unexpected eof".to_string(),
        })?;
    if end > bytes.len() {
        return Err(TraceError::InvalidFormat {
            message: "unexpected eof".to_string(),
        });
    }
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&bytes[*cursor..end]);
    *cursor = end;
    Ok(u32::from_le_bytes(buf))
}

fn read_u8(bytes: &[u8], cursor: &mut usize) -> Result<u8, TraceError> {
    let end = cursor
        .checked_add(1)
        .ok_or_else(|| TraceError::InvalidFormat {
            message: "unexpected eof".to_string(),
        })?;
    if end > bytes.len() {
        return Err(TraceError::InvalidFormat {
            message: "unexpected eof".to_string(),
        });
    }
    let value = bytes[*cursor];
    *cursor = end;
    Ok(value)
}

fn read_digest(bytes: &[u8], cursor: &mut usize) -> Result<[u8; 32], TraceError> {
    let slice = read_slice(bytes, cursor, 32)?;
    let mut out = [0u8; 32];
    out.copy_from_slice(slice);
    Ok(out)
}

fn read_vec(bytes: &[u8], cursor: &mut usize, len: usize) -> Result<Vec<u8>, TraceError> {
    let slice = read_slice(bytes, cursor, len)?;
    Ok(slice.to_vec())
}

fn read_slice<'a>(bytes: &'a [u8], cursor: &mut usize, len: usize) -> Result<&'a [u8], TraceError> {
    let end = cursor
        .checked_add(len)
        .ok_or_else(|| TraceError::InvalidFormat {
            message: "unexpected eof".to_string(),
        })?;
    if end > bytes.len() {
        return Err(TraceError::InvalidFormat {
            message: "unexpected eof".to_string(),
        });
    }
    let slice = &bytes[*cursor..end];
    *cursor = end;
    Ok(slice)
}

#[cfg(feature = "biophys-trace")]
pub mod sn_l4;
