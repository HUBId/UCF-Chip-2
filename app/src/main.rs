#![forbid(unsafe_code)]

use engine::EngineInputs;
use hpa::{HpaClient, HpaConfig, PlaceholderHpa};
use profiles::{PlaceholderComposer, ProfileComposer, ProfileResolutionRequest};
use pvgs_client::{PlaceholderPvgsClient, PvgsClientConfig, PvgsProvider};
use rsv::{RegulatorState, StateError, StateStore};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
struct ConfigPaths {
    regulator_profiles: String,
    regulator_overlays: String,
    regulator_update_tables: String,
    windowing: String,
    class_thresholds: String,
    hpa: String,
}

impl Default for ConfigPaths {
    fn default() -> Self {
        Self {
            regulator_profiles: "config/regulator_profiles.yaml".to_string(),
            regulator_overlays: "config/regulator_overlays.yaml".to_string(),
            regulator_update_tables: "config/regulator_update_tables.yaml".to_string(),
            windowing: "config/windowing.yaml".to_string(),
            class_thresholds: "config/class_thresholds.yaml".to_string(),
            hpa: "config/hpa.yaml".to_string(),
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Error)]
enum AppError {
    #[error("configuration path missing for {0}")]
    MissingPath(&'static str),
    #[error("configuration serialization failed: {0}")]
    ConfigSerde(#[from] serde_yaml::Error),
    #[error("state store placeholder error: {0}")]
    State(#[from] StateError),
}

impl From<serde_yaml::Error> for AppError {
    fn from(error: serde_yaml::Error) -> Self {
        AppError::ConfigSerde(error)
    }
}

impl From<StateError> for AppError {
    fn from(error: StateError) -> Self {
        AppError::State(error)
    }
}

struct MemoryStore {
    state: RegulatorState,
}

impl MemoryStore {
    fn new(state: RegulatorState) -> Self {
        Self { state }
    }
}

impl StateStore for MemoryStore {
    fn load(&self) -> Result<RegulatorState, StateError> {
        Ok(self.state.clone())
    }

    fn persist(&self, _state: &RegulatorState) -> Result<(), StateError> {
        Ok(())
    }
}

fn validate_paths(paths: &ConfigPaths) -> Result<(), AppError> {
    if paths.regulator_profiles.is_empty() {
        return Err(AppError::MissingPath("regulator_profiles"));
    }
    if paths.regulator_overlays.is_empty() {
        return Err(AppError::MissingPath("regulator_overlays"));
    }
    if paths.regulator_update_tables.is_empty() {
        return Err(AppError::MissingPath("regulator_update_tables"));
    }
    if paths.windowing.is_empty() {
        return Err(AppError::MissingPath("windowing"));
    }
    if paths.class_thresholds.is_empty() {
        return Err(AppError::MissingPath("class_thresholds"));
    }
    if paths.hpa.is_empty() {
        return Err(AppError::MissingPath("hpa"));
    }

    Ok(())
}

fn announce_placeholders() {
    let composer = PlaceholderComposer;
    let _ = composer.compose(ProfileResolutionRequest {
        profile: "baseline".to_string(),
        overlays: vec![],
    });

    let _wire_inputs = EngineInputs {
        tick: 0,
        inbound: Vec::new(),
    };

    let mut hpa = PlaceholderHpa;
    let _ = hpa.configure(HpaConfig::default());

    let mut pvgs = PlaceholderPvgsClient;
    let _ = pvgs.configure(PvgsClientConfig::default());
}

fn main() -> Result<(), AppError> {
    let paths = ConfigPaths::default();
    validate_paths(&paths)?;

    let _ = serde_yaml::to_string(&paths)?;

    let state_store = MemoryStore::new(RegulatorState::default());
    let _ = state_store.load()?;

    announce_placeholders();

    println!("boot ok: loaded config paths {:?}", paths);
    Ok(())
}
