#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Default)]
pub struct HpaConfig {
    pub endpoint: String,
    pub calibration_table: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct HpaSnapshot {
    pub timestamp: u64,
    pub payload: Vec<u8>,
}

#[derive(Debug, Error)]
pub enum HpaError {
    #[error("HPA interaction is not implemented")]
    NotImplemented,
}

pub trait HpaClient {
    fn configure(&mut self, _config: HpaConfig) -> Result<(), HpaError> {
        Err(HpaError::NotImplemented)
    }

    fn measure(&mut self) -> Result<HpaSnapshot, HpaError> {
        Err(HpaError::NotImplemented)
    }
}

#[derive(Debug, Default)]
pub struct PlaceholderHpa;

impl HpaClient for PlaceholderHpa {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hpa_is_a_placeholder() {
        let mut client = PlaceholderHpa;
        let result = client.measure();
        assert!(matches!(result, Err(HpaError::NotImplemented)));
    }
}
