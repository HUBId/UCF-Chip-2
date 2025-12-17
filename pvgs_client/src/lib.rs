#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Default)]
pub struct PvgsClientConfig {
    pub cbv_endpoint: String,
    pub hbv_endpoint: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct PvgsSnapshot {
    pub channel: String,
    pub payload: Vec<u8>,
}

#[derive(Debug, Error)]
pub enum PvgsError {
    #[error("PVGS fetch is not implemented")]
    NotImplemented,
}

pub trait PvgsProvider {
    fn configure(&mut self, _config: PvgsClientConfig) -> Result<(), PvgsError> {
        Err(PvgsError::NotImplemented)
    }

    fn fetch_cbv(&self) -> Result<PvgsSnapshot, PvgsError> {
        Err(PvgsError::NotImplemented)
    }

    fn fetch_hbv(&self) -> Result<PvgsSnapshot, PvgsError> {
        Err(PvgsError::NotImplemented)
    }
}

#[derive(Debug, Default)]
pub struct PlaceholderPvgsClient;

impl PvgsProvider for PlaceholderPvgsClient {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pvgs_placeholder_rejects_fetches() {
        let client = PlaceholderPvgsClient;
        assert!(matches!(client.fetch_cbv(), Err(PvgsError::NotImplemented)));
        assert!(matches!(client.fetch_hbv(), Err(PvgsError::NotImplemented)));
    }
}
