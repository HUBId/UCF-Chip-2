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

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum PvgsError {
    #[error("PVGS fetch is not implemented")]
    NotImplemented,
    #[error("PVGS commit failed: {reason}")]
    CommitFailed { reason: String },
}

pub trait PvgsReader: Send + Sync {
    fn get_latest_cbv_digest(&self) -> Option<[u8; 32]>;

    fn get_latest_pev_digest(&self) -> Option<[u8; 32]> {
        None
    }

    fn get_latest_ruleset_digest(&self) -> Option<[u8; 32]> {
        None
    }
}

pub trait PvgsWriter: Send {
    fn commit_control_frame_evidence(
        &mut self,
        session_id: &str,
        control_frame_digest: [u8; 32],
    ) -> Result<(), PvgsError>;
}

#[derive(Debug, Clone, Default)]
pub struct MockPvgsReader {
    pub cbv_digest: Option<[u8; 32]>,
    pub pev_digest: Option<[u8; 32]>,
    pub ruleset_digest: Option<[u8; 32]>,
}

impl MockPvgsReader {
    pub fn with_cbv(cbv_digest: [u8; 32]) -> Self {
        Self {
            cbv_digest: Some(cbv_digest),
            ..Default::default()
        }
    }
}

impl PvgsReader for MockPvgsReader {
    fn get_latest_cbv_digest(&self) -> Option<[u8; 32]> {
        self.cbv_digest
    }

    fn get_latest_pev_digest(&self) -> Option<[u8; 32]> {
        self.pev_digest
    }

    fn get_latest_ruleset_digest(&self) -> Option<[u8; 32]> {
        self.ruleset_digest
    }
}

#[derive(Debug, Default)]
pub struct MockPvgsWriter {
    pub committed: Vec<(String, [u8; 32])>,
    pub fail_with: Option<String>,
}

impl MockPvgsWriter {
    pub fn failing(reason: impl Into<String>) -> Self {
        Self {
            committed: Vec::new(),
            fail_with: Some(reason.into()),
        }
    }
}

impl PvgsWriter for MockPvgsWriter {
    fn commit_control_frame_evidence(
        &mut self,
        session_id: &str,
        control_frame_digest: [u8; 32],
    ) -> Result<(), PvgsError> {
        if let Some(reason) = &self.fail_with {
            return Err(PvgsError::CommitFailed {
                reason: reason.clone(),
            });
        }

        self.committed
            .push((session_id.to_string(), control_frame_digest));
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct PlaceholderPvgsClient;

impl PvgsReader for PlaceholderPvgsClient {
    fn get_latest_cbv_digest(&self) -> Option<[u8; 32]> {
        None
    }
}

impl PvgsWriter for PlaceholderPvgsClient {
    fn commit_control_frame_evidence(
        &mut self,
        _session_id: &str,
        _control_frame_digest: [u8; 32],
    ) -> Result<(), PvgsError> {
        Err(PvgsError::NotImplemented)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_reader_exposes_cbv_digest() {
        let cbv = [1u8; 32];
        let reader = MockPvgsReader::with_cbv(cbv);

        assert_eq!(reader.get_latest_cbv_digest(), Some(cbv));
        assert!(reader.get_latest_pev_digest().is_none());
    }

    #[test]
    fn mock_writer_records_commit() {
        let mut writer = MockPvgsWriter::default();
        let digest = [2u8; 32];
        writer
            .commit_control_frame_evidence("session-1", digest)
            .unwrap();

        assert_eq!(writer.committed.len(), 1);
        assert_eq!(writer.committed[0].0, "session-1");
        assert_eq!(writer.committed[0].1, digest);
    }

    #[test]
    fn placeholder_returns_none_and_rejects_commit() {
        let mut placeholder = PlaceholderPvgsClient;
        assert!(placeholder.get_latest_cbv_digest().is_none());
        assert!(matches!(
            placeholder.commit_control_frame_evidence("s", [0u8; 32]),
            Err(PvgsError::NotImplemented)
        ));
    }
}
