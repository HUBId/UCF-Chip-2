#![forbid(unsafe_code)]

#[cfg(any(test, feature = "local-pvgs"))]
pub mod local;
#[cfg(any(test, feature = "local-pvgs"))]
pub use local::LocalPvgsReader;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use ucf::v1::CharacterBaselineVector;

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

    fn get_latest_cbv(&self) -> Option<CharacterBaselineVector> {
        None
    }

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
    pub cbv: Option<CharacterBaselineVector>,
}

impl MockPvgsReader {
    pub fn with_cbv(cbv_digest: [u8; 32]) -> Self {
        Self::with_cbv_digest(cbv_digest)
    }

    pub fn with_cbv_digest(cbv_digest: [u8; 32]) -> Self {
        Self {
            cbv_digest: Some(cbv_digest),
            ..Default::default()
        }
    }

    pub fn with_cbv_vector(cbv: CharacterBaselineVector) -> Self {
        Self {
            cbv: Some(cbv),
            ..Default::default()
        }
    }
}

impl PvgsReader for MockPvgsReader {
    fn get_latest_cbv_digest(&self) -> Option<[u8; 32]> {
        self.cbv_digest
    }

    fn get_latest_cbv(&self) -> Option<CharacterBaselineVector> {
        self.cbv.clone()
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
    use chip4::pvgs::{Cbv, Digest32, InMemoryPvgs};

    #[test]
    fn mock_reader_exposes_cbv_digest() {
        let cbv = [1u8; 32];
        let reader = MockPvgsReader::with_cbv(cbv);

        assert_eq!(reader.get_latest_cbv_digest(), Some(cbv));
        assert!(reader.get_latest_pev_digest().is_none());
    }

    #[test]
    fn mock_reader_exposes_cbv_vector() {
        let cbv = CharacterBaselineVector {
            baseline_caution_offset: 1,
            baseline_novelty_dampening_offset: 0,
            baseline_approval_strictness_offset: 0,
            baseline_export_strictness_offset: 0,
            baseline_chain_conservatism_offset: 0,
            baseline_cooldown_multiplier_class: 0,
        };
        let reader = MockPvgsReader::with_cbv_vector(cbv.clone());

        assert_eq!(reader.get_latest_cbv(), Some(cbv));
        assert!(reader.get_latest_cbv_digest().is_none());
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

    #[test]
    fn local_reader_returns_none_without_cbv() {
        let store = InMemoryPvgs::new();
        let reader = LocalPvgsReader::new(store);

        assert!(reader.get_latest_cbv_digest().is_none());
    }

    #[test]
    fn local_reader_returns_latest_cbv_digest() {
        let store = InMemoryPvgs::new();
        let digest = [9u8; 32];
        let cbv = Cbv {
            epoch: 1,
            cbv_digest: Some(Digest32::from_array(digest)),
            proof_receipt_ref: Some(vec![1, 2, 3]),
            signature: Some(vec![4, 5, 6]),
            cbv: None,
        };
        store.commit_cbv_update(cbv);

        let reader = LocalPvgsReader::new(store);
        assert_eq!(reader.get_latest_cbv_digest(), Some(digest));
    }

    #[test]
    fn local_reader_rejects_short_digest() {
        let store = InMemoryPvgs::new();
        let cbv = Cbv {
            epoch: 1,
            cbv_digest: Some(Digest32 {
                value: vec![1, 2, 3],
            }),
            proof_receipt_ref: None,
            signature: None,
            cbv: None,
        };
        store.commit_cbv_update(cbv);

        let reader = LocalPvgsReader::new(store);
        assert!(reader.get_latest_cbv_digest().is_none());
    }

    #[test]
    fn local_reader_exposes_full_cbv_when_available() {
        let store = InMemoryPvgs::new();
        let cbv = CharacterBaselineVector {
            baseline_caution_offset: 2,
            baseline_novelty_dampening_offset: 2,
            baseline_approval_strictness_offset: 1,
            baseline_export_strictness_offset: 1,
            baseline_chain_conservatism_offset: 2,
            baseline_cooldown_multiplier_class: 2,
        };

        let stored = Cbv {
            epoch: 1,
            cbv_digest: Some(Digest32::from_array([1u8; 32])),
            proof_receipt_ref: None,
            signature: None,
            cbv: Some(cbv.clone()),
        };
        store.commit_cbv_update(stored);

        let reader = LocalPvgsReader::new(store);
        assert_eq!(reader.get_latest_cbv(), Some(cbv));
    }
}
