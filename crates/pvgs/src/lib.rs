#![forbid(unsafe_code)]

use assets::{AssetManifestEvidence, AssetManifestError, AssetStore};
use blake3::Hasher;
use thiserror::Error;
use ucf::v1::{PvgsReceipt, ReasonCode};

pub const ASSET_MANIFEST_DOMAIN: &str = "UCF:ASSET:MANIFEST";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssetManifestAppend {
    pub payload: Vec<u8>,
    pub payload_digest: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofReceipt {
    pub receipt: Vec<u8>,
    pub ruleset_digest: Option<[u8; 32]>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CommitResult {
    pub pvgs_receipt: PvgsReceipt,
    pub proof_receipt: ProofReceipt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SepEventType {
    Milestone,
    RecoveryGov,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SepEvent {
    pub event_type: SepEventType,
    pub reason_code: ReasonCode,
    pub object_digest: [u8; 32],
}

#[derive(Debug, Default, Clone)]
pub struct PvgsStore {
    asset_store: AssetStore,
    sep_events: Vec<SepEvent>,
    ruleset_digest: Option<[u8; 32]>,
}

impl PvgsStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_ruleset_digest(&mut self, digest: [u8; 32]) {
        self.ruleset_digest = Some(digest);
    }

    pub fn asset_store(&self) -> &AssetStore {
        &self.asset_store
    }

    pub fn sep_events(&self) -> &[SepEvent] {
        &self.sep_events
    }

    pub fn commit_asset_manifest(
        &mut self,
        append: AssetManifestAppend,
    ) -> Result<CommitResult, PvgsError> {
        let evidence = AssetManifestEvidence::from_fixed_bytes(&append.payload)
            .map_err(PvgsError::AssetManifest)?;
        let computed_digest = compute_manifest_digest(&append.payload);

        if computed_digest != evidence.manifest_digest || computed_digest != append.payload_digest {
            return Err(PvgsError::DigestMismatch);
        }

        if self.asset_store.insert(evidence) {
            self.sep_events.push(SepEvent {
                event_type: SepEventType::Milestone,
                reason_code: ReasonCode::RcGvAssetManifestAppended,
                object_digest: evidence.manifest_digest,
            });
        }

        Ok(CommitResult {
            pvgs_receipt: PvgsReceipt { receipt: Vec::new() },
            proof_receipt: ProofReceipt {
                receipt: Vec::new(),
                ruleset_digest: self.ruleset_digest,
            },
        })
    }
}

pub fn compute_manifest_digest(payload: &[u8]) -> [u8; 32] {
    const MANIFEST_DIGEST_OFFSET: usize = 12;
    const DIGEST_LEN: usize = 32;
    let mut hasher = Hasher::new();
    hasher.update(ASSET_MANIFEST_DOMAIN.as_bytes());
    if payload.len() >= MANIFEST_DIGEST_OFFSET + DIGEST_LEN {
        let mut normalized = payload.to_vec();
        normalized[MANIFEST_DIGEST_OFFSET..MANIFEST_DIGEST_OFFSET + DIGEST_LEN].fill(0);
        hasher.update(&normalized);
    } else {
        hasher.update(payload);
    }
    let hash = hasher.finalize();
    *hash.as_bytes()
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum PvgsError {
    #[error("asset manifest decode failed: {0}")]
    AssetManifest(AssetManifestError),
    #[error("asset manifest digest mismatch")]
    DigestMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;
    use assets::ASSET_MANIFEST_EVIDENCE_LEN;

    fn build_payload(created_at_ms: u64) -> Vec<u8> {
        let mut payload = Vec::with_capacity(ASSET_MANIFEST_EVIDENCE_LEN);
        payload.extend_from_slice(&1u32.to_le_bytes());
        payload.extend_from_slice(&created_at_ms.to_le_bytes());
        payload.extend_from_slice(&[0u8; 32]);
        payload.extend_from_slice(&[2u8; 32]);
        payload.extend_from_slice(&[3u8; 32]);
        payload.extend_from_slice(&[4u8; 32]);
        payload.extend_from_slice(&[5u8; 32]);
        let digest = compute_manifest_digest(&payload);
        payload[12..44].copy_from_slice(&digest);
        payload
    }

    #[test]
    fn appends_asset_manifest() {
        let payload = build_payload(100);
        let payload_digest = compute_manifest_digest(&payload);
        let mut store = PvgsStore::new();

        let result = store
            .commit_asset_manifest(AssetManifestAppend {
                payload,
                payload_digest,
            })
            .unwrap();

        assert_eq!(result.proof_receipt.ruleset_digest, None);
        assert_eq!(store.asset_store().manifests().len(), 1);
        assert_eq!(
            store.asset_store().latest().unwrap().manifest_digest,
            payload_digest
        );
    }

    #[test]
    fn dedup_is_idempotent() {
        let payload = build_payload(100);
        let payload_digest = compute_manifest_digest(&payload);
        let mut store = PvgsStore::new();

        store
            .commit_asset_manifest(AssetManifestAppend {
                payload: payload.clone(),
                payload_digest,
            })
            .unwrap();
        store
            .commit_asset_manifest(AssetManifestAppend {
                payload,
                payload_digest,
            })
            .unwrap();

        assert_eq!(store.asset_store().manifests().len(), 1);
    }

    #[test]
    fn logs_sep_event() {
        let payload = build_payload(100);
        let payload_digest = compute_manifest_digest(&payload);
        let mut store = PvgsStore::new();

        store
            .commit_asset_manifest(AssetManifestAppend {
                payload,
                payload_digest,
            })
            .unwrap();

        assert_eq!(store.sep_events().len(), 1);
        let event = &store.sep_events()[0];
        assert_eq!(event.reason_code, ReasonCode::RcGvAssetManifestAppended);
        assert_eq!(event.object_digest, payload_digest);
    }

    #[test]
    fn rejects_invalid_length() {
        let payload = vec![0u8; 10];
        let payload_digest = compute_manifest_digest(&payload);
        let mut store = PvgsStore::new();

        let err = store
            .commit_asset_manifest(AssetManifestAppend {
                payload,
                payload_digest,
            })
            .unwrap_err();

        assert!(matches!(err, PvgsError::AssetManifest(_)));
    }

    #[test]
    fn rejects_digest_mismatch() {
        let payload = build_payload(100);
        let mut store = PvgsStore::new();

        let err = store
            .commit_asset_manifest(AssetManifestAppend {
                payload,
                payload_digest: [2u8; 32],
            })
            .unwrap_err();

        assert_eq!(err, PvgsError::DigestMismatch);
    }
}
