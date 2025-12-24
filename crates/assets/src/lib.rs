#![forbid(unsafe_code)]

use std::collections::HashMap;
use thiserror::Error;

pub const ASSET_MANIFEST_EVIDENCE_LEN: usize = 172;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AssetManifestEvidence {
    pub manifest_version: u32,
    pub manifest_digest: [u8; 32],
    pub morph_digest: [u8; 32],
    pub channel_params_digest: [u8; 32],
    pub syn_params_digest: [u8; 32],
    pub connectivity_digest: [u8; 32],
    pub created_at_ms: u64,
}

impl AssetManifestEvidence {
    pub fn validate(&self) -> Result<(), AssetManifestError> {
        if self.manifest_version == 0 {
            return Err(AssetManifestError::InvalidVersion);
        }
        Ok(())
    }

    pub fn from_fixed_bytes(payload: &[u8]) -> Result<Self, AssetManifestError> {
        if payload.len() != ASSET_MANIFEST_EVIDENCE_LEN {
            return Err(AssetManifestError::InvalidLength {
                expected: ASSET_MANIFEST_EVIDENCE_LEN,
                actual: payload.len(),
            });
        }

        let manifest_version = u32::from_le_bytes(payload[0..4].try_into().unwrap());
        let created_at_ms = u64::from_le_bytes(payload[4..12].try_into().unwrap());
        let mut offset = 12;

        let manifest_digest = read_digest(payload, &mut offset);
        let morph_digest = read_digest(payload, &mut offset);
        let channel_params_digest = read_digest(payload, &mut offset);
        let syn_params_digest = read_digest(payload, &mut offset);
        let connectivity_digest = read_digest(payload, &mut offset);

        let evidence = Self {
            manifest_version,
            manifest_digest,
            morph_digest,
            channel_params_digest,
            syn_params_digest,
            connectivity_digest,
            created_at_ms,
        };
        evidence.validate()?;
        Ok(evidence)
    }
}

fn read_digest(payload: &[u8], offset: &mut usize) -> [u8; 32] {
    let bytes: [u8; 32] = payload[*offset..*offset + 32].try_into().unwrap();
    *offset += 32;
    bytes
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AssetManifestError {
    #[error("asset manifest evidence has invalid version")]
    InvalidVersion,
    #[error("asset manifest evidence payload length mismatch: expected {expected}, got {actual}")]
    InvalidLength { expected: usize, actual: usize },
}

#[derive(Debug, Default, Clone)]
pub struct AssetStore {
    manifests: Vec<AssetManifestEvidence>,
    index: HashMap<[u8; 32], usize>,
}

impl AssetStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn latest(&self) -> Option<AssetManifestEvidence> {
        self.manifests.iter().copied().max_by(compare_manifest)
    }

    pub fn get(&self, manifest_digest: [u8; 32]) -> Option<AssetManifestEvidence> {
        self.index
            .get(&manifest_digest)
            .and_then(|&idx| self.manifests.get(idx).copied())
    }

    pub fn insert(&mut self, evidence: AssetManifestEvidence) -> bool {
        if self.index.contains_key(&evidence.manifest_digest) {
            return false;
        }
        let idx = self.manifests.len();
        self.manifests.push(evidence);
        self.index.insert(evidence.manifest_digest, idx);
        true
    }

    pub fn manifests(&self) -> &[AssetManifestEvidence] {
        &self.manifests
    }
}

fn compare_manifest(a: &AssetManifestEvidence, b: &AssetManifestEvidence) -> std::cmp::Ordering {
    a.created_at_ms
        .cmp(&b.created_at_ms)
        .then_with(|| a.manifest_digest.cmp(&b.manifest_digest))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_fixed_bytes() {
        let mut payload = Vec::with_capacity(ASSET_MANIFEST_EVIDENCE_LEN);
        payload.extend_from_slice(&1u32.to_le_bytes());
        payload.extend_from_slice(&42u64.to_le_bytes());
        payload.extend_from_slice(&[1u8; 32]);
        payload.extend_from_slice(&[2u8; 32]);
        payload.extend_from_slice(&[3u8; 32]);
        payload.extend_from_slice(&[4u8; 32]);
        payload.extend_from_slice(&[5u8; 32]);

        let evidence = AssetManifestEvidence::from_fixed_bytes(&payload).unwrap();
        assert_eq!(evidence.manifest_version, 1);
        assert_eq!(evidence.created_at_ms, 42);
        assert_eq!(evidence.connectivity_digest, [5u8; 32]);
    }

    #[test]
    fn rejects_zero_version() {
        let evidence = AssetManifestEvidence {
            manifest_version: 0,
            manifest_digest: [0u8; 32],
            morph_digest: [0u8; 32],
            channel_params_digest: [0u8; 32],
            syn_params_digest: [0u8; 32],
            connectivity_digest: [0u8; 32],
            created_at_ms: 0,
        };
        assert_eq!(evidence.validate(), Err(AssetManifestError::InvalidVersion));
    }
}
