#![cfg(any(test, feature = "local-pvgs"))]

use crate::{PvgsError, PvgsReader, PvgsWriter};
use chip4::pvgs::{
    AssetManifestAppend, AssetManifestCommit, CbvQuery, Digest32, MicrocircuitConfigCommit,
    PevQuery,
};
use ucf::v1::{
    AssetManifest, CharacterBaselineVector, MicrocircuitConfigAppend, MicrocircuitConfigEvidence,
    PolicyEcologyVector, PvgsReceipt,
};

#[derive(Clone)]
pub struct LocalPvgsReader<Q: CbvQuery> {
    query: Q,
}

impl<Q: CbvQuery> LocalPvgsReader<Q> {
    pub fn new(query: Q) -> Self {
        Self { query }
    }

    fn digest_from_proto(proto: Digest32) -> Option<[u8; 32]> {
        let bytes = proto.value;
        if bytes.len() != 32 {
            return None;
        }

        let mut digest = [0u8; 32];
        digest.copy_from_slice(&bytes);
        Some(digest)
    }
}

impl<Q: CbvQuery + PevQuery> PvgsReader for LocalPvgsReader<Q> {
    fn get_latest_cbv_digest(&self) -> Option<[u8; 32]> {
        self.query
            .get_latest_cbv()
            .and_then(|cbv| cbv.cbv_digest)
            .and_then(Self::digest_from_proto)
    }

    fn get_latest_cbv(&self) -> Option<CharacterBaselineVector> {
        self.query.get_latest_cbv().and_then(|cbv| cbv.cbv)
    }

    fn get_latest_pev_digest(&self) -> Option<[u8; 32]> {
        self.query
            .get_latest_pev()
            .and_then(|pev| pev.pev_digest)
            .and_then(Self::digest_from_proto)
    }

    fn get_latest_pev(&self) -> Option<PolicyEcologyVector> {
        self.query.get_latest_pev().and_then(|pev| pev.pev)
    }
}

#[derive(Clone)]
pub struct LocalPvgsWriter<C: MicrocircuitConfigCommit> {
    commit: C,
}

impl<C: MicrocircuitConfigCommit> LocalPvgsWriter<C> {
    pub fn new(commit: C) -> Self {
        Self { commit }
    }
}

impl<C: MicrocircuitConfigCommit> PvgsWriter for LocalPvgsWriter<C> {
    fn commit_control_frame_evidence(
        &mut self,
        _session_id: &str,
        _control_frame_digest: [u8; 32],
    ) -> Result<(), PvgsError> {
        Err(PvgsError::NotImplemented)
    }

    fn commit_microcircuit_config(
        &mut self,
        evidence: MicrocircuitConfigEvidence,
    ) -> Result<PvgsReceipt, PvgsError> {
        Ok(self
            .commit
            .commit_microcircuit_config(MicrocircuitConfigAppend {
                evidence: Some(evidence),
            }))
    }

    fn commit_asset_manifest(
        &mut self,
        _manifest: AssetManifest,
    ) -> Result<PvgsReceipt, PvgsError> {
        Err(PvgsError::NotImplemented)
    }
}

#[derive(Clone)]
pub struct LocalAssetManifestWriter<C: AssetManifestCommit> {
    commit: C,
}

impl<C: AssetManifestCommit> LocalAssetManifestWriter<C> {
    pub fn new(commit: C) -> Self {
        Self { commit }
    }
}

impl<C: AssetManifestCommit> PvgsWriter for LocalAssetManifestWriter<C> {
    fn commit_control_frame_evidence(
        &mut self,
        _session_id: &str,
        _control_frame_digest: [u8; 32],
    ) -> Result<(), PvgsError> {
        Err(PvgsError::NotImplemented)
    }

    fn commit_microcircuit_config(
        &mut self,
        _evidence: MicrocircuitConfigEvidence,
    ) -> Result<PvgsReceipt, PvgsError> {
        Err(PvgsError::NotImplemented)
    }

    fn commit_asset_manifest(&mut self, manifest: AssetManifest) -> Result<PvgsReceipt, PvgsError> {
        Ok(self.commit.commit_asset_manifest(AssetManifestAppend {
            manifest: Some(manifest),
        }))
    }
}
