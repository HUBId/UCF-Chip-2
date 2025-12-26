#![forbid(unsafe_code)]

#[cfg(any(test, feature = "local-pvgs"))]
pub mod local;
#[cfg(any(test, feature = "local-pvgs"))]
pub use local::{LocalAssetBundleWriter, LocalPvgsReader, LocalPvgsWriter};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::collections::HashSet;
use thiserror::Error;
use ucf::v1::{
    AssetBundle, AssetManifest, CharacterBaselineVector, MicrocircuitConfigEvidence,
    PolicyEcologyVector, PvgsReceipt, ReplayRunEvidence,
};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TraceRunStatus {
    Pass = 1,
    Fail = 2,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceRunEvidenceLike {
    pub trace_id: String,
    pub trace_run_digest: [u8; 32],
    pub asset_manifest_digest: [u8; 32],
    pub circuit_config_digest: [u8; 32],
    pub steps: u32,
    pub status: TraceRunStatus,
    pub created_at_ms: u64,
    pub reason_codes: Vec<String>,
}

const MAX_TRACE_ID_BYTES: usize = 256;
const MAX_REASON_CODES: usize = 16;
const MAX_REASON_CODE_BYTES: usize = 128;

pub fn encode_trace_run_evidence(ev: &TraceRunEvidenceLike) -> Vec<u8> {
    let mut out = Vec::new();
    let trace_id_bytes = bounded_bytes(ev.trace_id.as_bytes(), MAX_TRACE_ID_BYTES);
    out.extend_from_slice(&(trace_id_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(trace_id_bytes);
    out.extend_from_slice(&ev.trace_run_digest);
    out.extend_from_slice(&ev.asset_manifest_digest);
    out.extend_from_slice(&ev.circuit_config_digest);
    out.extend_from_slice(&ev.steps.to_le_bytes());
    out.push(ev.status as u8);
    out.extend_from_slice(&ev.created_at_ms.to_le_bytes());

    let reason_codes: Vec<&String> = ev.reason_codes.iter().take(MAX_REASON_CODES).collect();
    out.extend_from_slice(&(reason_codes.len() as u32).to_le_bytes());
    for code in reason_codes {
        let code_bytes = bounded_bytes(code.as_bytes(), MAX_REASON_CODE_BYTES);
        out.extend_from_slice(&(code_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(code_bytes);
    }

    out
}

fn bounded_bytes(bytes: &[u8], max_len: usize) -> &[u8] {
    if bytes.len() <= max_len {
        bytes
    } else {
        &bytes[..max_len]
    }
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

    fn get_latest_pev(&self) -> Option<PolicyEcologyVector> {
        None
    }

    fn get_latest_ruleset_digest(&self) -> Option<[u8; 32]> {
        None
    }

    fn get_latest_asset_bundle(&mut self) -> Result<Option<AssetBundle>, PvgsError> {
        Ok(None)
    }

    fn get_asset_bundle(&mut self, _digest: [u8; 32]) -> Result<Option<AssetBundle>, PvgsError> {
        Ok(None)
    }

    fn list_asset_bundles(&mut self) -> Result<Vec<AssetBundle>, PvgsError> {
        Ok(Vec::new())
    }

    fn find_asset_bundle_by_manifest_digest(
        &mut self,
        manifest_digest: [u8; 32],
    ) -> Result<Option<AssetBundle>, PvgsError> {
        let mut bundles = self.list_asset_bundles()?;
        if bundles.is_empty() {
            return Ok(self
                .get_latest_asset_bundle()?
                .filter(|bundle| manifest_matches(bundle, &manifest_digest)));
        }

        bundles.retain(|bundle| manifest_matches(bundle, &manifest_digest));
        bundles.sort_by(|a, b| {
            a.bundle_digest
                .cmp(&b.bundle_digest)
                .then_with(|| a.bundle_id.cmp(&b.bundle_id))
        });
        Ok(bundles.into_iter().next())
    }
}

fn manifest_matches(bundle: &AssetBundle, digest: &[u8; 32]) -> bool {
    bundle
        .manifest
        .as_ref()
        .map(|manifest| manifest.manifest_digest.as_slice() == digest)
        .unwrap_or(false)
}

pub trait PvgsWriter: Send {
    fn commit_control_frame_evidence(
        &mut self,
        session_id: &str,
        control_frame_digest: [u8; 32],
    ) -> Result<(), PvgsError>;

    fn commit_microcircuit_config(
        &mut self,
        evidence: MicrocircuitConfigEvidence,
    ) -> Result<PvgsReceipt, PvgsError>;

    fn commit_asset_manifest(&mut self, manifest: AssetManifest) -> Result<PvgsReceipt, PvgsError>;

    fn commit_asset_bundle(&mut self, bundle: AssetBundle) -> Result<PvgsReceipt, PvgsError>;

    fn commit_replay_run_evidence(
        &mut self,
        evidence: ReplayRunEvidence,
    ) -> Result<PvgsReceipt, PvgsError>;

    fn commit_trace_run_evidence(
        &mut self,
        evidence: TraceRunEvidenceLike,
    ) -> Result<PvgsReceipt, PvgsError>;
}

#[derive(Debug, Clone, Default)]
pub struct MockPvgsReader {
    pub cbv_digest: Option<[u8; 32]>,
    pub pev_digest: Option<[u8; 32]>,
    pub ruleset_digest: Option<[u8; 32]>,
    pub cbv: Option<CharacterBaselineVector>,
    pub pev: Option<PolicyEcologyVector>,
    pub asset_bundles: Vec<AssetBundle>,
    pub asset_bundle_index: HashMap<[u8; 32], AssetBundle>,
    pub latest_asset_bundle: Option<AssetBundle>,
}

impl MockPvgsReader {
    fn deterministic_bundle() -> AssetBundle {
        AssetBundle {
            bundle_id: "bundle:mock".to_string(),
            created_at_ms: 0,
            bundle_digest: vec![0u8; 32],
            manifest: None,
            chunks: Vec::new(),
        }
    }

    pub fn with_asset_bundle(bundle: AssetBundle) -> Self {
        let mut index = HashMap::new();
        if bundle.bundle_digest.len() == 32 {
            let mut digest = [0u8; 32];
            digest.copy_from_slice(&bundle.bundle_digest);
            index.insert(digest, bundle.clone());
        }
        let asset_bundles = vec![bundle.clone()];
        Self {
            latest_asset_bundle: Some(bundle),
            asset_bundles,
            asset_bundle_index: index,
            ..Default::default()
        }
    }

    pub fn push_asset_bundle(&mut self, bundle: AssetBundle) {
        if bundle.bundle_digest.len() == 32 {
            let mut digest = [0u8; 32];
            digest.copy_from_slice(&bundle.bundle_digest);
            self.asset_bundle_index.insert(digest, bundle.clone());
        }
        self.latest_asset_bundle = Some(bundle.clone());
        self.asset_bundles.push(bundle);
    }

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

    pub fn with_pev_vector(pev: PolicyEcologyVector) -> Self {
        Self {
            pev: Some(pev),
            ..Default::default()
        }
    }

    pub fn with_pev_digest(pev_digest: [u8; 32]) -> Self {
        Self {
            pev_digest: Some(pev_digest),
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

    fn get_latest_pev(&self) -> Option<PolicyEcologyVector> {
        self.pev.clone()
    }

    fn get_latest_ruleset_digest(&self) -> Option<[u8; 32]> {
        self.ruleset_digest
    }

    fn get_latest_asset_bundle(&mut self) -> Result<Option<AssetBundle>, PvgsError> {
        let bundle = self
            .latest_asset_bundle
            .clone()
            .or_else(|| self.asset_bundles.last().cloned())
            .unwrap_or_else(Self::deterministic_bundle);
        Ok(Some(bundle))
    }

    fn get_asset_bundle(&mut self, digest: [u8; 32]) -> Result<Option<AssetBundle>, PvgsError> {
        if let Some(bundle) = self.asset_bundle_index.get(&digest).cloned() {
            return Ok(Some(bundle));
        }
        if digest == [0u8; 32] {
            return Ok(Some(Self::deterministic_bundle()));
        }
        Ok(None)
    }

    fn list_asset_bundles(&mut self) -> Result<Vec<AssetBundle>, PvgsError> {
        Ok(self.asset_bundles.clone())
    }
}

#[derive(Debug, Default)]
pub struct MockPvgsWriter {
    pub committed: Vec<(String, [u8; 32])>,
    pub committed_microcircuit_configs: Vec<MicrocircuitConfigEvidence>,
    pub committed_asset_manifests: Vec<AssetManifest>,
    pub committed_asset_bundles: Vec<AssetBundle>,
    pub committed_replay_runs: Vec<ReplayRunEvidence>,
    pub committed_replay_run_digests: HashSet<[u8; 32]>,
    pub committed_trace_runs: Vec<TraceRunEvidenceLike>,
    pub committed_trace_run_keys: HashSet<(String, [u8; 32])>,
    pub fail_with: Option<String>,
}

impl MockPvgsWriter {
    pub fn failing(reason: impl Into<String>) -> Self {
        Self {
            committed: Vec::new(),
            committed_microcircuit_configs: Vec::new(),
            committed_asset_manifests: Vec::new(),
            committed_asset_bundles: Vec::new(),
            committed_replay_runs: Vec::new(),
            committed_replay_run_digests: HashSet::new(),
            committed_trace_runs: Vec::new(),
            committed_trace_run_keys: HashSet::new(),
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

    fn commit_microcircuit_config(
        &mut self,
        evidence: MicrocircuitConfigEvidence,
    ) -> Result<PvgsReceipt, PvgsError> {
        if let Some(reason) = &self.fail_with {
            return Err(PvgsError::CommitFailed {
                reason: reason.clone(),
            });
        }

        self.committed_microcircuit_configs.push(evidence);
        Ok(PvgsReceipt::default())
    }

    fn commit_asset_manifest(&mut self, manifest: AssetManifest) -> Result<PvgsReceipt, PvgsError> {
        if let Some(reason) = &self.fail_with {
            return Err(PvgsError::CommitFailed {
                reason: reason.clone(),
            });
        }

        self.committed_asset_manifests.push(manifest);
        Ok(PvgsReceipt::default())
    }

    fn commit_asset_bundle(&mut self, bundle: AssetBundle) -> Result<PvgsReceipt, PvgsError> {
        if let Some(reason) = &self.fail_with {
            return Err(PvgsError::CommitFailed {
                reason: reason.clone(),
            });
        }

        self.committed_asset_bundles.push(bundle);
        Ok(PvgsReceipt::default())
    }

    fn commit_replay_run_evidence(
        &mut self,
        evidence: ReplayRunEvidence,
    ) -> Result<PvgsReceipt, PvgsError> {
        if let Some(reason) = &self.fail_with {
            return Err(PvgsError::CommitFailed {
                reason: reason.clone(),
            });
        }

        if let Some(digest) = replay_run_digest(&evidence) {
            if !self.committed_replay_run_digests.insert(digest) {
                return Ok(PvgsReceipt::default());
            }
        }

        self.committed_replay_runs.push(evidence);
        Ok(PvgsReceipt::default())
    }

    fn commit_trace_run_evidence(
        &mut self,
        evidence: TraceRunEvidenceLike,
    ) -> Result<PvgsReceipt, PvgsError> {
        if let Some(reason) = &self.fail_with {
            return Err(PvgsError::CommitFailed {
                reason: reason.clone(),
            });
        }

        let key = (evidence.trace_id.clone(), evidence.trace_run_digest);
        if !self.committed_trace_run_keys.insert(key) {
            return Ok(PvgsReceipt::default());
        }

        self.committed_trace_runs.push(evidence);
        Ok(PvgsReceipt::default())
    }
}

#[derive(Debug, Default)]
pub struct PlaceholderPvgsClient;

impl PvgsReader for PlaceholderPvgsClient {
    fn get_latest_cbv_digest(&self) -> Option<[u8; 32]> {
        None
    }

    fn get_latest_asset_bundle(&mut self) -> Result<Option<AssetBundle>, PvgsError> {
        Err(PvgsError::NotImplemented)
    }

    fn get_asset_bundle(&mut self, _digest: [u8; 32]) -> Result<Option<AssetBundle>, PvgsError> {
        Err(PvgsError::NotImplemented)
    }

    fn list_asset_bundles(&mut self) -> Result<Vec<AssetBundle>, PvgsError> {
        Err(PvgsError::NotImplemented)
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

    fn commit_microcircuit_config(
        &mut self,
        _evidence: MicrocircuitConfigEvidence,
    ) -> Result<PvgsReceipt, PvgsError> {
        Err(PvgsError::NotImplemented)
    }

    fn commit_asset_manifest(
        &mut self,
        _manifest: AssetManifest,
    ) -> Result<PvgsReceipt, PvgsError> {
        Err(PvgsError::NotImplemented)
    }

    fn commit_asset_bundle(&mut self, _bundle: AssetBundle) -> Result<PvgsReceipt, PvgsError> {
        Err(PvgsError::NotImplemented)
    }

    fn commit_replay_run_evidence(
        &mut self,
        _evidence: ReplayRunEvidence,
    ) -> Result<PvgsReceipt, PvgsError> {
        Err(PvgsError::NotImplemented)
    }

    fn commit_trace_run_evidence(
        &mut self,
        _evidence: TraceRunEvidenceLike,
    ) -> Result<PvgsReceipt, PvgsError> {
        Err(PvgsError::NotImplemented)
    }
}

fn replay_run_digest(evidence: &ReplayRunEvidence) -> Option<[u8; 32]> {
    let digest = evidence.run_digest.as_ref()?.value.as_slice();
    if digest.len() != 32 {
        return None;
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(digest);
    Some(out)
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
    fn mock_writer_records_microcircuit_commit() {
        let mut writer = MockPvgsWriter::default();
        let evidence = ucf::v1::MicrocircuitConfigEvidence {
            module: ucf::v1::MicrocircuitModule::Lc as i32,
            config_version: 1,
            config_digest: vec![1; 32],
            created_at_ms: 10,
            prev_config_digest: None,
        };

        writer.commit_microcircuit_config(evidence.clone()).unwrap();

        assert_eq!(writer.committed_microcircuit_configs.len(), 1);
        assert_eq!(writer.committed_microcircuit_configs[0], evidence);
    }

    #[test]
    fn mock_writer_records_asset_manifest_commit() {
        let mut writer = MockPvgsWriter::default();
        let manifest = ucf::v1::AssetManifest {
            manifest_version: 1,
            created_at_ms: 10,
            manifest_digest: vec![1; 32],
            components: Vec::new(),
        };

        writer.commit_asset_manifest(manifest.clone()).unwrap();

        assert_eq!(writer.committed_asset_manifests.len(), 1);
        assert_eq!(writer.committed_asset_manifests[0], manifest);
    }

    #[test]
    fn mock_writer_records_asset_bundle_commit() {
        let mut writer = MockPvgsWriter::default();
        let bundle = ucf::v1::AssetBundle {
            bundle_id: "bundle:test".to_string(),
            created_at_ms: 10,
            bundle_digest: vec![2; 32],
            manifest: None,
            chunks: Vec::new(),
        };

        writer.commit_asset_bundle(bundle.clone()).unwrap();

        assert_eq!(writer.committed_asset_bundles.len(), 1);
        assert_eq!(writer.committed_asset_bundles[0], bundle);
    }

    #[test]
    fn mock_writer_commits_replay_run_once() {
        let mut writer = MockPvgsWriter::default();
        let run_digest = [8u8; 32];
        let evidence = ucf::v1::ReplayRunEvidence {
            run_id: "replay_run:session:replay:deadbeef".to_string(),
            run_digest: Some(ucf::v1::Digest32 {
                value: run_digest.to_vec(),
            }),
            replay_plan_ref: None,
            asset_manifest_ref: None,
            micro_configs: Vec::new(),
            steps: 0,
            dt_us: 0,
            substeps_per_tick: 0,
            summary_profile_seq_digest: None,
            summary_dwm_seq_digest: None,
            created_at_ms: 0,
            proof_receipt_ref: None,
            signatures: Vec::new(),
        };

        writer.commit_replay_run_evidence(evidence.clone()).unwrap();
        writer.commit_replay_run_evidence(evidence).unwrap();

        assert_eq!(writer.committed_replay_runs.len(), 1);
    }

    #[test]
    fn trace_run_evidence_encoding_is_deterministic() {
        let ev = TraceRunEvidenceLike {
            trace_id: "trace:unit-test".to_string(),
            trace_run_digest: [1u8; 32],
            asset_manifest_digest: [2u8; 32],
            circuit_config_digest: [3u8; 32],
            steps: 42,
            status: TraceRunStatus::Pass,
            created_at_ms: 1234,
            reason_codes: vec!["RC.GV.TRACE.PASS".to_string()],
        };

        let payload_a = encode_trace_run_evidence(&ev);
        let payload_b = encode_trace_run_evidence(&ev);

        assert_eq!(payload_a, payload_b);
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
    fn mock_reader_exposes_pev_vector() {
        let pev = PolicyEcologyVector {
            conservatism_bias: 1,
            novelty_penalty_bias: 0,
            manipulation_aversion_bias: 0,
            reversibility_bias: 0,
        };

        let reader = MockPvgsReader::with_pev_vector(pev.clone());
        assert_eq!(reader.get_latest_pev(), Some(pev));
        assert!(reader.get_latest_pev_digest().is_none());
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
    fn local_reader_exposes_latest_pev_digest_and_vector() {
        let store = InMemoryPvgs::new();
        let digest = [7u8; 32];
        let pev = ucf::v1::PolicyEcologyVector {
            conservatism_bias: 1,
            novelty_penalty_bias: 1,
            manipulation_aversion_bias: 0,
            reversibility_bias: 0,
        };

        let stored = chip4::pvgs::Pev {
            epoch: 3,
            pev_digest: Some(Digest32::from_array(digest)),
            pev: Some(pev.clone()),
        };
        store.commit_pev_update(stored);

        let reader = LocalPvgsReader::new(store);
        assert_eq!(reader.get_latest_pev_digest(), Some(digest));
        assert_eq!(reader.get_latest_pev(), Some(pev));
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

    #[test]
    fn mock_reader_returns_deterministic_asset_bundle() {
        let mut reader = MockPvgsReader::default();
        let bundle = reader
            .get_latest_asset_bundle()
            .expect("asset bundle query")
            .expect("bundle");

        assert_eq!(bundle.bundle_id, "bundle:mock");
        assert_eq!(bundle.bundle_digest, vec![0u8; 32]);
        assert!(bundle.chunks.is_empty());
    }
}
