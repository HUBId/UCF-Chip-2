#![forbid(unsafe_code)]

use biophys_assets::{
    demo_channel_params, demo_connectivity, demo_morphology_3comp, demo_syn_params, to_asset_digest,
};
use blake3::Hasher;
use prost::Message;
use pvgs_client::PvgsWriter;
use ucf::v1::{AssetDigest, AssetKind, AssetManifest};

const ASSET_MANIFEST_DOMAIN: &str = "UCF:ASSET:MANIFEST";
pub(crate) const ASSET_MANIFEST_VERSION: u32 = 1;
pub(crate) const DEFAULT_ASSET_NEURONS: u32 = 4;

#[derive(Debug, Default)]
pub(crate) struct AssetPublisherState {
    published_manifest_digest: Option<[u8; 32]>,
    created_at_ms_fixed: Option<u64>,
}

impl AssetPublisherState {
    pub(crate) fn fixed_created_at_ms(&mut self, now_ms: u64) -> u64 {
        *self.created_at_ms_fixed.get_or_insert(now_ms)
    }

    pub(crate) fn maybe_publish(
        &mut self,
        now_ms: u64,
        components: Vec<AssetDigest>,
        writer: Option<&mut (dyn PvgsWriter + Send + '_)>,
    ) -> [u8; 32] {
        let created_at_ms = self.fixed_created_at_ms(now_ms);
        let manifest = build_manifest(ASSET_MANIFEST_VERSION, created_at_ms, components);
        let digest = manifest_digest_from_proto(&manifest);

        let Some(writer) = writer else {
            self.published_manifest_digest = None;
            return digest;
        };

        if self
            .published_manifest_digest
            .map(|current| current == digest)
            .unwrap_or(false)
        {
            return digest;
        }

        if writer.commit_asset_manifest(manifest).is_ok() {
            self.published_manifest_digest = Some(digest);
        }

        digest
    }
}

pub(crate) fn build_biophys_components(created_at_ms: u64, neurons: u32) -> Vec<AssetDigest> {
    let morph = demo_morphology_3comp(neurons);
    let channels = demo_channel_params(&morph);
    let syn = demo_syn_params();
    let graph = demo_connectivity(neurons, &syn);

    vec![
        to_asset_digest(
            AssetKind::Morphology,
            morph.version,
            morph.digest(),
            created_at_ms,
            None,
        ),
        to_asset_digest(
            AssetKind::ChannelParams,
            channels.version,
            channels.digest(),
            created_at_ms,
            None,
        ),
        to_asset_digest(
            AssetKind::SynapseParams,
            syn.version,
            syn.digest(),
            created_at_ms,
            None,
        ),
        to_asset_digest(
            AssetKind::Connectivity,
            graph.version,
            graph.digest(),
            created_at_ms,
            None,
        ),
    ]
}

pub(crate) fn build_manifest(
    manifest_version: u32,
    created_at_ms: u64,
    components: Vec<AssetDigest>,
) -> AssetManifest {
    let mut manifest = AssetManifest {
        manifest_version,
        created_at_ms,
        manifest_digest: vec![0u8; 32],
        components,
    };

    let digest = compute_manifest_digest(&manifest);
    manifest.manifest_digest = digest.to_vec();
    manifest
}

pub(crate) fn compute_manifest_digest(manifest: &AssetManifest) -> [u8; 32] {
    let mut normalized = manifest.clone();
    normalized.manifest_digest = vec![0u8; 32];
    let payload = encode_manifest_deterministic(&normalized);
    let mut hasher = Hasher::new();
    hasher.update(ASSET_MANIFEST_DOMAIN.as_bytes());
    hasher.update(&payload);
    *hasher.finalize().as_bytes()
}

fn encode_manifest_deterministic(manifest: &AssetManifest) -> Vec<u8> {
    manifest.encode_to_vec()
}

fn manifest_digest_from_proto(manifest: &AssetManifest) -> [u8; 32] {
    let mut digest = [0u8; 32];
    digest.copy_from_slice(&manifest.manifest_digest[..32]);
    digest
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct RecordingWriter {
        commits: Arc<Mutex<Vec<AssetManifest>>>,
    }

    impl PvgsWriter for RecordingWriter {
        fn commit_control_frame_evidence(
            &mut self,
            _session_id: &str,
            _control_frame_digest: [u8; 32],
        ) -> Result<(), pvgs_client::PvgsError> {
            Ok(())
        }

        fn commit_microcircuit_config(
            &mut self,
            _evidence: ucf::v1::MicrocircuitConfigEvidence,
        ) -> Result<ucf::v1::PvgsReceipt, pvgs_client::PvgsError> {
            Ok(ucf::v1::PvgsReceipt::default())
        }

        fn commit_asset_manifest(
            &mut self,
            manifest: AssetManifest,
        ) -> Result<ucf::v1::PvgsReceipt, pvgs_client::PvgsError> {
            self.commits.lock().unwrap().push(manifest);
            Ok(ucf::v1::PvgsReceipt::default())
        }
    }

    fn build_components(created_at_ms: u64, version_override: Option<u32>) -> Vec<AssetDigest> {
        let mut components = build_biophys_components(created_at_ms, 2);
        if let Some(version) = version_override {
            if let Some(first) = components.first_mut() {
                first.version = version;
            }
        }
        components
    }

    #[test]
    fn manifest_digest_is_deterministic() {
        let components = build_components(10, None);
        let manifest_a = build_manifest(ASSET_MANIFEST_VERSION, 10, components.clone());
        let manifest_b = build_manifest(ASSET_MANIFEST_VERSION, 10, components);

        assert_eq!(manifest_a.manifest_digest, manifest_b.manifest_digest);
    }

    #[test]
    fn commit_called_once_on_startup() {
        let mut publisher = AssetPublisherState::default();
        let commits = Arc::new(Mutex::new(Vec::new()));
        let mut writer = RecordingWriter {
            commits: commits.clone(),
        };

        let components = build_components(10, None);
        publisher.maybe_publish(10, components, Some(&mut writer));

        let commits = commits.lock().unwrap();
        assert_eq!(commits.len(), 1);
    }

    #[test]
    fn no_recommit_on_repeat_ticks() {
        let mut publisher = AssetPublisherState::default();
        let commits = Arc::new(Mutex::new(Vec::new()));
        let mut writer = RecordingWriter {
            commits: commits.clone(),
        };

        publisher.maybe_publish(10, build_components(10, None), Some(&mut writer));
        publisher.maybe_publish(20, build_components(10, None), Some(&mut writer));

        let commits = commits.lock().unwrap();
        assert_eq!(commits.len(), 1);
    }

    #[test]
    fn recommit_on_version_bump() {
        let mut publisher = AssetPublisherState::default();
        let commits = Arc::new(Mutex::new(Vec::new()));
        let mut writer = RecordingWriter {
            commits: commits.clone(),
        };

        publisher.maybe_publish(10, build_components(10, Some(1)), Some(&mut writer));
        publisher.maybe_publish(20, build_components(10, Some(2)), Some(&mut writer));

        let commits = commits.lock().unwrap();
        assert_eq!(commits.len(), 2);
    }

    #[test]
    fn deterministic_encoding_roundtrip() {
        let manifest = build_manifest(ASSET_MANIFEST_VERSION, 10, build_components(10, None));
        let bytes_a = encode_manifest_deterministic(&manifest);
        let decoded = AssetManifest::decode(bytes_a.as_slice()).unwrap();
        let bytes_b = encode_manifest_deterministic(&decoded);

        assert_eq!(bytes_a, bytes_b);
    }
}
