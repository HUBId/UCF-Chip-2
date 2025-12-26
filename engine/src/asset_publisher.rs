#![forbid(unsafe_code)]

use asset_chunker::{
    build_asset_bundle_with_policy, chunk_asset, BundleIdPolicy, ChunkerConfig, ChunkerError,
};
use biophys_assets::{
    demo_channel_params, demo_connectivity, demo_morphology_3comp, demo_syn_params,
    to_asset_digest, ChannelParamsSet, ConnectivityGraph, MorphologySet, SynapseParamsSet,
};
use blake3::Hasher;
use prost::Message;
use pvgs_client::PvgsWriter;
use ucf::v1::{AssetChunk, AssetDigest, AssetKind, AssetManifest, Compression};

const ASSET_MANIFEST_DOMAIN: &str = "UCF:ASSET:MANIFEST";
pub(crate) const ASSET_MANIFEST_VERSION: u32 = 1;
pub(crate) const DEFAULT_ASSET_NEURONS: u32 = 4;

#[derive(Debug, Default)]
pub(crate) struct AssetPublisherState {
    published_bundle_digest: Option<[u8; 32]>,
    created_at_ms_fixed: Option<u64>,
}

impl AssetPublisherState {
    pub(crate) fn fixed_created_at_ms(&mut self, now_ms: u64) -> u64 {
        *self.created_at_ms_fixed.get_or_insert(now_ms)
    }

    pub(crate) fn maybe_publish(
        &mut self,
        now_ms: u64,
        neurons: u32,
        writer: Option<&mut (dyn PvgsWriter + Send + '_)>,
    ) -> [u8; 32] {
        let created_at_ms = self.fixed_created_at_ms(now_ms);
        let assets = build_biophys_assets(neurons);
        let components = build_biophys_components(created_at_ms, &assets);
        let manifest = build_manifest(ASSET_MANIFEST_VERSION, created_at_ms, components);
        let manifest_digest = manifest_digest_from_proto(&manifest);
        let cfg = default_chunker_config();
        let chunks = match build_biophys_chunks(&assets, &cfg, created_at_ms) {
            Ok(chunks) => chunks,
            Err(_) => {
                self.published_bundle_digest = None;
                return manifest_digest;
            }
        };
        let bundle =
            build_asset_bundle_with_policy(manifest, chunks, created_at_ms, cfg.bundle_id_policy);
        let bundle_digest = bundle_digest_from_proto(&bundle);

        let Some(writer) = writer else {
            self.published_bundle_digest = None;
            return manifest_digest;
        };

        if self
            .published_bundle_digest
            .map(|current| current == bundle_digest)
            .unwrap_or(false)
        {
            return manifest_digest;
        }

        if writer.commit_asset_bundle(bundle).is_ok() {
            self.published_bundle_digest = Some(bundle_digest);
        }

        manifest_digest
    }
}

#[derive(Debug, Clone)]
pub(crate) struct BiophysAssets {
    pub(crate) morph: MorphologySet,
    pub(crate) channels: ChannelParamsSet,
    pub(crate) syn: SynapseParamsSet,
    pub(crate) graph: ConnectivityGraph,
}

pub(crate) fn build_biophys_assets(neurons: u32) -> BiophysAssets {
    let morph = demo_morphology_3comp(neurons);
    let channels = demo_channel_params(&morph);
    let syn = demo_syn_params();
    let graph = demo_connectivity(neurons, &syn);

    BiophysAssets {
        morph,
        channels,
        syn,
        graph,
    }
}

pub(crate) fn build_biophys_components(
    created_at_ms: u64,
    assets: &BiophysAssets,
) -> Vec<AssetDigest> {
    let morph = &assets.morph;
    let channels = &assets.channels;
    let syn = &assets.syn;
    let graph = &assets.graph;

    vec![
        to_asset_digest(
            AssetKind::MorphologySet,
            morph.version,
            morph.digest(),
            created_at_ms,
            None,
        ),
        to_asset_digest(
            AssetKind::ChannelParamsSet,
            channels.version,
            channels.digest(),
            created_at_ms,
            None,
        ),
        to_asset_digest(
            AssetKind::SynapseParamsSet,
            syn.version,
            syn.digest(),
            created_at_ms,
            None,
        ),
        to_asset_digest(
            AssetKind::ConnectivityGraph,
            graph.version,
            graph.digest(),
            created_at_ms,
            None,
        ),
    ]
}

fn build_biophys_chunks(
    assets: &BiophysAssets,
    cfg: &ChunkerConfig,
    created_at_ms: u64,
) -> Result<Vec<AssetChunk>, ChunkerError> {
    let mut chunks = Vec::new();
    chunks.extend(chunk_asset(
        AssetKind::MorphologySet,
        assets.morph.version,
        assets.morph.digest(),
        &assets.morph.to_canonical_bytes(),
        cfg,
        created_at_ms,
    )?);
    chunks.extend(chunk_asset(
        AssetKind::ChannelParamsSet,
        assets.channels.version,
        assets.channels.digest(),
        &assets.channels.to_canonical_bytes(),
        cfg,
        created_at_ms,
    )?);
    chunks.extend(chunk_asset(
        AssetKind::SynapseParamsSet,
        assets.syn.version,
        assets.syn.digest(),
        &assets.syn.to_canonical_bytes(),
        cfg,
        created_at_ms,
    )?);
    chunks.extend(chunk_asset(
        AssetKind::ConnectivityGraph,
        assets.graph.version,
        assets.graph.digest(),
        &assets.graph.to_canonical_bytes(),
        cfg,
        created_at_ms,
    )?);

    Ok(chunks)
}

fn default_chunker_config() -> ChunkerConfig {
    ChunkerConfig {
        max_chunk_bytes: 64 * 1024,
        compression: Compression::None,
        max_chunks_total: 4096,
        bundle_id_policy: BundleIdPolicy::ManifestDigestPrefix { prefix_len: 12 },
    }
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

fn bundle_digest_from_proto(bundle: &ucf::v1::AssetBundle) -> [u8; 32] {
    let mut digest = [0u8; 32];
    digest.copy_from_slice(&bundle.bundle_digest[..32]);
    digest
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct RecordingWriter {
        commits: Arc<Mutex<Vec<ucf::v1::AssetBundle>>>,
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
            _manifest: AssetManifest,
        ) -> Result<ucf::v1::PvgsReceipt, pvgs_client::PvgsError> {
            Ok(ucf::v1::PvgsReceipt::default())
        }

        fn commit_asset_bundle(
            &mut self,
            bundle: ucf::v1::AssetBundle,
        ) -> Result<ucf::v1::PvgsReceipt, pvgs_client::PvgsError> {
            self.commits.lock().unwrap().push(bundle);
            Ok(ucf::v1::PvgsReceipt::default())
        }

        fn commit_replay_run_evidence(
            &mut self,
            _evidence: ucf::v1::ReplayRunEvidence,
        ) -> Result<ucf::v1::PvgsReceipt, pvgs_client::PvgsError> {
            Ok(ucf::v1::PvgsReceipt::default())
        }

        fn commit_trace_run_evidence(
            &mut self,
            _evidence: pvgs_client::TraceRunEvidenceLike,
        ) -> Result<ucf::v1::PvgsReceipt, pvgs_client::PvgsError> {
            Ok(ucf::v1::PvgsReceipt::default())
        }
    }

    fn build_assets() -> BiophysAssets {
        build_biophys_assets(2)
    }

    fn build_components(created_at_ms: u64, version_override: Option<u32>) -> Vec<AssetDigest> {
        let assets = build_assets();
        let mut components = build_biophys_components(created_at_ms, &assets);
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

        publisher.maybe_publish(10, 2, Some(&mut writer));

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

        publisher.maybe_publish(10, 2, Some(&mut writer));
        publisher.maybe_publish(20, 2, Some(&mut writer));

        let commits = commits.lock().unwrap();
        assert_eq!(commits.len(), 1);
    }

    #[test]
    fn recommit_on_asset_change() {
        let mut publisher = AssetPublisherState::default();
        let commits = Arc::new(Mutex::new(Vec::new()));
        let mut writer = RecordingWriter {
            commits: commits.clone(),
        };

        publisher.maybe_publish(10, 2, Some(&mut writer));
        publisher.maybe_publish(20, 3, Some(&mut writer));

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
