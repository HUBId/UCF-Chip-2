#![forbid(unsafe_code)]

use asset_rehydration::AssetRehydrator;
use biophys_asset_builder::CircuitBuilderFromAssets;
use biophys_assets::{
    channel_params_from_payload, channel_params_payload_digest, connectivity_from_payload,
    connectivity_payload_digest, morphology_from_payload, morphology_payload_digest,
    synapse_params_from_payload, synapse_params_payload_digest, ChannelParamsSet,
    ConnectivityGraph, MorphologySet, SynapseParamsSet,
};
use blake3::Hasher;
use microcircuit_amygdala_l4::AmygdalaL4Microcircuit;
use microcircuit_amygdala_stub::{AmyInput, AmyOutput};
use microcircuit_core::MicrocircuitBackend;
use microcircuit_hypothalamus_l4::HypothalamusL4Microcircuit;
use microcircuit_hypothalamus_setpoint::{HypoInput, HypoOutput};
use microcircuit_sn_l4::SnL4Microcircuit;
use microcircuit_sn_stub::{SnInput, SnOutput};
use pvgs_client::{PvgsError, PvgsReader};
use thiserror::Error;
use ucf::v1::{AssetBundle, AssetKind};

const RUN_DIGEST_DOMAIN: &str = "UCF:REPLAY:RUN";
const PROFILE_SEQ_DIGEST_DOMAIN: &str = "UCF:REPLAY:PROFILE_SEQ";
const DWM_SEQ_DIGEST_DOMAIN: &str = "UCF:REPLAY:DWM_SEQ";
const DEFAULT_CACHE_CAPACITY: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayStrictness {
    FailClosed,
}

#[derive(Debug, Clone)]
pub struct ReplayExecutorConfig {
    pub max_init_time_steps: u32,
    pub strictness: ReplayStrictness,
    pub cache_capacity: usize,
}

impl Default for ReplayExecutorConfig {
    fn default() -> Self {
        Self {
            max_init_time_steps: 1024,
            strictness: ReplayStrictness::FailClosed,
            cache_capacity: DEFAULT_CACHE_CAPACITY,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReplayPlan {
    pub sn_asset_manifest_ref: Option<[u8; 32]>,
    pub amygdala_asset_manifest_ref: Option<[u8; 32]>,
    pub hypothalamus_asset_manifest_ref: Option<[u8; 32]>,
}

impl ReplayPlan {
    fn require_manifest_ref(
        ref_opt: Option<[u8; 32]>,
        label: &'static str,
    ) -> Result<[u8; 32], Error> {
        ref_opt.ok_or(Error::MissingAssetManifestRef { label })
    }
}

#[derive(Debug, Clone, Default)]
pub struct ReplayInput {
    pub now_ms: u64,
    pub sn: SnInput,
    pub amygdala: AmyInput,
    pub hypothalamus: HypoInput,
}

#[derive(Debug)]
pub struct PreparedReplay {
    pub sn_manifest_digest: [u8; 32],
    pub amygdala_manifest_digest: [u8; 32],
    pub hypothalamus_manifest_digest: [u8; 32],
    pub sn: SnL4Microcircuit,
    pub amygdala: AmygdalaL4Microcircuit,
    pub hypothalamus: HypothalamusL4Microcircuit,
}

#[derive(Debug, Clone)]
pub struct ReplayRunReport {
    pub steps: u32,
    pub dwm_sequence: Vec<dbm_core::DwmMode>,
    pub threat_vectors_sequence: Vec<Vec<dbm_core::ThreatVector>>,
    pub profile_sequence: Vec<dbm_core::ProfileState>,
    pub run_digest: [u8; 32],
    pub summary_profile_seq_digest: [u8; 32],
    pub summary_dwm_seq_digest: [u8; 32],
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("missing asset manifest ref for {label}")]
    MissingAssetManifestRef { label: &'static str },
    #[error("asset bundle missing for manifest digest")]
    MissingAssetBundle,
    #[error("asset bundle verification failed: {0}")]
    BundleVerification(#[from] asset_rehydration::RehydrationError),
    #[error("asset build failed: {0}")]
    AssetBuild(#[from] biophys_asset_builder::Error),
    #[error("pvgs error: {0}")]
    Pvgs(#[from] PvgsError),
}

#[derive(Debug)]
pub struct ReplayExecutor {
    cache: BundleCache,
    max_init_time_steps: u32,
    strictness: ReplayStrictness,
}

impl ReplayExecutor {
    pub fn new(config: ReplayExecutorConfig) -> Self {
        Self {
            cache: BundleCache::new(config.cache_capacity),
            max_init_time_steps: config.max_init_time_steps,
            strictness: config.strictness,
        }
    }

    pub fn prepare(
        &mut self,
        plan: &ReplayPlan,
        pvgs: &mut dyn PvgsReader,
    ) -> Result<PreparedReplay, Error> {
        match self.strictness {
            ReplayStrictness::FailClosed => {}
        }
        let sn_digest = ReplayPlan::require_manifest_ref(plan.sn_asset_manifest_ref, "sn")?;
        let amygdala_digest =
            ReplayPlan::require_manifest_ref(plan.amygdala_asset_manifest_ref, "amygdala")?;
        let hypothalamus_digest =
            ReplayPlan::require_manifest_ref(plan.hypothalamus_asset_manifest_ref, "hypothalamus")?;

        let sn_bundle = self
            .fetch_bundle(sn_digest, pvgs)?
            .ok_or(Error::MissingAssetBundle)?;
        let amygdala_bundle = self
            .fetch_bundle(amygdala_digest, pvgs)?
            .ok_or(Error::MissingAssetBundle)?;
        let hypothalamus_bundle = self
            .fetch_bundle(hypothalamus_digest, pvgs)?
            .ok_or(Error::MissingAssetBundle)?;

        let rehydrator = AssetRehydrator::new();
        rehydrator.verify_bundle(&sn_bundle)?;
        rehydrator.verify_bundle(&amygdala_bundle)?;
        rehydrator.verify_bundle(&hypothalamus_bundle)?;

        let sn_assets = rehydrate_assets(&rehydrator, &sn_bundle)?;
        let amygdala_assets = rehydrate_assets(&rehydrator, &amygdala_bundle)?;
        let hypothalamus_assets = rehydrate_assets(&rehydrator, &hypothalamus_bundle)?;

        let sn = SnL4Microcircuit::build_from_assets(
            &sn_assets.morph,
            &sn_assets.chan,
            &sn_assets.syn,
            &sn_assets.conn,
        )?;
        let amygdala = AmygdalaL4Microcircuit::build_from_assets(
            &amygdala_assets.morph,
            &amygdala_assets.chan,
            &amygdala_assets.syn,
            &amygdala_assets.conn,
        )?;
        let hypothalamus = HypothalamusL4Microcircuit::build_from_assets(
            &hypothalamus_assets.morph,
            &hypothalamus_assets.chan,
            &hypothalamus_assets.syn,
            &hypothalamus_assets.conn,
        )?;

        Ok(PreparedReplay {
            sn_manifest_digest: sn_digest,
            amygdala_manifest_digest: amygdala_digest,
            hypothalamus_manifest_digest: hypothalamus_digest,
            sn,
            amygdala,
            hypothalamus,
        })
    }

    pub fn run(
        &self,
        prepared: &mut PreparedReplay,
        steps: u32,
        input_sequence: &[ReplayInput],
    ) -> ReplayRunReport {
        let clamped_steps = steps.min(self.max_init_time_steps);
        let mut dwm_sequence = Vec::with_capacity(clamped_steps as usize);
        let mut threat_vectors_sequence = Vec::with_capacity(clamped_steps as usize);
        let mut profile_sequence = Vec::with_capacity(clamped_steps as usize);

        for step in 0..clamped_steps as usize {
            let input = input_sequence.get(step).cloned().unwrap_or_default();
            let now_ms = input.now_ms;

            let sn_output: SnOutput = prepared.sn.step(&input.sn, now_ms);
            let amy_output: AmyOutput = prepared.amygdala.step(&input.amygdala, now_ms);
            let hypo_output: HypoOutput = prepared.hypothalamus.step(&input.hypothalamus, now_ms);

            dwm_sequence.push(sn_output.dwm);
            threat_vectors_sequence.push(amy_output.vectors);
            profile_sequence.push(hypo_output.profile_state);
        }

        let run_digest = digest_run(&dwm_sequence, &threat_vectors_sequence, &profile_sequence);
        let summary_profile_seq_digest = digest_profile_sequence(&profile_sequence);
        let summary_dwm_seq_digest = digest_dwm_sequence(&dwm_sequence);

        ReplayRunReport {
            steps: clamped_steps,
            dwm_sequence,
            threat_vectors_sequence,
            profile_sequence,
            run_digest,
            summary_profile_seq_digest,
            summary_dwm_seq_digest,
        }
    }

    fn fetch_bundle(
        &mut self,
        digest: [u8; 32],
        pvgs: &mut dyn PvgsReader,
    ) -> Result<Option<AssetBundle>, Error> {
        if let Some(bundle) = self.cache.get(&digest) {
            return Ok(Some(bundle));
        }
        let bundle = pvgs.find_asset_bundle_by_manifest_digest(digest)?;
        if let Some(bundle) = bundle.clone() {
            self.cache.insert(digest, bundle);
        }
        Ok(bundle)
    }
}

#[derive(Debug)]
struct BundleCache {
    capacity: usize,
    entries: Vec<([u8; 32], AssetBundle)>,
}

impl BundleCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: Vec::new(),
        }
    }

    fn get(&mut self, key: &[u8; 32]) -> Option<AssetBundle> {
        if let Some(pos) = self.entries.iter().position(|(digest, _)| digest == key) {
            let entry = self.entries.remove(pos);
            self.entries.insert(0, entry.clone());
            return Some(entry.1);
        }
        None
    }

    fn insert(&mut self, key: [u8; 32], bundle: AssetBundle) {
        if let Some(pos) = self.entries.iter().position(|(digest, _)| *digest == key) {
            self.entries.remove(pos);
        }
        self.entries.insert(0, (key, bundle));
        if self.entries.len() > self.capacity {
            self.entries.truncate(self.capacity);
        }
    }
}

#[derive(Debug)]
struct RehydratedAssets {
    morph: MorphologySet,
    chan: ChannelParamsSet,
    syn: SynapseParamsSet,
    conn: ConnectivityGraph,
}

fn rehydrate_assets(
    rehydrator: &AssetRehydrator,
    bundle: &AssetBundle,
) -> Result<RehydratedAssets, Error> {
    let manifest = bundle
        .manifest
        .as_ref()
        .ok_or(asset_rehydration::RehydrationError::MissingManifest)?;
    let morph_digest = manifest_digest_for_kind(manifest, AssetKind::MorphologySet)?;
    let chan_digest = manifest_digest_for_kind(manifest, AssetKind::ChannelParamsSet)?;
    let syn_digest = manifest_digest_for_kind(manifest, AssetKind::SynapseParamsSet)?;
    let conn_digest = manifest_digest_for_kind(manifest, AssetKind::ConnectivityGraph)?;

    let morph_bytes = rehydrator.reassemble(bundle, AssetKind::MorphologySet, morph_digest)?;
    let chan_bytes = rehydrator.reassemble(bundle, AssetKind::ChannelParamsSet, chan_digest)?;
    let syn_bytes = rehydrator.reassemble(bundle, AssetKind::SynapseParamsSet, syn_digest)?;
    let conn_bytes = rehydrator.reassemble(bundle, AssetKind::ConnectivityGraph, conn_digest)?;

    let morph_payload = rehydrator.decode_morphology(&morph_bytes)?;
    let chan_payload = rehydrator.decode_channel_params(&chan_bytes)?;
    let syn_payload = rehydrator.decode_synapse_params(&syn_bytes)?;
    let conn_payload = rehydrator.decode_connectivity(&conn_bytes)?;

    verify_asset_digest(
        "morphology",
        morphology_payload_digest(&morph_payload),
        morph_digest,
    )?;
    verify_asset_digest(
        "channel_params",
        channel_params_payload_digest(&chan_payload),
        chan_digest,
    )?;
    verify_asset_digest(
        "synapse_params",
        synapse_params_payload_digest(&syn_payload),
        syn_digest,
    )?;
    verify_asset_digest(
        "connectivity",
        connectivity_payload_digest(&conn_payload),
        conn_digest,
    )?;

    let morph = morphology_from_payload(&morph_payload)
        .map_err(|message| biophys_asset_builder::Error::InvalidAssetData { message })?;
    let chan = channel_params_from_payload(&chan_payload)
        .map_err(|message| biophys_asset_builder::Error::InvalidAssetData { message })?;
    let syn = synapse_params_from_payload(&syn_payload)
        .map_err(|message| biophys_asset_builder::Error::InvalidAssetData { message })?;
    let conn = connectivity_from_payload(&conn_payload, &syn_payload)
        .map_err(|message| biophys_asset_builder::Error::InvalidAssetData { message })?;

    Ok(RehydratedAssets {
        morph,
        chan,
        syn,
        conn,
    })
}

fn manifest_digest_for_kind(
    manifest: &ucf::v1::AssetManifest,
    kind: AssetKind,
) -> Result<[u8; 32], asset_rehydration::RehydrationError> {
    let component = manifest
        .components
        .iter()
        .find(|component| component.kind == kind as i32)
        .ok_or(asset_rehydration::RehydrationError::MissingAssetDigest { kind })?;
    digest_from_vec(&component.digest, "asset_digest")
}

fn digest_from_vec(
    bytes: &[u8],
    label: &'static str,
) -> Result<[u8; 32], asset_rehydration::RehydrationError> {
    if bytes.len() != 32 {
        return Err(asset_rehydration::RehydrationError::InvalidDigestLength {
            label,
            len: bytes.len(),
        });
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(bytes);
    Ok(out)
}

fn verify_asset_digest(
    label: &'static str,
    computed: [u8; 32],
    expected: [u8; 32],
) -> Result<(), asset_rehydration::RehydrationError> {
    if computed != expected {
        return Err(asset_rehydration::RehydrationError::DecodeFailed {
            message: format!("{label} digest mismatch"),
        });
    }
    Ok(())
}

fn digest_run(
    dwm_sequence: &[dbm_core::DwmMode],
    threat_vectors_sequence: &[Vec<dbm_core::ThreatVector>],
    profile_sequence: &[dbm_core::ProfileState],
) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(RUN_DIGEST_DOMAIN.as_bytes());
    for dwm in dwm_sequence {
        hasher.update(&[dwm_code(*dwm)]);
    }
    for vectors in threat_vectors_sequence {
        hasher.update(&(vectors.len() as u32).to_le_bytes());
        for vector in vectors {
            hasher.update(&[threat_vector_code(*vector)]);
        }
    }
    for profile in profile_sequence {
        hasher.update(&[profile_code(*profile)]);
    }
    *hasher.finalize().as_bytes()
}

fn digest_profile_sequence(profile_sequence: &[dbm_core::ProfileState]) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(PROFILE_SEQ_DIGEST_DOMAIN.as_bytes());
    for profile in profile_sequence {
        hasher.update(&[profile_code(*profile)]);
    }
    *hasher.finalize().as_bytes()
}

fn digest_dwm_sequence(dwm_sequence: &[dbm_core::DwmMode]) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(DWM_SEQ_DIGEST_DOMAIN.as_bytes());
    for dwm in dwm_sequence {
        hasher.update(&[dwm_code(*dwm)]);
    }
    *hasher.finalize().as_bytes()
}

fn dwm_code(mode: dbm_core::DwmMode) -> u8 {
    match mode {
        dbm_core::DwmMode::ExecPlan => 0,
        dbm_core::DwmMode::Simulate => 1,
        dbm_core::DwmMode::Stabilize => 2,
        dbm_core::DwmMode::Report => 3,
    }
}

fn threat_vector_code(vector: dbm_core::ThreatVector) -> u8 {
    match vector {
        dbm_core::ThreatVector::Exfil => 0,
        dbm_core::ThreatVector::Probing => 1,
        dbm_core::ThreatVector::IntegrityCompromise => 2,
        dbm_core::ThreatVector::RuntimeEscape => 3,
        dbm_core::ThreatVector::ToolSideEffects => 4,
    }
}

fn profile_code(profile: dbm_core::ProfileState) -> u8 {
    match profile {
        dbm_core::ProfileState::M0 => 0,
        dbm_core::ProfileState::M1 => 1,
        dbm_core::ProfileState::M2 => 2,
        dbm_core::ProfileState::M3 => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use asset_chunker::{
        build_asset_bundle_with_policy, chunk_asset, BundleIdPolicy, ChunkerConfig,
    };
    use asset_rehydration::ASSET_MANIFEST_DOMAIN;
    use biophys_assets::{
        demo_channel_params, demo_connectivity, demo_morphology_3comp, demo_syn_params,
        to_asset_digest,
    };
    use prost::Message;
    use pvgs_client::MockPvgsReader;
    use ucf::v1::{AssetBundle, AssetManifest, Compression};

    const SN_NEURONS: u32 = 14;
    const AMYGDALA_NEURONS: u32 = 9;
    const HYPOTHALAMUS_NEURONS: u32 = 16;

    fn compute_manifest_digest(manifest: &AssetManifest) -> [u8; 32] {
        let mut normalized = manifest.clone();
        normalized.manifest_digest = vec![0u8; 32];
        let mut hasher = blake3::Hasher::new();
        hasher.update(ASSET_MANIFEST_DOMAIN.as_bytes());
        hasher.update(&normalized.encode_to_vec());
        *hasher.finalize().as_bytes()
    }

    fn build_bundle(neuron_count: u32, created_at_ms: u64) -> AssetBundle {
        let morph = demo_morphology_3comp(neuron_count);
        let chan = demo_channel_params(&morph);
        let syn = demo_syn_params();
        let conn = demo_connectivity(neuron_count, &syn);

        let mut manifest = AssetManifest {
            manifest_version: 1,
            created_at_ms,
            manifest_digest: vec![0u8; 32],
            components: vec![
                to_asset_digest(
                    AssetKind::MorphologySet,
                    morph.version,
                    morph.digest(),
                    created_at_ms,
                    None,
                ),
                to_asset_digest(
                    AssetKind::ChannelParamsSet,
                    chan.version,
                    chan.digest(),
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
                    conn.version,
                    conn.digest(),
                    created_at_ms,
                    None,
                ),
            ],
        };

        let manifest_digest = compute_manifest_digest(&manifest);
        manifest.manifest_digest = manifest_digest.to_vec();

        let chunker = ChunkerConfig {
            max_chunk_bytes: 128,
            compression: Compression::None,
            max_chunks_total: 512,
            bundle_id_policy: BundleIdPolicy::ManifestDigestPrefix { prefix_len: 8 },
        };

        let mut chunks = Vec::new();
        chunks.extend(
            chunk_asset(
                AssetKind::MorphologySet,
                morph.version,
                morph.digest(),
                &morph.to_canonical_bytes(),
                &chunker,
                created_at_ms,
            )
            .expect("morph chunks"),
        );
        chunks.extend(
            chunk_asset(
                AssetKind::ChannelParamsSet,
                chan.version,
                chan.digest(),
                &chan.to_canonical_bytes(),
                &chunker,
                created_at_ms,
            )
            .expect("chan chunks"),
        );
        chunks.extend(
            chunk_asset(
                AssetKind::SynapseParamsSet,
                syn.version,
                syn.digest(),
                &syn.to_canonical_bytes(),
                &chunker,
                created_at_ms,
            )
            .expect("syn chunks"),
        );
        chunks.extend(
            chunk_asset(
                AssetKind::ConnectivityGraph,
                conn.version,
                conn.digest(),
                &conn.to_canonical_bytes(),
                &chunker,
                created_at_ms,
            )
            .expect("conn chunks"),
        );
        chunks.sort_by(|a, b| {
            a.asset_digest
                .cmp(&b.asset_digest)
                .then_with(|| a.chunk_index.cmp(&b.chunk_index))
        });

        build_asset_bundle_with_policy(
            manifest,
            chunks,
            created_at_ms,
            BundleIdPolicy::ManifestDigestPrefix { prefix_len: 8 },
        )
    }

    fn manifest_digest(bundle: &AssetBundle) -> [u8; 32] {
        let mut digest = [0u8; 32];
        let manifest = bundle.manifest.as_ref().expect("manifest");
        digest.copy_from_slice(&manifest.manifest_digest);
        digest
    }

    fn executor_with_bundles(bundles: Vec<AssetBundle>) -> (ReplayExecutor, MockPvgsReader) {
        let mut reader = MockPvgsReader::default();
        for bundle in bundles {
            reader.push_asset_bundle(bundle);
        }
        let executor = ReplayExecutor::new(ReplayExecutorConfig {
            max_init_time_steps: 4,
            strictness: ReplayStrictness::FailClosed,
            cache_capacity: 4,
        });
        (executor, reader)
    }

    #[test]
    fn prepare_fails_without_asset_ref() {
        let plan = ReplayPlan {
            sn_asset_manifest_ref: None,
            amygdala_asset_manifest_ref: Some([1u8; 32]),
            hypothalamus_asset_manifest_ref: Some([2u8; 32]),
        };
        let mut executor = ReplayExecutor::new(ReplayExecutorConfig::default());
        let mut pvgs = MockPvgsReader::default();

        let err = executor.prepare(&plan, &mut pvgs).unwrap_err();
        assert!(matches!(err, Error::MissingAssetManifestRef { .. }));
    }

    #[test]
    fn prepare_loads_correct_bundle() {
        let sn_bundle_a = build_bundle(SN_NEURONS, 10);
        let sn_bundle_b = build_bundle(SN_NEURONS, 11);
        let amyg_bundle = build_bundle(AMYGDALA_NEURONS, 12);
        let hypo_bundle = build_bundle(HYPOTHALAMUS_NEURONS, 13);

        let sn_digest_a = manifest_digest(&sn_bundle_a);
        let sn_digest_b = manifest_digest(&sn_bundle_b);

        let (mut executor, mut pvgs) = executor_with_bundles(vec![
            sn_bundle_a.clone(),
            sn_bundle_b.clone(),
            amyg_bundle.clone(),
            hypo_bundle.clone(),
        ]);

        let plan = ReplayPlan {
            sn_asset_manifest_ref: Some(sn_digest_b),
            amygdala_asset_manifest_ref: Some(manifest_digest(&amyg_bundle)),
            hypothalamus_asset_manifest_ref: Some(manifest_digest(&hypo_bundle)),
        };

        let prepared = executor.prepare(&plan, &mut pvgs).expect("prepare");
        assert_eq!(prepared.sn_manifest_digest, sn_digest_b);
        assert_ne!(prepared.sn_manifest_digest, sn_digest_a);
    }

    #[test]
    fn deterministic_run_digest() {
        let sn_bundle = build_bundle(SN_NEURONS, 20);
        let amyg_bundle = build_bundle(AMYGDALA_NEURONS, 21);
        let hypo_bundle = build_bundle(HYPOTHALAMUS_NEURONS, 22);

        let plan = ReplayPlan {
            sn_asset_manifest_ref: Some(manifest_digest(&sn_bundle)),
            amygdala_asset_manifest_ref: Some(manifest_digest(&amyg_bundle)),
            hypothalamus_asset_manifest_ref: Some(manifest_digest(&hypo_bundle)),
        };

        let (mut executor, mut pvgs) = executor_with_bundles(vec![
            sn_bundle.clone(),
            amyg_bundle.clone(),
            hypo_bundle.clone(),
        ]);

        let mut prepared_a = executor.prepare(&plan, &mut pvgs).expect("prepare a");
        let mut prepared_b = executor.prepare(&plan, &mut pvgs).expect("prepare b");

        let input_sequence = vec![
            ReplayInput {
                now_ms: 1,
                ..ReplayInput::default()
            },
            ReplayInput {
                now_ms: 2,
                sn: SnInput {
                    replay_hint: true,
                    ..Default::default()
                },
                ..ReplayInput::default()
            },
        ];

        let report_a = executor.run(&mut prepared_a, 2, &input_sequence);
        let report_b = executor.run(&mut prepared_b, 2, &input_sequence);

        assert_eq!(report_a.run_digest, report_b.run_digest);
    }

    #[test]
    fn run_clamps_steps() {
        let sn_bundle = build_bundle(SN_NEURONS, 30);
        let amyg_bundle = build_bundle(AMYGDALA_NEURONS, 31);
        let hypo_bundle = build_bundle(HYPOTHALAMUS_NEURONS, 32);

        let plan = ReplayPlan {
            sn_asset_manifest_ref: Some(manifest_digest(&sn_bundle)),
            amygdala_asset_manifest_ref: Some(manifest_digest(&amyg_bundle)),
            hypothalamus_asset_manifest_ref: Some(manifest_digest(&hypo_bundle)),
        };

        let (mut executor, mut pvgs) = executor_with_bundles(vec![
            sn_bundle.clone(),
            amyg_bundle.clone(),
            hypo_bundle.clone(),
        ]);
        executor.max_init_time_steps = 1;

        let mut prepared = executor.prepare(&plan, &mut pvgs).expect("prepare");
        let report = executor.run(&mut prepared, 5, &[]);

        assert_eq!(report.steps, 1);
        assert_eq!(report.dwm_sequence.len(), 1);
    }

    #[test]
    fn prepare_fails_on_invalid_bundle_digest() {
        let mut sn_bundle = build_bundle(SN_NEURONS, 40);
        sn_bundle.bundle_digest[0] ^= 0xFF;
        let amyg_bundle = build_bundle(AMYGDALA_NEURONS, 41);
        let hypo_bundle = build_bundle(HYPOTHALAMUS_NEURONS, 42);

        let plan = ReplayPlan {
            sn_asset_manifest_ref: Some(manifest_digest(&sn_bundle)),
            amygdala_asset_manifest_ref: Some(manifest_digest(&amyg_bundle)),
            hypothalamus_asset_manifest_ref: Some(manifest_digest(&hypo_bundle)),
        };

        let (mut executor, mut pvgs) = executor_with_bundles(vec![
            sn_bundle.clone(),
            amyg_bundle.clone(),
            hypo_bundle.clone(),
        ]);

        let err = executor.prepare(&plan, &mut pvgs).unwrap_err();
        assert!(matches!(err, Error::BundleVerification(_)));
    }
}
