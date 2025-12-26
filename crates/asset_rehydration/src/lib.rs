#![forbid(unsafe_code)]

use biophys_assets::{
    channel_params_from_payload, morphology_from_payload, normalize_payload_channel_params,
    normalize_payload_connectivity, normalize_payload_morphology, normalize_payload_synapse_params,
    synapse_params_from_payload, MAX_EDGES,
};
use blake3::Hasher;
use prost::Message;
use std::io::{Read, Result as IoResult};
use thiserror::Error;
use ucf::v1::{
    AssetBundle, AssetChunk, AssetKind, AssetManifest, ChannelParamsSetPayload, Compression,
    ConnectivityGraphPayload, MorphologySetPayload, SynapseParamsSetPayload,
};

pub const ASSET_CHUNK_DOMAIN: &str = "UCF:ASSET:CHUNK";
pub const ASSET_BUNDLE_DOMAIN: &str = "UCF:ASSET:BUNDLE";
pub const ASSET_MANIFEST_DOMAIN: &str = "UCF:ASSET:MANIFEST";
pub const MAX_ASSET_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum RehydrationError {
    #[error("invalid digest length for {label}: {len}")]
    InvalidDigestLength { label: &'static str, len: usize },
    #[error("asset payload exceeds max bytes")]
    AssetTooLarge,
    #[error("chunk digest mismatch at index {index}")]
    ChunkDigestMismatch { index: usize },
    #[error("bundle digest mismatch")]
    BundleDigestMismatch,
    #[error("manifest digest mismatch")]
    ManifestDigestMismatch,
    #[error("missing chunks for asset")]
    MissingChunks,
    #[error("missing asset manifest")]
    MissingManifest,
    #[error("missing asset digest for kind {kind:?}")]
    MissingAssetDigest { kind: AssetKind },
    #[error("chunk index {index} out of range {count}")]
    ChunkIndexOutOfRange { index: u32, count: u32 },
    #[error("duplicate chunk index {index}")]
    DuplicateChunkIndex { index: u32 },
    #[error("missing chunk index {index}")]
    ChunkIndexGap { index: u32 },
    #[error("chunk count mismatch")]
    ChunkCountMismatch,
    #[error("asset version mismatch")]
    AssetVersionMismatch,
    #[error("unsupported compression {compression:?}")]
    UnsupportedCompression { compression: Compression },
    #[error("decompression failed")]
    DecompressionFailed,
    #[error("decode failed")]
    DecodeFailed,
    #[error("canonical bytes mismatch after decode")]
    CanonicalMismatch,
}

pub struct ChunkStream<'a> {
    chunks: Vec<&'a AssetChunk>,
    idx: usize,
    reader: Option<ChunkReader<'a>>,
}

enum ChunkReader<'a> {
    Raw {
        data: &'a [u8],
        offset: usize,
    },
    Zstd {
        decoder: zstd::stream::Decoder<'a, std::io::BufReader<std::io::Cursor<&'a [u8]>>>,
    },
}

impl<'a> ChunkStream<'a> {
    pub fn new(chunks: Vec<&'a AssetChunk>) -> Self {
        Self {
            chunks,
            idx: 0,
            reader: None,
        }
    }

    fn init_reader(&mut self) -> IoResult<()> {
        let chunk = match self.chunks.get(self.idx) {
            Some(chunk) => *chunk,
            None => return Ok(()),
        };
        let compression = Compression::try_from(chunk.compression).unwrap_or(Compression::Unknown);
        self.reader = Some(match compression {
            Compression::None => ChunkReader::Raw {
                data: &chunk.payload,
                offset: 0,
            },
            Compression::Zstd => ChunkReader::Zstd {
                decoder: zstd::stream::Decoder::new(std::io::Cursor::new(chunk.payload.as_slice()))
                    .map_err(std::io::Error::other)?,
            },
            Compression::Unknown => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "unsupported compression",
                ))
            }
        });
        Ok(())
    }
}

impl Read for ChunkStream<'_> {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<usize> {
        if self.reader.is_none() {
            self.init_reader()?;
        }
        loop {
            let reader = match self.reader.as_mut() {
                Some(reader) => reader,
                None => return Ok(0),
            };
            let read = match reader {
                ChunkReader::Raw { data, offset } => {
                    if *offset >= data.len() {
                        0
                    } else {
                        let remaining = &data[*offset..];
                        let to_copy = remaining.len().min(buf.len());
                        buf[..to_copy].copy_from_slice(&remaining[..to_copy]);
                        *offset += to_copy;
                        to_copy
                    }
                }
                ChunkReader::Zstd { decoder } => decoder.read(buf)?,
            };
            if read == 0 {
                self.idx += 1;
                self.reader = None;
                if self.idx >= self.chunks.len() {
                    return Ok(0);
                }
                self.init_reader()?;
                continue;
            }
            return Ok(read);
        }
    }
}

pub fn verify_asset_bundle(bundle: &AssetBundle) -> Result<(), RehydrationError> {
    if bundle.bundle_digest.len() != 32 {
        return Err(RehydrationError::InvalidDigestLength {
            label: "bundle_digest",
            len: bundle.bundle_digest.len(),
        });
    }

    if let Some(manifest) = &bundle.manifest {
        if manifest.manifest_digest.len() != 32 {
            return Err(RehydrationError::InvalidDigestLength {
                label: "manifest_digest",
                len: manifest.manifest_digest.len(),
            });
        }
        let computed = compute_manifest_digest(manifest);
        if manifest.manifest_digest.as_slice() != computed.as_slice() {
            return Err(RehydrationError::ManifestDigestMismatch);
        }
    }

    for (idx, chunk) in bundle.chunks.iter().enumerate() {
        if chunk.chunk_digest.len() != 32 {
            return Err(RehydrationError::InvalidDigestLength {
                label: "chunk_digest",
                len: chunk.chunk_digest.len(),
            });
        }
        let computed = chunk_digest(&chunk.payload);
        if chunk.chunk_digest.as_slice() != computed.as_slice() {
            return Err(RehydrationError::ChunkDigestMismatch { index: idx });
        }
    }

    let computed = compute_bundle_digest(bundle.manifest.as_ref(), &bundle.chunks)?;
    if bundle.bundle_digest.as_slice() != computed.as_slice() {
        return Err(RehydrationError::BundleDigestMismatch);
    }

    Ok(())
}

pub fn reassemble_asset(
    bundle: &AssetBundle,
    asset_kind: AssetKind,
    asset_digest: [u8; 32],
) -> Result<Vec<u8>, RehydrationError> {
    let mut chunks: Vec<&AssetChunk> = bundle
        .chunks
        .iter()
        .filter(|chunk| {
            chunk.kind == asset_kind as i32 && chunk.asset_digest.as_slice() == asset_digest
        })
        .collect();

    if chunks.is_empty() {
        return Err(RehydrationError::MissingChunks);
    }

    let expected_version = chunks[0].version;
    if chunks.iter().any(|chunk| chunk.version != expected_version) {
        return Err(RehydrationError::AssetVersionMismatch);
    }

    chunks.sort_by_key(|chunk| chunk.chunk_index);
    let expected_count = chunks[0].chunk_count;
    if expected_count == 0 {
        return Err(RehydrationError::ChunkCountMismatch);
    }

    let mut total_len = 0usize;
    let mut ordered = vec![None; expected_count as usize];
    for chunk in chunks {
        if chunk.chunk_count != expected_count {
            return Err(RehydrationError::ChunkCountMismatch);
        }
        if chunk.chunk_index >= expected_count {
            return Err(RehydrationError::ChunkIndexOutOfRange {
                index: chunk.chunk_index,
                count: expected_count,
            });
        }
        let compression = Compression::try_from(chunk.compression).unwrap_or(Compression::Unknown);
        if compression == Compression::Unknown {
            return Err(RehydrationError::UnsupportedCompression { compression });
        }
        if chunk.chunk_digest.len() != 32 {
            return Err(RehydrationError::InvalidDigestLength {
                label: "chunk_digest",
                len: chunk.chunk_digest.len(),
            });
        }
        let computed = chunk_digest(&chunk.payload);
        if chunk.chunk_digest.as_slice() != computed.as_slice() {
            return Err(RehydrationError::ChunkDigestMismatch {
                index: chunk.chunk_index as usize,
            });
        }
        let slot = &mut ordered[chunk.chunk_index as usize];
        if slot.is_some() {
            return Err(RehydrationError::DuplicateChunkIndex {
                index: chunk.chunk_index,
            });
        }
        total_len = total_len.saturating_add(chunk.payload.len());
        *slot = Some(chunk);
    }

    if total_len > MAX_ASSET_BYTES {
        return Err(RehydrationError::AssetTooLarge);
    }

    let chunks = ordered
        .into_iter()
        .enumerate()
        .map(|(idx, slot)| slot.ok_or(RehydrationError::ChunkIndexGap { index: idx as u32 }))
        .collect::<Result<Vec<_>, _>>()?;

    let mut payload = Vec::with_capacity(total_len);
    let mut stream = ChunkStream::new(chunks);
    stream
        .read_to_end(&mut payload)
        .map_err(|_| RehydrationError::DecompressionFailed)?;
    if payload.len() > MAX_ASSET_BYTES {
        return Err(RehydrationError::AssetTooLarge);
    }

    Ok(payload)
}

pub fn load_morphology_payload(
    bundle: &AssetBundle,
) -> Result<MorphologySetPayload, RehydrationError> {
    let manifest = bundle
        .manifest
        .as_ref()
        .ok_or(RehydrationError::MissingManifest)?;
    let digest = manifest_digest_for_kind(manifest, AssetKind::MorphologySet)?;
    let bytes = reassemble_asset(bundle, AssetKind::MorphologySet, digest)?;
    decode_morphology(&bytes)
}

pub fn load_channel_params_payload(
    bundle: &AssetBundle,
) -> Result<ChannelParamsSetPayload, RehydrationError> {
    let manifest = bundle
        .manifest
        .as_ref()
        .ok_or(RehydrationError::MissingManifest)?;
    let digest = manifest_digest_for_kind(manifest, AssetKind::ChannelParamsSet)?;
    let bytes = reassemble_asset(bundle, AssetKind::ChannelParamsSet, digest)?;
    decode_channel_params(&bytes)
}

pub fn load_synapse_params_payload(
    bundle: &AssetBundle,
) -> Result<SynapseParamsSetPayload, RehydrationError> {
    let manifest = bundle
        .manifest
        .as_ref()
        .ok_or(RehydrationError::MissingManifest)?;
    let digest = manifest_digest_for_kind(manifest, AssetKind::SynapseParamsSet)?;
    let bytes = reassemble_asset(bundle, AssetKind::SynapseParamsSet, digest)?;
    decode_synapse_params(&bytes)
}

pub fn load_connectivity_payload(
    bundle: &AssetBundle,
) -> Result<ConnectivityGraphPayload, RehydrationError> {
    let manifest = bundle
        .manifest
        .as_ref()
        .ok_or(RehydrationError::MissingManifest)?;
    let digest = manifest_digest_for_kind(manifest, AssetKind::ConnectivityGraph)?;
    let bytes = reassemble_asset(bundle, AssetKind::ConnectivityGraph, digest)?;
    decode_connectivity(&bytes)
}

pub fn decode_morphology(bytes: &[u8]) -> Result<MorphologySetPayload, RehydrationError> {
    let mut payload =
        MorphologySetPayload::decode(bytes).map_err(|_| RehydrationError::DecodeFailed)?;
    normalize_payload_morphology(&mut payload);
    if payload.encode_to_vec() != bytes {
        return Err(RehydrationError::CanonicalMismatch);
    }
    morphology_from_payload(&payload).map_err(|_| RehydrationError::DecodeFailed)?;
    Ok(payload)
}

pub fn decode_channel_params(bytes: &[u8]) -> Result<ChannelParamsSetPayload, RehydrationError> {
    let mut payload =
        ChannelParamsSetPayload::decode(bytes).map_err(|_| RehydrationError::DecodeFailed)?;
    normalize_payload_channel_params(&mut payload);
    if payload.encode_to_vec() != bytes {
        return Err(RehydrationError::CanonicalMismatch);
    }
    channel_params_from_payload(&payload).map_err(|_| RehydrationError::DecodeFailed)?;
    Ok(payload)
}

pub fn decode_synapse_params(bytes: &[u8]) -> Result<SynapseParamsSetPayload, RehydrationError> {
    let mut payload =
        SynapseParamsSetPayload::decode(bytes).map_err(|_| RehydrationError::DecodeFailed)?;
    normalize_payload_synapse_params(&mut payload);
    if payload.encode_to_vec() != bytes {
        return Err(RehydrationError::CanonicalMismatch);
    }
    synapse_params_from_payload(&payload).map_err(|_| RehydrationError::DecodeFailed)?;
    Ok(payload)
}

pub fn decode_connectivity(bytes: &[u8]) -> Result<ConnectivityGraphPayload, RehydrationError> {
    let mut payload =
        ConnectivityGraphPayload::decode(bytes).map_err(|_| RehydrationError::DecodeFailed)?;
    normalize_payload_connectivity(&mut payload);
    if payload.encode_to_vec() != bytes {
        return Err(RehydrationError::CanonicalMismatch);
    }
    if payload.edges.len() > MAX_EDGES {
        return Err(RehydrationError::DecodeFailed);
    }
    Ok(payload)
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AssetRehydrator;

impl AssetRehydrator {
    pub fn new() -> Self {
        Self
    }

    pub fn verify_bundle(&self, bundle: &AssetBundle) -> Result<(), RehydrationError> {
        verify_asset_bundle(bundle)
    }

    pub fn reassemble(
        &self,
        bundle: &AssetBundle,
        asset_kind: AssetKind,
        asset_digest: [u8; 32],
    ) -> Result<Vec<u8>, RehydrationError> {
        reassemble_asset(bundle, asset_kind, asset_digest)
    }

    pub fn decode_morphology(
        &self,
        bytes: &[u8],
    ) -> Result<MorphologySetPayload, RehydrationError> {
        decode_morphology(bytes)
    }

    pub fn decode_channel_params(
        &self,
        bytes: &[u8],
    ) -> Result<ChannelParamsSetPayload, RehydrationError> {
        decode_channel_params(bytes)
    }

    pub fn decode_synapse_params(
        &self,
        bytes: &[u8],
    ) -> Result<SynapseParamsSetPayload, RehydrationError> {
        decode_synapse_params(bytes)
    }

    pub fn decode_connectivity(
        &self,
        bytes: &[u8],
    ) -> Result<ConnectivityGraphPayload, RehydrationError> {
        decode_connectivity(bytes)
    }
}

fn compute_manifest_digest(manifest: &AssetManifest) -> [u8; 32] {
    let mut normalized = manifest.clone();
    normalized.manifest_digest = vec![0u8; 32];
    let mut hasher = Hasher::new();
    hasher.update(ASSET_MANIFEST_DOMAIN.as_bytes());
    hasher.update(&normalized.encode_to_vec());
    *hasher.finalize().as_bytes()
}

fn compute_bundle_digest(
    manifest: Option<&AssetManifest>,
    chunks: &[AssetChunk],
) -> Result<[u8; 32], RehydrationError> {
    let mut entries = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let asset_digest = digest_from_vec(&chunk.asset_digest, "asset_digest")?;
        let chunk_digest = digest_from_vec(&chunk.chunk_digest, "chunk_digest")?;
        entries.push((asset_digest, chunk.chunk_index, chunk_digest));
    }
    entries.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    let mut hasher = Hasher::new();
    hasher.update(ASSET_BUNDLE_DOMAIN.as_bytes());
    if let Some(manifest) = manifest {
        hasher.update(&manifest.encode_to_vec());
    }
    for (_, _, digest) in entries {
        hasher.update(&digest);
    }
    Ok(*hasher.finalize().as_bytes())
}

fn chunk_digest(payload: &[u8]) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(ASSET_CHUNK_DOMAIN.as_bytes());
    hasher.update(payload);
    *hasher.finalize().as_bytes()
}

fn digest_from_vec(bytes: &[u8], label: &'static str) -> Result<[u8; 32], RehydrationError> {
    if bytes.len() != 32 {
        return Err(RehydrationError::InvalidDigestLength {
            label,
            len: bytes.len(),
        });
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(bytes);
    Ok(out)
}

fn manifest_digest_for_kind(
    manifest: &AssetManifest,
    kind: AssetKind,
) -> Result<[u8; 32], RehydrationError> {
    let component = manifest
        .components
        .iter()
        .find(|component| component.kind == kind as i32)
        .ok_or(RehydrationError::MissingAssetDigest { kind })?;
    digest_from_vec(&component.digest, "asset_digest")
}

#[cfg(test)]
mod tests {
    use super::*;
    use asset_chunker::{
        build_asset_bundle_with_policy, chunk_asset, BundleIdPolicy, ChunkerConfig,
    };
    use biophys_assets::{
        channel_params_from_payload, channel_params_payload_bytes, connectivity_from_payload,
        connectivity_payload_bytes, demo_connectivity, demo_morphology_3comp, demo_syn_params,
        morphology_from_payload, morphology_payload_bytes, synapse_params_from_payload,
        synapse_params_payload_bytes, to_asset_digest,
    };
    use biophys_core::{
        LifParams, LifState, ModChannel, NeuronId, StpParams, StpState, SynapseEdge, STP_SCALE,
    };
    use biophys_runtime::BiophysRuntime;
    use pvgs_client::{MockPvgsReader, PvgsReader};
    use ucf::v1::Compression;

    fn build_bundle() -> AssetBundle {
        let morph = demo_morphology_3comp(3);
        let channel = biophys_assets::demo_channel_params(&morph);
        let syn = demo_syn_params();
        let connectivity = demo_connectivity(3, &syn);
        let morph_bytes = morph.to_canonical_bytes();
        let channel_bytes = channel.to_canonical_bytes();
        let syn_bytes = syn.to_canonical_bytes();
        let connectivity_bytes = connectivity.to_canonical_bytes();
        let morph_digest = morph.digest();
        let channel_digest = channel.digest();
        let syn_digest = syn.digest();
        let conn_digest = connectivity.digest();

        let created_at_ms = 10;
        let manifest_version = 1;
        let mut manifest = AssetManifest {
            manifest_version,
            created_at_ms,
            manifest_digest: vec![0u8; 32],
            components: vec![
                to_asset_digest(
                    AssetKind::MorphologySet,
                    1,
                    morph_digest,
                    created_at_ms,
                    None,
                ),
                to_asset_digest(
                    AssetKind::ChannelParamsSet,
                    1,
                    channel_digest,
                    created_at_ms,
                    None,
                ),
                to_asset_digest(
                    AssetKind::SynapseParamsSet,
                    1,
                    syn_digest,
                    created_at_ms,
                    None,
                ),
                to_asset_digest(
                    AssetKind::ConnectivityGraph,
                    1,
                    conn_digest,
                    created_at_ms,
                    None,
                ),
            ],
        };
        let manifest_digest = compute_manifest_digest(&manifest);
        manifest.manifest_digest = manifest_digest.to_vec();

        let chunker = ChunkerConfig {
            max_chunk_bytes: 64,
            compression: Compression::None,
            max_chunks_total: 128,
            bundle_id_policy: BundleIdPolicy::ManifestDigestPrefix { prefix_len: 8 },
        };
        let mut chunks = Vec::new();
        chunks.extend(
            chunk_asset(
                AssetKind::MorphologySet,
                1,
                morph_digest,
                &morph_bytes,
                &chunker,
                created_at_ms,
            )
            .expect("morph chunks"),
        );
        chunks.extend(
            chunk_asset(
                AssetKind::ChannelParamsSet,
                1,
                channel_digest,
                &channel_bytes,
                &chunker,
                created_at_ms,
            )
            .expect("channel chunks"),
        );
        chunks.extend(
            chunk_asset(
                AssetKind::SynapseParamsSet,
                1,
                syn_digest,
                &syn_bytes,
                &chunker,
                created_at_ms,
            )
            .expect("syn chunks"),
        );
        chunks.extend(
            chunk_asset(
                AssetKind::ConnectivityGraph,
                1,
                conn_digest,
                &connectivity_bytes,
                &chunker,
                created_at_ms,
            )
            .expect("conn chunks"),
        );

        let mut bundle = build_asset_bundle_with_policy(
            manifest,
            chunks,
            created_at_ms,
            BundleIdPolicy::ManifestDigestPrefix { prefix_len: 8 },
        );
        let digest =
            compute_bundle_digest(bundle.manifest.as_ref(), &bundle.chunks).expect("bundle digest");
        bundle.bundle_digest = digest.to_vec();
        bundle
    }

    #[test]
    fn verify_valid_bundle() {
        let bundle = build_bundle();
        verify_asset_bundle(&bundle).expect("bundle verify");
    }

    #[test]
    fn verify_fails_on_chunk_digest_mismatch() {
        let mut bundle = build_bundle();
        bundle.chunks[0].payload.push(0);
        let err = verify_asset_bundle(&bundle).expect_err("should fail");
        assert!(matches!(err, RehydrationError::ChunkDigestMismatch { .. }));
    }

    #[test]
    fn reassemble_returns_original_bytes() {
        let bundle = build_bundle();
        let morph = demo_morphology_3comp(3);
        let bytes =
            reassemble_asset(&bundle, AssetKind::MorphologySet, morph.digest()).expect("rehydrate");
        assert_eq!(bytes, morph.to_canonical_bytes());
    }

    #[test]
    fn load_payloads_roundtrip_ok() {
        let bundle = build_bundle();
        let morph_payload = load_morphology_payload(&bundle).expect("morph payload");
        let channel_payload = load_channel_params_payload(&bundle).expect("channel payload");
        let syn_payload = load_synapse_params_payload(&bundle).expect("syn payload");
        let conn_payload = load_connectivity_payload(&bundle).expect("conn payload");

        let morph = morphology_from_payload(&morph_payload).expect("morph conversion");
        let channel = channel_params_from_payload(&channel_payload).expect("channel conversion");
        let syn = synapse_params_from_payload(&syn_payload).expect("syn conversion");
        let conn = connectivity_from_payload(&conn_payload, &syn_payload).expect("conn conversion");

        assert_eq!(morph, demo_morphology_3comp(3));
        assert_eq!(channel, biophys_assets::demo_channel_params(&morph));
        assert_eq!(syn, demo_syn_params());
        assert_eq!(conn, demo_connectivity(3, &syn));
    }

    #[test]
    fn decode_roundtrip_matches_bytes() {
        let morph = demo_morphology_3comp(2);
        let bytes = morph.to_canonical_bytes();
        let decoded = decode_morphology(&bytes).expect("decode");
        assert_eq!(morphology_payload_bytes(&decoded), bytes);
        let roundtrip = morphology_from_payload(&decoded).expect("morph payload");
        assert_eq!(roundtrip, morph);

        let channel = biophys_assets::demo_channel_params(&morph);
        let bytes = channel.to_canonical_bytes();
        let decoded = decode_channel_params(&bytes).expect("decode channel");
        assert_eq!(channel_params_payload_bytes(&decoded), bytes);
        let roundtrip = channel_params_from_payload(&decoded).expect("channel payload");
        assert_eq!(roundtrip, channel);

        let syn = demo_syn_params();
        let bytes = syn.to_canonical_bytes();
        let decoded = decode_synapse_params(&bytes).expect("decode syn");
        assert_eq!(synapse_params_payload_bytes(&decoded), bytes);
        let roundtrip = synapse_params_from_payload(&decoded).expect("syn payload");
        assert_eq!(roundtrip, syn);

        let syn = demo_syn_params();
        let connectivity = demo_connectivity(2, &syn);
        let bytes = connectivity.to_canonical_bytes();
        let decoded = decode_connectivity(&bytes).expect("decode");
        assert_eq!(connectivity_payload_bytes(&decoded), bytes);
        let syn_payload = decode_synapse_params(&syn.to_canonical_bytes()).expect("syn payload");
        let roundtrip = connectivity_from_payload(&decoded, &syn_payload).expect("conn payload");
        assert_eq!(roundtrip, connectivity);
    }

    #[test]
    fn pvgs_rehydration_initializes_runtime() {
        let bundle = build_bundle();
        let mut reader = MockPvgsReader::with_asset_bundle(bundle);
        let bundle = reader
            .get_latest_asset_bundle()
            .expect("bundle fetch")
            .expect("bundle");
        verify_asset_bundle(&bundle).expect("verify");

        let morph_digest = demo_morphology_3comp(3).digest();
        let syn = demo_syn_params();
        let syn_digest = syn.digest();
        let conn_digest = demo_connectivity(3, &syn).digest();
        let morph_bytes =
            reassemble_asset(&bundle, AssetKind::MorphologySet, morph_digest).expect("morph bytes");
        let syn_bytes =
            reassemble_asset(&bundle, AssetKind::SynapseParamsSet, syn_digest).expect("syn bytes");
        let conn_bytes = reassemble_asset(&bundle, AssetKind::ConnectivityGraph, conn_digest)
            .expect("conn bytes");

        let morph_payload = decode_morphology(&morph_bytes).expect("decode morph");
        let conn_payload = decode_connectivity(&conn_bytes).expect("decode conn");
        let syn_payload = decode_synapse_params(&syn_bytes).expect("decode syn");

        let morph = morphology_from_payload(&morph_payload).expect("morph payload");
        let conn =
            connectivity_from_payload(&conn_payload, &syn_payload).expect("connectivity payload");

        let neuron_count = morph.neurons.len();
        let params = vec![
            LifParams {
                tau_ms: 10,
                v_rest: -65,
                v_reset: -65,
                v_threshold: -50,
            };
            neuron_count
        ];
        let states = vec![
            LifState {
                v: -65,
                refractory_steps: 0,
            };
            neuron_count
        ];
        let mut edges = Vec::new();
        let mut stp_params = Vec::new();
        for edge in conn.edges {
            edges.push(SynapseEdge {
                pre: NeuronId(edge.pre),
                post: NeuronId(edge.post),
                weight_base: match edge.syn_type {
                    biophys_assets::SynType::Exc => 5,
                    biophys_assets::SynType::Inh => -5,
                },
                weight_effective: 0,
                delay_steps: edge.delay_steps,
                mod_channel: ModChannel::None,
                stp: StpState {
                    x: STP_SCALE,
                    u: STP_SCALE / 2,
                },
            });
            stp_params.push(StpParams {
                u: 200,
                tau_rec_steps: 10,
                tau_fac_steps: 10,
                mod_channel: None,
            });
        }
        let runtime =
            BiophysRuntime::new_with_synapses(params, states, 1, 16, edges, stp_params, 1024);
        assert_eq!(runtime.params.len(), neuron_count);
        assert_eq!(runtime.edges.len(), 3);
    }

    #[test]
    fn boundedness_rejects_payload_sizes() {
        let mut morph_payload = MorphologySetPayload {
            version: 1,
            neurons: vec![
                ucf::v1::MorphNeuronPayload {
                    neuron_id: 0,
                    compartments: vec![ucf::v1::CompartmentPayload {
                        comp_id: 0,
                        parent: None,
                        kind: ucf::v1::CompartmentKind::Soma as i32,
                        length_um: 10,
                        diameter_um: 8,
                    }],
                    labels: Vec::new(),
                };
                biophys_assets::MAX_NEURONS + 1
            ],
        };
        morph_payload
            .neurons
            .iter_mut()
            .enumerate()
            .for_each(|(idx, neuron)| neuron.neuron_id = idx as u32);
        let bytes = morph_payload.encode_to_vec();
        let err = decode_morphology(&bytes).expect_err("expected morph bounds error");
        assert!(matches!(err, RehydrationError::DecodeFailed));

        let conn_payload = ConnectivityGraphPayload {
            version: 1,
            edges: (0..(MAX_EDGES as u32 + 1))
                .map(|_idx| ucf::v1::ConnectivityEdgePayload {
                    pre: 0,
                    post: 0,
                    post_compartment: 0,
                    syn_param_id: 0,
                    delay_steps: 1,
                })
                .collect(),
        };
        let bytes = conn_payload.encode_to_vec();
        let err = decode_connectivity(&bytes).expect_err("expected conn bounds error");
        assert!(matches!(err, RehydrationError::DecodeFailed));
    }

    #[test]
    fn asset_too_large_rejected() {
        let payload = vec![1u8; MAX_ASSET_BYTES + 1];
        let digest = [7u8; 32];
        let chunk = AssetChunk {
            kind: AssetKind::MorphologySet as i32,
            version: 1,
            asset_digest: digest.to_vec(),
            chunk_index: 0,
            chunk_count: 1,
            compression: Compression::None as i32,
            created_at_ms: 0,
            chunk_digest: chunk_digest(&payload).to_vec(),
            payload,
        };
        let bundle = AssetBundle {
            bundle_digest: vec![0u8; 32],
            bundle_id: String::new(),
            created_at_ms: 0,
            manifest: None,
            chunks: vec![chunk],
        };
        let err =
            reassemble_asset(&bundle, AssetKind::MorphologySet, digest).expect_err("too large");
        assert!(matches!(err, RehydrationError::AssetTooLarge));
    }
}
