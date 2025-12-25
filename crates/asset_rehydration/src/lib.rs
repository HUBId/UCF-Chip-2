#![forbid(unsafe_code)]

use biophys_assets::{ChannelParamsSet, ConnectivityGraph, MorphologySet, SynapseParamsSet};
use blake3::Hasher;
use prost::Message;
use thiserror::Error;
use ucf::v1::{AssetBundle, AssetChunk, AssetKind, AssetManifest, Compression};

pub const ASSET_CHUNK_DOMAIN: &str = "UCF:ASSET:CHUNK";
pub const ASSET_BUNDLE_DOMAIN: &str = "UCF:ASSET:BUNDLE";
pub const ASSET_MANIFEST_DOMAIN: &str = "UCF:ASSET:MANIFEST";

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum RehydrationError {
    #[error("invalid digest length for {label}: {len}")]
    InvalidDigestLength { label: &'static str, len: usize },
    #[error("chunk digest mismatch at index {index}")]
    ChunkDigestMismatch { index: usize },
    #[error("bundle digest mismatch")]
    BundleDigestMismatch,
    #[error("manifest digest mismatch")]
    ManifestDigestMismatch,
    #[error("missing chunks for asset")]
    MissingChunks,
    #[error("chunk index {index} out of range {count}")]
    ChunkIndexOutOfRange { index: u32, count: u32 },
    #[error("duplicate chunk index {index}")]
    DuplicateChunkIndex { index: u32 },
    #[error("missing chunk index {index}")]
    MissingChunkIndex { index: u32 },
    #[error("chunk count mismatch")]
    ChunkCountMismatch,
    #[error("asset version mismatch")]
    AssetVersionMismatch,
    #[error("unsupported compression {compression:?}")]
    UnsupportedCompression { compression: Compression },
    #[error("decode failed: {message}")]
    DecodeFailed { message: String },
    #[error("canonical bytes mismatch after decode")]
    CanonicalMismatch,
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
        let slot = &mut ordered[chunk.chunk_index as usize];
        if slot.is_some() {
            return Err(RehydrationError::DuplicateChunkIndex {
                index: chunk.chunk_index,
            });
        }
        *slot = Some(chunk);
    }

    let mut payload = Vec::new();
    for (idx, slot) in ordered.into_iter().enumerate() {
        let chunk = slot.ok_or(RehydrationError::MissingChunkIndex { index: idx as u32 })?;
        let compression = Compression::try_from(chunk.compression).unwrap_or(Compression::Unknown);
        match compression {
            Compression::None => payload.extend_from_slice(&chunk.payload),
            Compression::Zstd | Compression::Unknown => {
                return Err(RehydrationError::UnsupportedCompression { compression });
            }
        }
    }

    Ok(payload)
}

pub fn decode_morphology(bytes: &[u8]) -> Result<MorphologySet, RehydrationError> {
    let mut cursor = Cursor::new(bytes);
    let version = cursor.take_u32()?;
    let neuron_count = cursor.take_u32()?;
    let mut neurons = Vec::with_capacity(neuron_count as usize);
    for _ in 0..neuron_count {
        let neuron_id = cursor.take_u32()?;
        let comp_count = cursor.take_u32()?;
        let mut compartments = Vec::with_capacity(comp_count as usize);
        for _ in 0..comp_count {
            let comp_id = cursor.take_u32()?;
            let has_parent = cursor.take_u8()?;
            let parent = match has_parent {
                0 => None,
                1 => Some(cursor.take_u32()?),
                _ => {
                    return Err(RehydrationError::DecodeFailed {
                        message: "invalid parent flag".to_string(),
                    })
                }
            };
            let kind = match cursor.take_u8()? {
                0 => biophys_assets::CompartmentKind::Soma,
                1 => biophys_assets::CompartmentKind::Dendrite,
                2 => biophys_assets::CompartmentKind::Axon,
                other => {
                    return Err(RehydrationError::DecodeFailed {
                        message: format!("invalid compartment kind {other}"),
                    })
                }
            };
            let length_um = cursor.take_u16()?;
            let diameter_um = cursor.take_u16()?;
            compartments.push(biophys_assets::Compartment {
                comp_id,
                parent,
                kind,
                length_um,
                diameter_um,
            });
        }
        neurons.push(biophys_assets::MorphNeuron {
            neuron_id,
            compartments,
        });
    }
    cursor.ensure_consumed()?;
    let morph = MorphologySet { version, neurons };
    if morph.to_canonical_bytes() != bytes {
        return Err(RehydrationError::CanonicalMismatch);
    }
    Ok(morph)
}

pub fn decode_channel_params(bytes: &[u8]) -> Result<ChannelParamsSet, RehydrationError> {
    let mut cursor = Cursor::new(bytes);
    let version = cursor.take_u32()?;
    let param_count = cursor.take_u32()?;
    let mut per_compartment = Vec::with_capacity(param_count as usize);
    for _ in 0..param_count {
        let neuron_id = cursor.take_u32()?;
        let comp_id = cursor.take_u32()?;
        let leak_g = cursor.take_u16()?;
        let na_g = cursor.take_u16()?;
        let k_g = cursor.take_u16()?;
        per_compartment.push(biophys_assets::ChannelParams {
            neuron_id,
            comp_id,
            leak_g,
            na_g,
            k_g,
        });
    }
    cursor.ensure_consumed()?;
    let params = ChannelParamsSet {
        version,
        per_compartment,
    };
    if params.to_canonical_bytes() != bytes {
        return Err(RehydrationError::CanonicalMismatch);
    }
    Ok(params)
}

pub fn decode_synapse_params(bytes: &[u8]) -> Result<SynapseParamsSet, RehydrationError> {
    let mut cursor = Cursor::new(bytes);
    let version = cursor.take_u32()?;
    let param_count = cursor.take_u32()?;
    let mut params = Vec::with_capacity(param_count as usize);
    for _ in 0..param_count {
        let syn_type = match cursor.take_u8()? {
            0 => biophys_assets::SynType::Exc,
            1 => biophys_assets::SynType::Inh,
            other => {
                return Err(RehydrationError::DecodeFailed {
                    message: format!("invalid synapse type {other}"),
                })
            }
        };
        let weight_base = cursor.take_i32()?;
        let stp_u = cursor.take_u16()?;
        let tau_rec = cursor.take_u16()?;
        let tau_fac = cursor.take_u16()?;
        let mod_channel = match cursor.take_u8()? {
            0 => biophys_assets::ModChannel::None,
            1 => biophys_assets::ModChannel::A,
            2 => biophys_assets::ModChannel::B,
            other => {
                return Err(RehydrationError::DecodeFailed {
                    message: format!("invalid mod channel {other}"),
                })
            }
        };
        params.push(biophys_assets::SynapseParams {
            syn_type,
            weight_base,
            stp_u,
            tau_rec,
            tau_fac,
            mod_channel,
        });
    }
    cursor.ensure_consumed()?;
    let params = SynapseParamsSet { version, params };
    if params.to_canonical_bytes() != bytes {
        return Err(RehydrationError::CanonicalMismatch);
    }
    Ok(params)
}

pub fn decode_connectivity(bytes: &[u8]) -> Result<ConnectivityGraph, RehydrationError> {
    let mut cursor = Cursor::new(bytes);
    let version = cursor.take_u32()?;
    let edge_count = cursor.take_u32()?;
    let mut edges = Vec::with_capacity(edge_count as usize);
    for _ in 0..edge_count {
        let pre = cursor.take_u32()?;
        let post = cursor.take_u32()?;
        let syn_type = match cursor.take_u8()? {
            0 => biophys_assets::SynType::Exc,
            1 => biophys_assets::SynType::Inh,
            other => {
                return Err(RehydrationError::DecodeFailed {
                    message: format!("invalid synapse type {other}"),
                })
            }
        };
        let delay_steps = cursor.take_u16()?;
        let syn_param_id = cursor.take_u32()?;
        edges.push(biophys_assets::ConnEdge {
            pre,
            post,
            syn_type,
            delay_steps,
            syn_param_id,
        });
    }
    cursor.ensure_consumed()?;
    let graph = ConnectivityGraph { version, edges };
    if graph.to_canonical_bytes() != bytes {
        return Err(RehydrationError::CanonicalMismatch);
    }
    Ok(graph)
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

    pub fn decode_morphology(&self, bytes: &[u8]) -> Result<MorphologySet, RehydrationError> {
        decode_morphology(bytes)
    }

    pub fn decode_channel_params(
        &self,
        bytes: &[u8],
    ) -> Result<ChannelParamsSet, RehydrationError> {
        decode_channel_params(bytes)
    }

    pub fn decode_synapse_params(
        &self,
        bytes: &[u8],
    ) -> Result<SynapseParamsSet, RehydrationError> {
        decode_synapse_params(bytes)
    }

    pub fn decode_connectivity(&self, bytes: &[u8]) -> Result<ConnectivityGraph, RehydrationError> {
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

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn ensure_consumed(&self) -> Result<(), RehydrationError> {
        if self.offset == self.bytes.len() {
            Ok(())
        } else {
            Err(RehydrationError::DecodeFailed {
                message: format!(
                    "trailing bytes: {} remaining",
                    self.bytes.len().saturating_sub(self.offset)
                ),
            })
        }
    }

    fn take_u8(&mut self) -> Result<u8, RehydrationError> {
        if self.offset + 1 > self.bytes.len() {
            return Err(RehydrationError::DecodeFailed {
                message: "unexpected end of bytes".to_string(),
            });
        }
        let value = self.bytes[self.offset];
        self.offset += 1;
        Ok(value)
    }

    fn take_u16(&mut self) -> Result<u16, RehydrationError> {
        let bytes = self.take_exact(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn take_u32(&mut self) -> Result<u32, RehydrationError> {
        let bytes = self.take_exact(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn take_i32(&mut self) -> Result<i32, RehydrationError> {
        let bytes = self.take_exact(4)?;
        Ok(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn take_exact(&mut self, len: usize) -> Result<&'a [u8], RehydrationError> {
        if self.offset + len > self.bytes.len() {
            return Err(RehydrationError::DecodeFailed {
                message: "unexpected end of bytes".to_string(),
            });
        }
        let start = self.offset;
        self.offset += len;
        Ok(&self.bytes[start..start + len])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use asset_chunker::{
        build_asset_bundle_with_policy, chunk_asset, BundleIdPolicy, ChunkerConfig,
    };
    use biophys_assets::{
        demo_connectivity, demo_morphology_3comp, demo_syn_params, to_asset_digest,
    };
    use biophys_core::{
        LifParams, LifState, ModChannel, NeuronId, StpParams, StpState, SynapseEdge, STP_SCALE,
    };
    use biophys_runtime::BiophysRuntime;
    use pvgs_client::{MockPvgsReader, PvgsReader};
    use ucf::v1::Compression;

    fn build_bundle() -> AssetBundle {
        let morph = demo_morphology_3comp(3);
        let syn = demo_syn_params();
        let connectivity = demo_connectivity(3, &syn);
        let morph_bytes = morph.to_canonical_bytes();
        let connectivity_bytes = connectivity.to_canonical_bytes();
        let morph_digest = morph.digest();
        let conn_digest = connectivity.digest();

        let created_at_ms = 10;
        let manifest_version = 1;
        let mut manifest = AssetManifest {
            manifest_version,
            created_at_ms,
            manifest_digest: vec![0u8; 32],
            components: vec![
                to_asset_digest(AssetKind::Morphology, 1, morph_digest, created_at_ms, None),
                to_asset_digest(AssetKind::Connectivity, 1, conn_digest, created_at_ms, None),
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
                AssetKind::Morphology,
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
                AssetKind::Connectivity,
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
            reassemble_asset(&bundle, AssetKind::Morphology, morph.digest()).expect("rehydrate");
        assert_eq!(bytes, morph.to_canonical_bytes());
    }

    #[test]
    fn decode_roundtrip_matches_bytes() {
        let morph = demo_morphology_3comp(2);
        let bytes = morph.to_canonical_bytes();
        let decoded = decode_morphology(&bytes).expect("decode");
        assert_eq!(decoded, morph);

        let channel = biophys_assets::demo_channel_params(&morph);
        let bytes = channel.to_canonical_bytes();
        let decoded = decode_channel_params(&bytes).expect("decode channel");
        assert_eq!(decoded, channel);

        let syn = demo_syn_params();
        let bytes = syn.to_canonical_bytes();
        let decoded = decode_synapse_params(&bytes).expect("decode syn");
        assert_eq!(decoded, syn);

        let syn = demo_syn_params();
        let connectivity = demo_connectivity(2, &syn);
        let bytes = connectivity.to_canonical_bytes();
        let decoded = decode_connectivity(&bytes).expect("decode");
        assert_eq!(decoded, connectivity);
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
        let conn_digest = demo_connectivity(3, &demo_syn_params()).digest();
        let morph_bytes =
            reassemble_asset(&bundle, AssetKind::Morphology, morph_digest).expect("morph bytes");
        let conn_bytes =
            reassemble_asset(&bundle, AssetKind::Connectivity, conn_digest).expect("conn bytes");

        let morph = decode_morphology(&morph_bytes).expect("decode morph");
        let conn = decode_connectivity(&conn_bytes).expect("decode conn");

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
}
