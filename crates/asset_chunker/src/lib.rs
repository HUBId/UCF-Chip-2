#![forbid(unsafe_code)]

use blake3::Hasher;
use prost::Message;
use ucf::v1::{AssetBundle, AssetChunk, AssetKind, AssetManifest, Compression};

pub const ASSET_CHUNK_DOMAIN: &str = "UCF:ASSET:CHUNK";
pub const ASSET_BUNDLE_DOMAIN: &str = "UCF:ASSET:BUNDLE";
const DEFAULT_BUNDLE_ID_PREFIX_LEN: usize = 12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BundleIdPolicy {
    ManifestDigestPrefix { prefix_len: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkerConfig {
    pub max_chunk_bytes: usize,
    pub compression: Compression,
    pub max_chunks_total: u32,
    pub bundle_id_policy: BundleIdPolicy,
}

impl ChunkerConfig {
    pub fn default_deterministic() -> Self {
        Self {
            max_chunk_bytes: 64 * 1024,
            compression: Compression::None,
            max_chunks_total: 4096,
            bundle_id_policy: BundleIdPolicy::ManifestDigestPrefix {
                prefix_len: DEFAULT_BUNDLE_ID_PREFIX_LEN,
            },
        }
    }
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum ChunkerError {
    #[error("payload requires {required} chunks exceeding limit {limit}")]
    ChunkLimitExceeded { required: u32, limit: u32 },
    #[error("unsupported compression")]
    UnsupportedCompression,
    #[error("invalid max_chunk_bytes")]
    InvalidChunkSize,
}

pub fn chunk_asset(
    kind: AssetKind,
    version: u32,
    asset_digest: [u8; 32],
    bytes: &[u8],
    cfg: &ChunkerConfig,
    created_at_ms: u64,
) -> Result<Vec<AssetChunk>, ChunkerError> {
    if cfg.max_chunk_bytes == 0 {
        return Err(ChunkerError::InvalidChunkSize);
    }

    let chunk_count = chunk_count(bytes.len(), cfg.max_chunk_bytes);
    if chunk_count > cfg.max_chunks_total {
        return Err(ChunkerError::ChunkLimitExceeded {
            required: chunk_count,
            limit: cfg.max_chunks_total,
        });
    }

    let mut chunks = Vec::with_capacity(chunk_count as usize);
    if bytes.is_empty() {
        let payload = compress_payload(bytes, cfg.compression)?;
        let digest = chunk_digest(&payload);
        chunks.push(AssetChunk {
            kind: kind as i32,
            version,
            asset_digest: asset_digest.to_vec(),
            chunk_index: 0,
            chunk_count: 1,
            chunk_digest: digest.to_vec(),
            compression: cfg.compression as i32,
            payload,
            created_at_ms,
        });
        return Ok(chunks);
    }

    for (index, raw) in bytes.chunks(cfg.max_chunk_bytes).enumerate() {
        let payload = compress_payload(raw, cfg.compression)?;
        let digest = chunk_digest(&payload);
        chunks.push(AssetChunk {
            kind: kind as i32,
            version,
            asset_digest: asset_digest.to_vec(),
            chunk_index: index as u32,
            chunk_count,
            chunk_digest: digest.to_vec(),
            compression: cfg.compression as i32,
            payload,
            created_at_ms,
        });
    }

    Ok(chunks)
}

pub fn build_asset_bundle(
    manifest: AssetManifest,
    chunks: Vec<AssetChunk>,
    created_at_ms: u64,
) -> AssetBundle {
    build_asset_bundle_with_policy(
        manifest,
        chunks,
        created_at_ms,
        BundleIdPolicy::ManifestDigestPrefix {
            prefix_len: DEFAULT_BUNDLE_ID_PREFIX_LEN,
        },
    )
}

pub fn build_asset_bundle_with_policy(
    manifest: AssetManifest,
    chunks: Vec<AssetChunk>,
    created_at_ms: u64,
    bundle_id_policy: BundleIdPolicy,
) -> AssetBundle {
    let bundle_id = bundle_id_from_manifest(&manifest, bundle_id_policy);
    let bundle_digest = compute_bundle_digest(&manifest, &chunks);

    AssetBundle {
        bundle_id,
        created_at_ms,
        bundle_digest: bundle_digest.to_vec(),
        manifest: Some(manifest),
        chunks,
    }
}

pub fn compute_bundle_digest(manifest: &AssetManifest, chunks: &[AssetChunk]) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(ASSET_BUNDLE_DOMAIN.as_bytes());
    hasher.update(&manifest.encode_to_vec());
    for chunk in chunks {
        hasher.update(&chunk.chunk_digest);
    }
    *hasher.finalize().as_bytes()
}

fn chunk_count(total_bytes: usize, max_chunk_bytes: usize) -> u32 {
    let full = total_bytes / max_chunk_bytes;
    let remainder = total_bytes % max_chunk_bytes;
    let mut count = full as u32;
    if remainder > 0 || total_bytes == 0 {
        count += 1;
    }
    count
}

fn compress_payload(payload: &[u8], compression: Compression) -> Result<Vec<u8>, ChunkerError> {
    match compression {
        Compression::None => Ok(payload.to_vec()),
        Compression::Unknown | Compression::Zstd => Err(ChunkerError::UnsupportedCompression),
    }
}

fn chunk_digest(payload: &[u8]) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(ASSET_CHUNK_DOMAIN.as_bytes());
    hasher.update(payload);
    *hasher.finalize().as_bytes()
}

fn bundle_id_from_manifest(manifest: &AssetManifest, policy: BundleIdPolicy) -> String {
    let digest = manifest.manifest_digest.as_slice();
    let prefix_len = match policy {
        BundleIdPolicy::ManifestDigestPrefix { prefix_len } => prefix_len,
    };
    let mut hex = bytes_to_hex(digest);
    if hex.len() > prefix_len {
        hex.truncate(prefix_len);
    }
    format!("bundle:{hex}")
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_manifest() -> AssetManifest {
        AssetManifest {
            manifest_version: 1,
            created_at_ms: 10,
            manifest_digest: vec![9u8; 32],
            components: Vec::new(),
        }
    }

    #[test]
    fn chunking_is_deterministic() {
        let cfg = ChunkerConfig {
            max_chunk_bytes: 4,
            compression: Compression::None,
            max_chunks_total: 16,
            bundle_id_policy: BundleIdPolicy::ManifestDigestPrefix { prefix_len: 8 },
        };
        let bytes = b"deterministic-chunk";
        let chunks_a = chunk_asset(AssetKind::Morphology, 1, [1u8; 32], bytes, &cfg, 5).unwrap();
        let chunks_b = chunk_asset(AssetKind::Morphology, 1, [1u8; 32], bytes, &cfg, 5).unwrap();

        assert_eq!(chunks_a, chunks_b);
        let digests_a: Vec<_> = chunks_a
            .iter()
            .map(|chunk| chunk.chunk_digest.clone())
            .collect();
        let digests_b: Vec<_> = chunks_b
            .iter()
            .map(|chunk| chunk.chunk_digest.clone())
            .collect();
        assert_eq!(digests_a, digests_b);
    }

    #[test]
    fn bundle_digest_is_deterministic() {
        let manifest = sample_manifest();
        let cfg = ChunkerConfig {
            max_chunk_bytes: 8,
            compression: Compression::None,
            max_chunks_total: 16,
            bundle_id_policy: BundleIdPolicy::ManifestDigestPrefix { prefix_len: 8 },
        };
        let chunks =
            chunk_asset(AssetKind::Morphology, 1, [2u8; 32], b"payload", &cfg, 10).unwrap();

        let bundle_a = build_asset_bundle(manifest.clone(), chunks.clone(), 10);
        let bundle_b = build_asset_bundle(manifest, chunks, 10);

        assert_eq!(bundle_a.bundle_digest, bundle_b.bundle_digest);
    }

    #[test]
    fn boundedness_rejects_oversized_payload() {
        let cfg = ChunkerConfig {
            max_chunk_bytes: 1,
            compression: Compression::None,
            max_chunks_total: 2,
            bundle_id_policy: BundleIdPolicy::ManifestDigestPrefix { prefix_len: 8 },
        };

        let err = chunk_asset(AssetKind::Morphology, 1, [3u8; 32], b"too", &cfg, 10).unwrap_err();

        assert!(matches!(err, ChunkerError::ChunkLimitExceeded { .. }));
    }
}
