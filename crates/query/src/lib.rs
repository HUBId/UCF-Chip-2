#![forbid(unsafe_code)]

use assets::AssetManifestEvidence;
use pvgs::PvgsStore;

pub fn get_latest_asset_manifest(store: &PvgsStore) -> Option<AssetManifestEvidence> {
    list_asset_manifests(store).into_iter().next_back()
}

pub fn list_asset_manifests(store: &PvgsStore) -> Vec<AssetManifestEvidence> {
    let mut manifests = store.asset_store().manifests().to_vec();
    manifests.sort_by(|a, b| {
        a.created_at_ms
            .cmp(&b.created_at_ms)
            .then_with(|| a.manifest_digest.cmp(&b.manifest_digest))
    });
    manifests
}

pub fn get_asset_manifest(
    store: &PvgsStore,
    manifest_digest: [u8; 32],
) -> Option<AssetManifestEvidence> {
    store.asset_store().get(manifest_digest)
}

pub fn get_latest_asset_manifest_digest(store: &PvgsStore) -> Option<[u8; 32]> {
    get_latest_asset_manifest(store).map(|manifest| manifest.manifest_digest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use assets::ASSET_MANIFEST_EVIDENCE_LEN;
    use pvgs::{compute_manifest_digest, AssetManifestAppend, PvgsStore};

    fn build_payload(created_at_ms: u64, marker: u8) -> Vec<u8> {
        let mut payload = Vec::with_capacity(ASSET_MANIFEST_EVIDENCE_LEN);
        payload.extend_from_slice(&1u32.to_le_bytes());
        payload.extend_from_slice(&created_at_ms.to_le_bytes());
        payload.extend_from_slice(&[0u8; 32]);
        payload.extend_from_slice(&[marker; 32]);
        payload.extend_from_slice(&[3u8; 32]);
        payload.extend_from_slice(&[4u8; 32]);
        payload.extend_from_slice(&[5u8; 32]);
        let digest = compute_manifest_digest(&payload);
        payload[12..44].copy_from_slice(&digest);
        payload
    }

    #[test]
    fn queries_latest_asset_manifest() {
        let mut store = PvgsStore::new();
        let payload_a = build_payload(5, 2);
        let payload_b = build_payload(10, 3);

        store
            .commit_asset_manifest(AssetManifestAppend {
                payload: payload_a.clone(),
                payload_digest: compute_manifest_digest(&payload_a),
            })
            .unwrap();
        store
            .commit_asset_manifest(AssetManifestAppend {
                payload: payload_b.clone(),
                payload_digest: compute_manifest_digest(&payload_b),
            })
            .unwrap();

        let latest = get_latest_asset_manifest(&store).unwrap();
        assert_eq!(latest.manifest_digest, compute_manifest_digest(&payload_b));
    }

    #[test]
    fn list_is_deterministic() {
        let mut store = PvgsStore::new();
        let payload_a = build_payload(5, 2);
        let payload_b = build_payload(5, 1);

        store
            .commit_asset_manifest(AssetManifestAppend {
                payload: payload_a.clone(),
                payload_digest: compute_manifest_digest(&payload_a),
            })
            .unwrap();
        store
            .commit_asset_manifest(AssetManifestAppend {
                payload: payload_b.clone(),
                payload_digest: compute_manifest_digest(&payload_b),
            })
            .unwrap();

        let manifests = list_asset_manifests(&store);
        let digest_a = compute_manifest_digest(&payload_a);
        let digest_b = compute_manifest_digest(&payload_b);
        let (first, second) = if digest_a <= digest_b {
            (digest_a, digest_b)
        } else {
            (digest_b, digest_a)
        };
        assert_eq!(manifests[0].manifest_digest, first);
        assert_eq!(manifests[1].manifest_digest, second);
    }
}
