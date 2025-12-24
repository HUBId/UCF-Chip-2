#![forbid(unsafe_code)]

use hex::encode as hex_encode;
use pvgs::PvgsStore;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AssetsCard {
    pub latest_manifest_digest: Option<[u8; 32]>,
    pub morph_digest: Option<[u8; 32]>,
    pub channel_params_digest: Option<[u8; 32]>,
    pub syn_params_digest: Option<[u8; 32]>,
    pub connectivity_digest: Option<[u8; 32]>,
}

impl AssetsCard {
    pub fn from_store(store: &PvgsStore) -> Self {
        let latest = query::get_latest_asset_manifest(store);
        Self {
            latest_manifest_digest: latest.map(|manifest| manifest.manifest_digest),
            morph_digest: latest.map(|manifest| manifest.morph_digest),
            channel_params_digest: latest.map(|manifest| manifest.channel_params_digest),
            syn_params_digest: latest.map(|manifest| manifest.syn_params_digest),
            connectivity_digest: latest.map(|manifest| manifest.connectivity_digest),
        }
    }

    pub fn render_cli(&self) -> String {
        let lines = [
            format!(
                "latest_manifest_digest: {}",
                format_digest(self.latest_manifest_digest)
            ),
            format!("morph_digest: {}", format_digest(self.morph_digest)),
            format!(
                "channel_params_digest: {}",
                format_digest(self.channel_params_digest)
            ),
            format!("syn_params_digest: {}", format_digest(self.syn_params_digest)),
            format!(
                "connectivity_digest: {}",
                format_digest(self.connectivity_digest)
            ),
        ];
        lines.join("\n")
    }
}

fn format_digest(digest: Option<[u8; 32]>) -> String {
    digest
        .map(hex_encode)
        .unwrap_or_else(|| "NONE".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use assets::ASSET_MANIFEST_EVIDENCE_LEN;
    use pvgs::{compute_manifest_digest, AssetManifestAppend};

    fn build_payload(created_at_ms: u64) -> Vec<u8> {
        let mut payload = Vec::with_capacity(ASSET_MANIFEST_EVIDENCE_LEN);
        payload.extend_from_slice(&1u32.to_le_bytes());
        payload.extend_from_slice(&created_at_ms.to_le_bytes());
        payload.extend_from_slice(&[0u8; 32]);
        payload.extend_from_slice(&[2u8; 32]);
        payload.extend_from_slice(&[3u8; 32]);
        payload.extend_from_slice(&[4u8; 32]);
        payload.extend_from_slice(&[5u8; 32]);
        let digest = compute_manifest_digest(&payload);
        payload[12..44].copy_from_slice(&digest);
        payload
    }

    #[test]
    fn assets_card_includes_latest_manifest() {
        let mut store = PvgsStore::new();
        let payload = build_payload(10);
        store
            .commit_asset_manifest(AssetManifestAppend {
                payload: payload.clone(),
                payload_digest: compute_manifest_digest(&payload),
            })
            .unwrap();

        let card = AssetsCard::from_store(&store);
        assert_eq!(card.latest_manifest_digest, Some(compute_manifest_digest(&payload)));
        assert_eq!(card.morph_digest, Some([2u8; 32]));
        assert_eq!(card.connectivity_digest, Some([5u8; 32]));

        let cli_output = card.render_cli();
        assert!(cli_output.contains("latest_manifest_digest"));
    }
}
