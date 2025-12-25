#![forbid(unsafe_code)]

pub mod pvgs {
    use prost::Message;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    use ucf::v1::{
        AssetBundle, AssetManifest, CharacterBaselineVector, MicrocircuitConfigAppend,
        PolicyEcologyVector, PvgsReceipt,
    };

    #[derive(Clone, PartialEq, Message)]
    pub struct Digest32 {
        #[prost(bytes, tag = "1")]
        pub value: Vec<u8>,
    }

    impl Digest32 {
        pub fn from_array(bytes: [u8; 32]) -> Self {
            Self {
                value: bytes.to_vec(),
            }
        }

        pub fn as_slice(&self) -> &[u8] {
            &self.value
        }
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct Cbv {
        #[prost(uint64, tag = "1")]
        pub epoch: u64,
        #[prost(message, optional, tag = "2")]
        pub cbv_digest: Option<Digest32>,
        #[prost(bytes, optional, tag = "3")]
        pub proof_receipt_ref: Option<Vec<u8>>,
        #[prost(bytes, optional, tag = "4")]
        pub signature: Option<Vec<u8>>,
        #[prost(message, optional, tag = "5")]
        pub cbv: Option<CharacterBaselineVector>,
    }

    pub trait CbvQuery: Clone + Send + Sync {
        fn get_latest_cbv(&self) -> Option<Cbv>;
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct Pev {
        #[prost(uint64, tag = "1")]
        pub epoch: u64,
        #[prost(message, optional, tag = "2")]
        pub pev_digest: Option<Digest32>,
        #[prost(message, optional, tag = "3")]
        pub pev: Option<PolicyEcologyVector>,
    }

    pub trait PevQuery: Clone + Send + Sync {
        fn get_latest_pev(&self) -> Option<Pev>;
    }

    pub trait MicrocircuitConfigCommit: Clone + Send + Sync {
        fn commit_microcircuit_config(&self, append: MicrocircuitConfigAppend) -> PvgsReceipt;
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct AssetManifestAppend {
        #[prost(message, optional, tag = "1")]
        pub manifest: Option<AssetManifest>,
    }

    pub trait AssetManifestCommit: Clone + Send + Sync {
        fn commit_asset_manifest(&self, append: AssetManifestAppend) -> PvgsReceipt;
    }

    #[derive(Clone, PartialEq, Message)]
    pub struct AssetBundleAppend {
        #[prost(message, optional, tag = "1")]
        pub bundle: Option<AssetBundle>,
    }

    pub trait AssetBundleCommit: Clone + Send + Sync {
        fn commit_asset_bundle(&self, append: AssetBundleAppend) -> PvgsReceipt;
    }

    pub trait AssetBundleQuery: Clone + Send + Sync {
        fn get_latest_asset_bundle(&self) -> Option<AssetBundle>;
        fn get_asset_bundle(&self, digest: [u8; 32]) -> Option<AssetBundle>;
    }

    #[derive(Clone, Default)]
    pub struct InMemoryPvgs {
        latest_cbv: Arc<Mutex<Option<Cbv>>>,
        latest_pev: Arc<Mutex<Option<Pev>>>,
        latest_microcircuit_config: Arc<Mutex<Option<MicrocircuitConfigAppend>>>,
        latest_asset_manifest: Arc<Mutex<Option<AssetManifestAppend>>>,
        latest_asset_bundle: Arc<Mutex<Option<AssetBundleAppend>>>,
        asset_bundles: Arc<Mutex<HashMap<[u8; 32], AssetBundle>>>,
    }

    impl InMemoryPvgs {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn commit_cbv_update(&self, update: Cbv) {
            let mut guard = self.latest_cbv.lock().unwrap();
            if guard
                .as_ref()
                .map(|existing| existing.epoch <= update.epoch)
                .unwrap_or(true)
            {
                *guard = Some(update);
            }
        }

        pub fn commit_pev_update(&self, update: Pev) {
            let mut guard = self.latest_pev.lock().unwrap();
            if guard
                .as_ref()
                .map(|existing| existing.epoch <= update.epoch)
                .unwrap_or(true)
            {
                *guard = Some(update);
            }
        }

        pub fn latest_microcircuit_config(&self) -> Option<MicrocircuitConfigAppend> {
            self.latest_microcircuit_config.lock().unwrap().clone()
        }

        pub fn latest_asset_manifest(&self) -> Option<AssetManifestAppend> {
            self.latest_asset_manifest.lock().unwrap().clone()
        }

        pub fn latest_asset_bundle(&self) -> Option<AssetBundleAppend> {
            self.latest_asset_bundle.lock().unwrap().clone()
        }
    }

    impl CbvQuery for InMemoryPvgs {
        fn get_latest_cbv(&self) -> Option<Cbv> {
            self.latest_cbv.lock().unwrap().clone()
        }
    }

    impl PevQuery for InMemoryPvgs {
        fn get_latest_pev(&self) -> Option<Pev> {
            self.latest_pev.lock().unwrap().clone()
        }
    }

    impl MicrocircuitConfigCommit for InMemoryPvgs {
        fn commit_microcircuit_config(&self, append: MicrocircuitConfigAppend) -> PvgsReceipt {
            *self.latest_microcircuit_config.lock().unwrap() = Some(append);
            PvgsReceipt::default()
        }
    }

    impl AssetManifestCommit for InMemoryPvgs {
        fn commit_asset_manifest(&self, append: AssetManifestAppend) -> PvgsReceipt {
            *self.latest_asset_manifest.lock().unwrap() = Some(append);
            PvgsReceipt::default()
        }
    }

    impl AssetBundleCommit for InMemoryPvgs {
        fn commit_asset_bundle(&self, append: AssetBundleAppend) -> PvgsReceipt {
            if let Some(bundle) = append.bundle.as_ref() {
                if bundle.bundle_digest.len() == 32 {
                    let mut digest = [0u8; 32];
                    digest.copy_from_slice(&bundle.bundle_digest);
                    self.asset_bundles
                        .lock()
                        .unwrap()
                        .insert(digest, bundle.clone());
                }
            }
            *self.latest_asset_bundle.lock().unwrap() = Some(append);
            PvgsReceipt::default()
        }
    }

    impl AssetBundleQuery for InMemoryPvgs {
        fn get_latest_asset_bundle(&self) -> Option<AssetBundle> {
            self.latest_asset_bundle
                .lock()
                .unwrap()
                .clone()
                .and_then(|append| append.bundle)
        }

        fn get_asset_bundle(&self, digest: [u8; 32]) -> Option<AssetBundle> {
            self.asset_bundles.lock().unwrap().get(&digest).cloned()
        }
    }
}
