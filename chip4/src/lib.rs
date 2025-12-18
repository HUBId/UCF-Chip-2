#![forbid(unsafe_code)]

pub mod pvgs {
    use prost::Message;
    use std::sync::{Arc, Mutex};
    use ucf::v1::CharacterBaselineVector;

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

    #[derive(Clone, Default)]
    pub struct InMemoryPvgs {
        latest_cbv: Arc<Mutex<Option<Cbv>>>,
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
    }

    impl CbvQuery for InMemoryPvgs {
        fn get_latest_cbv(&self) -> Option<Cbv> {
            self.latest_cbv.lock().unwrap().clone()
        }
    }
}
