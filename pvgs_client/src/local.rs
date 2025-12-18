#![cfg(any(test, feature = "local-pvgs"))]

use crate::PvgsReader;
use chip4::pvgs::{CbvQuery, Digest32};
use ucf::v1::CharacterBaselineVector;

#[derive(Clone)]
pub struct LocalPvgsReader<Q: CbvQuery> {
    query: Q,
}

impl<Q: CbvQuery> LocalPvgsReader<Q> {
    pub fn new(query: Q) -> Self {
        Self { query }
    }

    fn digest_from_proto(proto: Digest32) -> Option<[u8; 32]> {
        let bytes = proto.value;
        if bytes.len() != 32 {
            return None;
        }

        let mut digest = [0u8; 32];
        digest.copy_from_slice(&bytes);
        Some(digest)
    }
}

impl<Q: CbvQuery> PvgsReader for LocalPvgsReader<Q> {
    fn get_latest_cbv_digest(&self) -> Option<[u8; 32]> {
        self.query
            .get_latest_cbv()
            .and_then(|cbv| cbv.cbv_digest)
            .and_then(Self::digest_from_proto)
    }

    fn get_latest_cbv(&self) -> Option<CharacterBaselineVector> {
        self.query.get_latest_cbv().and_then(|cbv| cbv.cbv)
    }

    fn get_latest_pev_digest(&self) -> Option<[u8; 32]> {
        None
    }
}
