#![forbid(unsafe_code)]

use dbm_bus::BrainBus;
use pvgs_client::PvgsWriter;
use ucf::v1::{MicrocircuitConfigEvidence, MicrocircuitModule};

const CONFIG_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct MicrocircuitDigests {
    pub(crate) lc: Option<[u8; 32]>,
    pub(crate) sn: Option<[u8; 32]>,
    pub(crate) hpa: Option<[u8; 32]>,
}

impl MicrocircuitDigests {
    pub(crate) fn from_brain_bus(bus: &BrainBus) -> Self {
        Self {
            lc: bus.lc_config_digest(),
            sn: bus.sn_config_digest(),
            hpa: bus.hpa_config_digest(),
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct MicrocircuitPublisherState {
    started_at_ms: Option<u64>,
    published_lc_digest: Option<[u8; 32]>,
    published_sn_digest: Option<[u8; 32]>,
    published_hpa_digest: Option<[u8; 32]>,
}

impl MicrocircuitPublisherState {
    pub(crate) fn maybe_publish(
        &mut self,
        now_ms: u64,
        digests: MicrocircuitDigests,
        writer: Option<&mut (dyn PvgsWriter + Send + '_)>,
    ) {
        let created_at_ms = *self.started_at_ms.get_or_insert(now_ms);

        if let Some(writer) = writer {
            publish_module(
                MicrocircuitModule::Lc,
                digests.lc,
                &mut self.published_lc_digest,
                created_at_ms,
                writer,
            );
            publish_module(
                MicrocircuitModule::Sn,
                digests.sn,
                &mut self.published_sn_digest,
                created_at_ms,
                writer,
            );
            publish_module(
                MicrocircuitModule::Hpa,
                digests.hpa,
                &mut self.published_hpa_digest,
                created_at_ms,
                writer,
            );
        } else {
            if digests.lc.is_none() {
                self.published_lc_digest = None;
            }
            if digests.sn.is_none() {
                self.published_sn_digest = None;
            }
            if digests.hpa.is_none() {
                self.published_hpa_digest = None;
            }
        }
    }
}

fn publish_module(
    module: MicrocircuitModule,
    digest: Option<[u8; 32]>,
    published: &mut Option<[u8; 32]>,
    created_at_ms: u64,
    writer: &mut (dyn PvgsWriter + Send + '_),
) {
    let Some(digest) = digest else {
        *published = None;
        return;
    };

    if published.map(|current| current == digest).unwrap_or(false) {
        return;
    }

    let evidence = build_evidence(module, digest, created_at_ms, *published);
    if writer.commit_microcircuit_config(evidence).is_ok() {
        *published = Some(digest);
    }
}

fn build_evidence(
    module: MicrocircuitModule,
    digest: [u8; 32],
    created_at_ms: u64,
    prev_digest: Option<[u8; 32]>,
) -> MicrocircuitConfigEvidence {
    MicrocircuitConfigEvidence {
        module: module as i32,
        config_version: CONFIG_VERSION,
        config_digest: digest.to_vec(),
        created_at_ms,
        prev_config_digest: prev_digest.map(|bytes| bytes.to_vec()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;
    use pvgs_client::{PvgsError, PvgsWriter};
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct RecordingWriter {
        commits: Arc<Mutex<Vec<MicrocircuitConfigEvidence>>>,
    }

    impl PvgsWriter for RecordingWriter {
        fn commit_control_frame_evidence(
            &mut self,
            _session_id: &str,
            _control_frame_digest: [u8; 32],
        ) -> Result<(), PvgsError> {
            Ok(())
        }

        fn commit_microcircuit_config(
            &mut self,
            evidence: MicrocircuitConfigEvidence,
        ) -> Result<ucf::v1::PvgsReceipt, PvgsError> {
            self.commits.lock().unwrap().push(evidence);
            Ok(ucf::v1::PvgsReceipt::default())
        }

        fn commit_asset_manifest(
            &mut self,
            _manifest: ucf::v1::AssetManifest,
        ) -> Result<ucf::v1::PvgsReceipt, PvgsError> {
            Ok(ucf::v1::PvgsReceipt::default())
        }
    }

    #[test]
    fn publishes_once_on_startup() {
        let mut publisher = MicrocircuitPublisherState::default();
        let writer = Arc::new(Mutex::new(Vec::new()));
        let mut writer_impl = RecordingWriter {
            commits: writer.clone(),
        };
        let digest = [1u8; 32];
        let digests = MicrocircuitDigests {
            lc: Some(digest),
            sn: None,
            hpa: None,
        };

        publisher.maybe_publish(10, digests, Some(&mut writer_impl));
        publisher.maybe_publish(20, digests, Some(&mut writer_impl));

        let commits = writer.lock().unwrap();
        assert_eq!(commits.len(), 1);
        assert_eq!(commits[0].config_digest, digest.to_vec());
    }

    #[test]
    fn republishes_on_config_change() {
        let mut publisher = MicrocircuitPublisherState::default();
        let writer = Arc::new(Mutex::new(Vec::new()));
        let mut writer_impl = RecordingWriter {
            commits: writer.clone(),
        };
        let digest_a = [1u8; 32];
        let digest_b = [2u8; 32];

        publisher.maybe_publish(
            10,
            MicrocircuitDigests {
                lc: Some(digest_a),
                sn: None,
                hpa: None,
            },
            Some(&mut writer_impl),
        );
        publisher.maybe_publish(
            20,
            MicrocircuitDigests {
                lc: Some(digest_b),
                sn: None,
                hpa: None,
            },
            Some(&mut writer_impl),
        );

        let commits = writer.lock().unwrap();
        assert_eq!(commits.len(), 2);
        assert_eq!(commits[1].config_digest, digest_b.to_vec());
        assert_eq!(commits[1].prev_config_digest, Some(digest_a.to_vec()));
    }

    #[test]
    fn evidence_is_deterministic() {
        let digest = [3u8; 32];
        let evidence_a = build_evidence(MicrocircuitModule::Lc, digest, 42, None);
        let evidence_b = build_evidence(MicrocircuitModule::Lc, digest, 42, None);

        assert_eq!(evidence_a.config_digest, evidence_b.config_digest);
        assert_eq!(evidence_a.encode_to_vec(), evidence_b.encode_to_vec());
    }

    #[test]
    fn publishes_hpa_once_per_digest() {
        let mut publisher = MicrocircuitPublisherState::default();
        let writer = Arc::new(Mutex::new(Vec::new()));
        let mut writer_impl = RecordingWriter {
            commits: writer.clone(),
        };
        let digest = [9u8; 32];

        publisher.maybe_publish(
            100,
            MicrocircuitDigests {
                lc: None,
                sn: None,
                hpa: Some(digest),
            },
            Some(&mut writer_impl),
        );
        publisher.maybe_publish(
            200,
            MicrocircuitDigests {
                lc: None,
                sn: None,
                hpa: Some(digest),
            },
            Some(&mut writer_impl),
        );

        let commits = writer.lock().unwrap();
        assert_eq!(commits.len(), 1);
        assert_eq!(commits[0].module, MicrocircuitModule::Hpa as i32);
        assert_eq!(commits[0].config_digest, digest.to_vec());
        assert_eq!(commits[0].config_version, CONFIG_VERSION);
        assert_eq!(commits[0].created_at_ms, 100);
    }

    #[test]
    fn republishes_hpa_on_digest_change() {
        let mut publisher = MicrocircuitPublisherState::default();
        let writer = Arc::new(Mutex::new(Vec::new()));
        let mut writer_impl = RecordingWriter {
            commits: writer.clone(),
        };
        let digest_a = [4u8; 32];
        let digest_b = [5u8; 32];

        publisher.maybe_publish(
            100,
            MicrocircuitDigests {
                lc: None,
                sn: None,
                hpa: Some(digest_a),
            },
            Some(&mut writer_impl),
        );
        publisher.maybe_publish(
            200,
            MicrocircuitDigests {
                lc: None,
                sn: None,
                hpa: Some(digest_b),
            },
            Some(&mut writer_impl),
        );

        let commits = writer.lock().unwrap();
        assert_eq!(commits.len(), 2);
        assert_eq!(commits[1].module, MicrocircuitModule::Hpa as i32);
        assert_eq!(commits[1].config_digest, digest_b.to_vec());
        assert_eq!(commits[1].prev_config_digest, Some(digest_a.to_vec()));
        assert_eq!(commits[1].created_at_ms, 100);
    }
}
