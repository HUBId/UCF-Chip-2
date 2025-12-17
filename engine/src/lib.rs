#![forbid(unsafe_code)]

use blake3::Hasher;
use profiles::{decide, ControlDecision, ProfileState};
use prost::Message;
use rsv::RsvState;
use ucf::v1::{ActiveProfile, ControlFrame, Overlays, ReasonCode, ToolClassMask};

const CONTROL_FRAME_DOMAIN: &str = "UCF:HASH:CONTROL_FRAME";

#[derive(Default)]
pub struct RegulationEngine {
    pub rsv: RsvState,
}

impl RegulationEngine {
    pub fn on_signal_frame(
        &mut self,
        mut frame: ucf::v1::SignalFrame,
        now_ms: u64,
    ) -> ControlFrame {
        if frame.timestamp_ms.is_none() {
            frame.timestamp_ms = Some(now_ms);
        }

        self.rsv.update_from_signal_frame(&frame);
        self.rsv.last_seen_frame_ts_ms = frame.timestamp_ms;

        let decision = decide(&self.rsv, now_ms);
        self.render_control_frame(decision)
    }

    pub fn on_tick(&mut self, now_ms: u64) -> ControlFrame {
        self.rsv.missing_frame_counter = self.rsv.missing_frame_counter.saturating_add(1);

        let decision = decide(&self.rsv, now_ms);
        self.render_control_frame(decision)
    }

    fn render_control_frame(&self, decision: ControlDecision) -> ControlFrame {
        let overlays = Overlays {
            simulate_first: decision.overlays.simulate_first,
            export_lock: decision.overlays.export_lock,
            novelty_lock: decision.overlays.novelty_lock,
        };

        let profile = ActiveProfile {
            profile: profile_string(decision.profile),
        };

        let toolclass_mask = match decision.profile {
            ProfileState::M0Research => ToolClassMask {
                read: true,
                write: false,
                execute: false,
                transform: true,
                export: true,
            },
            ProfileState::M1Restricted => ToolClassMask {
                read: true,
                write: false,
                execute: false,
                transform: true,
                export: false,
            },
            ProfileState::M2Quarantine | ProfileState::M3Forensic => ToolClassMask {
                read: true,
                write: false,
                execute: false,
                transform: false,
                export: false,
            },
        };

        let mut control_frame = ControlFrame {
            active_profile: Some(profile),
            overlays: Some(overlays),
            toolclass_mask: Some(toolclass_mask),
            profile_reason_codes: profile_reasons(&decision),
            control_frame_digest: None,
        };

        let mut buf = Vec::new();
        control_frame.encode(&mut buf).unwrap();

        let mut hasher = Hasher::new_derive_key(CONTROL_FRAME_DOMAIN);
        hasher.update(&buf);
        let digest = hasher.finalize();

        control_frame.control_frame_digest = Some(digest.as_bytes().to_vec());

        control_frame
    }
}

fn profile_string(profile: ProfileState) -> String {
    match profile {
        ProfileState::M0Research => "M0_RESEARCH".to_string(),
        ProfileState::M1Restricted => "M1_RESTRICTED".to_string(),
        ProfileState::M2Quarantine => "M2_QUARANTINE".to_string(),
        ProfileState::M3Forensic => "M3_FORENSIC".to_string(),
    }
}

fn profile_reasons(decision: &ControlDecision) -> Vec<i32> {
    match decision.profile {
        ProfileState::M3Forensic => {
            vec![ReasonCode::ReIntegrityFail as i32]
        }
        ProfileState::M2Quarantine => {
            vec![ReasonCode::ThExfilHighConfidence as i32]
        }
        ProfileState::M1Restricted => {
            if decision.missing_frame_override {
                vec![ReasonCode::ReIntegrityDegraded as i32]
            } else {
                vec![ReasonCode::ThPolicyProbing as i32]
            }
        }
        ProfileState::M0Research => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucf::v1::{ExecStats, IntegrityStateClass, PolicyStats, SignalFrame, WindowKind};

    fn base_frame() -> SignalFrame {
        SignalFrame {
            window_kind: WindowKind::Short as i32,
            window_index: Some(1),
            timestamp_ms: Some(1),
            policy_stats: Some(PolicyStats {
                deny_count: 0,
                allow_count: 10,
            }),
            exec_stats: Some(ExecStats { timeout_count: 0 }),
            integrity_state: IntegrityStateClass::Ok as i32,
            top_reason_codes: Vec::new(),
            signal_frame_digest: None,
        }
    }

    #[test]
    fn digest_is_deterministic() {
        let mut engine = RegulationEngine::default();
        let frame = base_frame();
        let control_a = engine.on_signal_frame(frame.clone(), 1);
        let control_b = engine.on_signal_frame(frame, 1);
        assert_eq!(
            control_a.control_frame_digest,
            control_b.control_frame_digest
        );
    }

    #[test]
    fn policy_pressure_triggers_m1() {
        let mut engine = RegulationEngine::default();
        let mut frame = base_frame();
        frame.policy_stats = Some(PolicyStats {
            deny_count: 5,
            allow_count: 0,
        });

        let control = engine.on_signal_frame(frame, 1);
        let profile = control.active_profile.unwrap().profile;
        assert_eq!(profile, "M1_RESTRICTED");
        let overlays = control.overlays.unwrap();
        assert!(overlays.simulate_first && overlays.export_lock && overlays.novelty_lock);
    }

    #[test]
    fn integrity_fail_triggers_m3() {
        let mut engine = RegulationEngine::default();
        let mut frame = base_frame();
        frame.integrity_state = IntegrityStateClass::Fail as i32;
        let control = engine.on_signal_frame(frame, 1);

        let mask = control.toolclass_mask.unwrap();
        assert!(!mask.export && !mask.write && !mask.execute);
        assert_eq!(control.active_profile.unwrap().profile, "M3_FORENSIC");
    }

    #[test]
    fn missing_frame_triggers_restriction() {
        let mut engine = RegulationEngine::default();
        let frame = base_frame();
        let _ = engine.on_signal_frame(frame, 0);
        let control = engine.on_tick(60_000);
        assert_eq!(control.active_profile.unwrap().profile, "M1_RESTRICTED");
        assert!(control.overlays.unwrap().export_lock);
    }

    #[test]
    fn toolclass_masks_match_profiles() {
        let mut engine = RegulationEngine::default();
        let frame = base_frame();
        let control = engine.on_signal_frame(frame.clone(), 1);
        let mask = control.toolclass_mask.unwrap();
        assert!(mask.read && mask.transform && mask.export);
        assert!(!mask.write && !mask.execute);

        let mut restricted_frame = frame.clone();
        restricted_frame.policy_stats = Some(PolicyStats {
            deny_count: 5,
            allow_count: 0,
        });
        let restricted_control = engine.on_signal_frame(restricted_frame, 1);
        let mask = restricted_control.toolclass_mask.unwrap();
        assert!(mask.read && mask.transform);
        assert!(!mask.export && !mask.write && !mask.execute);

        let mut quarantine_frame = frame;
        quarantine_frame.top_reason_codes = vec![ReasonCode::RcCdDlpSecretPattern as i32];
        let quarantine_control = engine.on_signal_frame(quarantine_frame, 1);
        let profile = quarantine_control.active_profile.unwrap();
        assert_eq!(profile.profile, "M2_QUARANTINE");
        let mask = quarantine_control.toolclass_mask.unwrap();
        assert!(mask.read);
        if mask.export || mask.write || mask.execute {
            panic!(
                "unexpected toolclass mask: read={}, write={}, execute={}, transform={}, export={}",
                mask.read, mask.write, mask.execute, mask.transform, mask.export
            );
        }
    }
}
