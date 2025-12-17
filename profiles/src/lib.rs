#![forbid(unsafe_code)]

use rsv::RsvState;
use serde::{Deserialize, Serialize};
use ucf::v1::ReasonCode;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProfileState {
    M0Research,
    M1Restricted,
    M2Quarantine,
    M3Forensic,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct OverlaySet {
    pub simulate_first: bool,
    pub export_lock: bool,
    pub novelty_lock: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlDecision {
    pub profile: ProfileState,
    pub overlays: OverlaySet,
    pub deescalation_lock: bool,
    pub missing_frame_override: bool,
    pub profile_reason_codes: Vec<ReasonCode>,
}

pub fn decide(rsv: &RsvState, now_ms: u64) -> ControlDecision {
    let overlays_all_true = OverlaySet {
        simulate_first: true,
        export_lock: true,
        novelty_lock: true,
    };

    let missing = rsv
        .last_seen_frame_ts_ms
        .map(|ts| now_ms.saturating_sub(ts) > 30_000)
        .unwrap_or(true)
        || rsv.missing_frame_counter >= 1;

    if rsv.integrity == ucf::v1::IntegrityStateClass::Fail {
        return ControlDecision {
            profile: ProfileState::M3Forensic,
            overlays: overlays_all_true.clone(),
            deescalation_lock: true,
            missing_frame_override: missing,
            profile_reason_codes: vec![ReasonCode::ReIntegrityFail],
        };
    }

    if rsv.threat == ucf::v1::LevelClass::High {
        return ControlDecision {
            profile: ProfileState::M2Quarantine,
            overlays: overlays_all_true.clone(),
            deescalation_lock: true,
            missing_frame_override: missing,
            profile_reason_codes: vec![ReasonCode::ThExfilHighConfidence],
        };
    }

    if rsv.receipt_failures == ucf::v1::LevelClass::High {
        let profile = if rsv.receipt_invalid_count_window >= 2 {
            ProfileState::M2Quarantine
        } else {
            ProfileState::M1Restricted
        };

        return ControlDecision {
            profile,
            overlays: overlays_all_true.clone(),
            deescalation_lock: true,
            missing_frame_override: missing,
            profile_reason_codes: vec![ReasonCode::RcGeExecDispatchBlocked],
        };
    }

    if rsv.receipt_failures == ucf::v1::LevelClass::Med {
        return ControlDecision {
            profile: ProfileState::M1Restricted,
            overlays: overlays_all_true.clone(),
            deescalation_lock: true,
            missing_frame_override: missing,
            profile_reason_codes: vec![ReasonCode::RcGeExecDispatchBlocked],
        };
    }

    if rsv.policy_pressure == ucf::v1::LevelClass::High {
        return ControlDecision {
            profile: ProfileState::M1Restricted,
            overlays: overlays_all_true.clone(),
            deescalation_lock: true,
            missing_frame_override: missing,
            profile_reason_codes: vec![ReasonCode::ThPolicyProbing],
        };
    }

    if missing {
        return ControlDecision {
            profile: ProfileState::M1Restricted,
            overlays: overlays_all_true,
            deescalation_lock: true,
            missing_frame_override: true,
            profile_reason_codes: vec![ReasonCode::ReIntegrityDegraded],
        };
    }

    ControlDecision {
        profile: ProfileState::M0Research,
        overlays: OverlaySet::default(),
        deescalation_lock: false,
        missing_frame_override: missing,
        profile_reason_codes: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rsv::RsvState;
    use ucf::v1::{IntegrityStateClass, LevelClass, ReasonCode};

    #[test]
    fn integrity_fail_forces_forensic() {
        let rsv = RsvState {
            integrity: IntegrityStateClass::Fail,
            ..Default::default()
        };
        let decision = decide(&rsv, 0);

        assert_eq!(decision.profile, ProfileState::M3Forensic);
        assert!(decision.deescalation_lock);
        assert!(decision.overlays.simulate_first);
        assert_eq!(
            decision.profile_reason_codes,
            vec![ReasonCode::ReIntegrityFail]
        );
    }

    #[test]
    fn missing_frames_enforce_restriction() {
        let rsv = RsvState {
            last_seen_frame_ts_ms: Some(0),
            ..Default::default()
        };
        let decision = decide(&rsv, 31_000);
        assert_eq!(decision.profile, ProfileState::M1Restricted);
        assert!(decision.overlays.export_lock);
        assert!(decision.deescalation_lock);
        assert_eq!(
            decision.profile_reason_codes,
            vec![ReasonCode::ReIntegrityDegraded]
        );
    }

    #[test]
    fn policy_pressure_and_threat_paths() {
        let mut rsv = RsvState {
            policy_pressure: LevelClass::High,
            ..Default::default()
        };
        let decision = decide(&rsv, 0);
        assert_eq!(decision.profile, ProfileState::M1Restricted);
        assert_eq!(
            decision.profile_reason_codes,
            vec![ReasonCode::ThPolicyProbing]
        );

        rsv.policy_pressure = LevelClass::Low;
        rsv.threat = LevelClass::High;
        let decision = decide(&rsv, 0);
        assert_eq!(decision.profile, ProfileState::M2Quarantine);
        assert_eq!(
            decision.profile_reason_codes,
            vec![ReasonCode::ThExfilHighConfidence]
        );
    }

    #[test]
    fn receipt_failures_trigger_restriction() {
        let rsv = RsvState {
            receipt_failures: LevelClass::High,
            receipt_missing_count_window: 1,
            ..Default::default()
        };

        let decision = decide(&rsv, 0);
        assert_eq!(decision.profile, ProfileState::M1Restricted);
        assert!(decision.overlays.simulate_first);
        assert_eq!(
            decision.profile_reason_codes,
            vec![ReasonCode::RcGeExecDispatchBlocked]
        );
    }

    #[test]
    fn receipt_invalid_escalates_to_quarantine() {
        let rsv = RsvState {
            receipt_failures: LevelClass::High,
            receipt_invalid_count_window: 2,
            ..Default::default()
        };

        let decision = decide(&rsv, 0);
        assert_eq!(decision.profile, ProfileState::M2Quarantine);
        assert!(decision.deescalation_lock);
        assert!(decision.overlays.export_lock);
        assert_eq!(
            decision.profile_reason_codes,
            vec![ReasonCode::RcGeExecDispatchBlocked]
        );
    }
}
