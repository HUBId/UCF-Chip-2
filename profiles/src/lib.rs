#![forbid(unsafe_code)]

use rsv::RsvState;
use serde::{Deserialize, Serialize};

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
        };
    }

    if rsv.threat == ucf::v1::LevelClass::High {
        return ControlDecision {
            profile: ProfileState::M2Quarantine,
            overlays: overlays_all_true.clone(),
            deescalation_lock: true,
            missing_frame_override: missing,
        };
    }

    if rsv.policy_pressure == ucf::v1::LevelClass::High {
        return ControlDecision {
            profile: ProfileState::M1Restricted,
            overlays: overlays_all_true.clone(),
            deescalation_lock: true,
            missing_frame_override: missing,
        };
    }

    if missing {
        return ControlDecision {
            profile: ProfileState::M1Restricted,
            overlays: overlays_all_true,
            deescalation_lock: true,
            missing_frame_override: true,
        };
    }

    ControlDecision {
        profile: ProfileState::M0Research,
        overlays: OverlaySet::default(),
        deescalation_lock: false,
        missing_frame_override: missing,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rsv::RsvState;
    use ucf::v1::{IntegrityStateClass, LevelClass};

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
    }

    #[test]
    fn policy_pressure_and_threat_paths() {
        let mut rsv = RsvState {
            policy_pressure: LevelClass::High,
            ..Default::default()
        };
        let decision = decide(&rsv, 0);
        assert_eq!(decision.profile, ProfileState::M1Restricted);

        rsv.policy_pressure = LevelClass::Low;
        rsv.threat = LevelClass::High;
        let decision = decide(&rsv, 0);
        assert_eq!(decision.profile, ProfileState::M2Quarantine);
    }
}
