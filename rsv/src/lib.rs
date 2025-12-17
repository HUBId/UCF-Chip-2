#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use ucf::v1::{IntegrityStateClass, LevelClass, ReasonCode, SignalFrame};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RsvState {
    pub integrity: IntegrityStateClass,
    pub policy_pressure: LevelClass,
    pub threat: LevelClass,
    pub arousal: LevelClass,
    pub stability: LevelClass,
    pub last_seen_frame_ts_ms: Option<u64>,
    pub missing_frame_counter: u32,
}

impl Default for RsvState {
    fn default() -> Self {
        Self {
            integrity: IntegrityStateClass::Ok,
            policy_pressure: LevelClass::Low,
            threat: LevelClass::Low,
            arousal: LevelClass::Low,
            stability: LevelClass::Low,
            last_seen_frame_ts_ms: None,
            missing_frame_counter: 0,
        }
    }
}

impl RsvState {
    pub fn update_from_signal_frame(&mut self, frame: &SignalFrame) {
        self.last_seen_frame_ts_ms = frame.timestamp_ms;
        self.missing_frame_counter = 0;

        let integrity_state = IntegrityStateClass::try_from(frame.integrity_state)
            .unwrap_or(IntegrityStateClass::Unknown);

        self.integrity = match integrity_state {
            IntegrityStateClass::Fail => IntegrityStateClass::Fail,
            IntegrityStateClass::Degraded => IntegrityStateClass::Degraded,
            IntegrityStateClass::Ok => IntegrityStateClass::Ok,
            _ => IntegrityStateClass::Unknown,
        };

        let deny_count = frame
            .policy_stats
            .as_ref()
            .map(|stats| stats.deny_count)
            .unwrap_or_default();

        self.policy_pressure = if deny_count >= 5 {
            LevelClass::High
        } else if deny_count >= 2 {
            LevelClass::Med
        } else {
            LevelClass::Low
        };

        let top_codes: Vec<ReasonCode> = frame
            .top_reason_codes
            .iter()
            .filter_map(|code| ReasonCode::try_from(*code).ok())
            .collect();

        self.threat = if top_codes.iter().any(|code| {
            matches!(
                code,
                ReasonCode::RcCdDlpSecretPattern
                    | ReasonCode::RcCdDlpObfuscation
                    | ReasonCode::RcCdDlpStegano
            )
        }) {
            LevelClass::High
        } else {
            LevelClass::Low
        };

        let timeout_count = frame
            .exec_stats
            .as_ref()
            .map(|stats| stats.timeout_count)
            .unwrap_or_default();

        self.arousal = if timeout_count >= 2 || self.policy_pressure == LevelClass::High {
            LevelClass::High
        } else if deny_count >= 2 {
            LevelClass::Med
        } else {
            LevelClass::Low
        };

        self.stability = if self.integrity != IntegrityStateClass::Ok {
            LevelClass::High
        } else if self.policy_pressure == LevelClass::Med {
            LevelClass::Med
        } else {
            LevelClass::Low
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucf::v1::{ExecStats, PolicyStats, WindowKind};

    #[test]
    fn default_state_is_low_and_ok() {
        let state = RsvState::default();
        assert_eq!(state.integrity, IntegrityStateClass::Ok);
        assert_eq!(state.policy_pressure, LevelClass::Low);
        assert_eq!(state.threat, LevelClass::Low);
        assert_eq!(state.arousal, LevelClass::Low);
        assert_eq!(state.stability, LevelClass::Low);
        assert_eq!(state.missing_frame_counter, 0);
    }

    #[test]
    fn policy_stats_drive_pressure_and_arousal() {
        let mut state = RsvState::default();
        let frame = SignalFrame {
            window_kind: WindowKind::Short as i32,
            window_index: Some(1),
            timestamp_ms: Some(1),
            policy_stats: Some(PolicyStats {
                deny_count: 3,
                allow_count: 0,
            }),
            exec_stats: Some(ExecStats { timeout_count: 0 }),
            ..SignalFrame::default()
        };

        state.update_from_signal_frame(&frame);
        assert_eq!(state.policy_pressure, LevelClass::Med);
        assert_eq!(state.arousal, LevelClass::Med);
    }

    #[test]
    fn threat_escalates_on_secret_patterns() {
        let mut state = RsvState::default();
        let frame = SignalFrame {
            window_kind: WindowKind::Short as i32,
            window_index: Some(1),
            timestamp_ms: Some(1),
            top_reason_codes: vec![ReasonCode::RcCdDlpStegano as i32],
            ..SignalFrame::default()
        };

        state.update_from_signal_frame(&frame);
        assert_eq!(state.threat, LevelClass::High);
    }

    #[test]
    fn integrity_fail_drives_stability() {
        let mut state = RsvState::default();
        let frame = SignalFrame {
            window_kind: WindowKind::Short as i32,
            window_index: Some(1),
            timestamp_ms: Some(1),
            integrity_state: IntegrityStateClass::Fail as i32,
            ..SignalFrame::default()
        };

        state.update_from_signal_frame(&frame);
        assert_eq!(state.integrity, IntegrityStateClass::Fail);
        assert_eq!(state.stability, LevelClass::High);
    }
}
