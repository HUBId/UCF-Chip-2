#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use ucf::v1::{IntegrityStateClass, LevelClass, ReasonCode, SignalFrame, WindowKind};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RsvState {
    pub integrity: IntegrityStateClass,
    pub policy_pressure: LevelClass,
    pub threat: LevelClass,
    pub arousal: LevelClass,
    pub stability: LevelClass,
    pub divergence: LevelClass,
    pub budget_stress: LevelClass,
    pub replay_mismatch: LevelClass,
    pub receipt_failures: LevelClass,
    pub receipt_missing_count_window: u32,
    pub receipt_invalid_count_window: u32,
    pub last_seen_frame_ts_ms: Option<u64>,
    pub missing_frame_counter: u32,
    pub missing_data: bool,
    pub forensic_latched: bool,
    pub unlock_seen: bool,
    pub unlock_stable_windows: u32,
    pub unlock_ready: bool,
}

impl Default for RsvState {
    fn default() -> Self {
        Self {
            integrity: IntegrityStateClass::Ok,
            policy_pressure: LevelClass::Low,
            threat: LevelClass::Low,
            arousal: LevelClass::Low,
            stability: LevelClass::Low,
            divergence: LevelClass::Low,
            budget_stress: LevelClass::Low,
            replay_mismatch: LevelClass::Low,
            receipt_failures: LevelClass::Low,
            receipt_missing_count_window: 0,
            receipt_invalid_count_window: 0,
            last_seen_frame_ts_ms: None,
            missing_frame_counter: 0,
            missing_data: false,
            forensic_latched: false,
            unlock_seen: false,
            unlock_stable_windows: 0,
            unlock_ready: false,
        }
    }
}

impl RsvState {
    pub fn reset_forensic(&mut self) {
        self.forensic_latched = false;
        self.unlock_seen = false;
        self.unlock_stable_windows = 0;
        self.unlock_ready = false;
    }

    pub fn update_unlock_state(&mut self, frame: &SignalFrame) {
        let window_kind = WindowKind::try_from(frame.window_kind).unwrap_or(WindowKind::Unknown);
        if window_kind != WindowKind::Medium {
            return;
        }

        let unlock_present = frame_contains_reason(frame, ReasonCode::RcGvRecoveryUnlockGranted);
        self.unlock_seen |= unlock_present;

        if !self.unlock_seen {
            self.unlock_ready = false;
            self.unlock_stable_windows = 0;
            return;
        }

        let integrity_state = IntegrityStateClass::try_from(frame.integrity_state)
            .unwrap_or(IntegrityStateClass::Degraded);

        if !unlock_present {
            self.unlock_stable_windows = 0;
            self.unlock_ready = false;
            return;
        }

        let critical_present = frame_has_critical_reason(frame, unlock_present);

        if integrity_state == IntegrityStateClass::Fail {
            if critical_present {
                self.unlock_stable_windows = 0;
                self.unlock_ready = false;
                return;
            }
            self.unlock_stable_windows = self.unlock_stable_windows.saturating_add(1);
        } else {
            self.unlock_stable_windows = self.unlock_stable_windows.saturating_add(1);
        }

        self.unlock_ready = self.unlock_stable_windows >= 2;
    }
}

fn frame_contains_reason(frame: &SignalFrame, code: ReasonCode) -> bool {
    let code = code as i32;
    frame.reason_codes.contains(&code)
        || frame.top_reason_codes.contains(&code)
        || frame
            .policy_stats
            .as_ref()
            .map(|stats| stats.top_reason_codes.contains(&code))
            .unwrap_or(false)
        || frame
            .exec_stats
            .as_ref()
            .map(|stats| stats.top_reason_codes.contains(&code))
            .unwrap_or(false)
}

fn frame_has_critical_reason(frame: &SignalFrame, unlock_present: bool) -> bool {
    let mut critical = false;
    for code in frame
        .top_reason_codes
        .iter()
        .chain(frame.reason_codes.iter())
        .chain(
            frame
                .policy_stats
                .as_ref()
                .map(|stats| stats.top_reason_codes.as_slice())
                .unwrap_or_default()
                .iter(),
        )
        .chain(
            frame
                .exec_stats
                .as_ref()
                .map(|stats| stats.top_reason_codes.as_slice())
                .unwrap_or_default()
                .iter(),
        )
    {
        if let Ok(reason) = ReasonCode::try_from(*code) {
            if reason == ReasonCode::ReIntegrityFail && !unlock_present {
                critical = true;
                break;
            }

            if format!("{:?}", reason).starts_with("RcRxCt") {
                critical = true;
                break;
            }
        }
    }

    critical
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_state_is_low_and_ok() {
        let state = RsvState::default();
        assert_eq!(state.integrity, IntegrityStateClass::Ok);
        assert_eq!(state.policy_pressure, LevelClass::Low);
        assert_eq!(state.threat, LevelClass::Low);
        assert_eq!(state.arousal, LevelClass::Low);
        assert_eq!(state.stability, LevelClass::Low);
        assert_eq!(state.replay_mismatch, LevelClass::Low);
        assert_eq!(state.receipt_failures, LevelClass::Low);
        assert_eq!(state.receipt_missing_count_window, 0);
        assert_eq!(state.receipt_invalid_count_window, 0);
        assert_eq!(state.missing_frame_counter, 0);
        assert!(!state.missing_data);
        assert!(!state.unlock_seen);
        assert_eq!(state.unlock_stable_windows, 0);
        assert!(!state.unlock_ready);
    }

    #[test]
    fn missing_frame_counter_increments() {
        let mut state = RsvState::default();
        state.missing_frame_counter = 2;
        state.missing_frame_counter = state.missing_frame_counter.saturating_add(1);
        assert_eq!(state.missing_frame_counter, 3);
    }
}
