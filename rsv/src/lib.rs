#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use ucf::v1::{IntegrityStateClass, LevelClass};

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
        }
    }
}

impl RsvState {
    pub fn reset_forensic(&mut self) {
        self.forensic_latched = false;
    }
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
    }

    #[test]
    fn missing_frame_counter_increments() {
        let mut state = RsvState::default();
        state.missing_frame_counter = 2;
        state.missing_frame_counter = state.missing_frame_counter.saturating_add(1);
        assert_eq!(state.missing_frame_counter, 3);
    }
}
