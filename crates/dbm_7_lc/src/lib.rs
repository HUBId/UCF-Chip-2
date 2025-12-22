#![forbid(unsafe_code)]

use dbm_core::{DbmModule, IntegrityState, LevelClass, ReasonSet};

#[derive(Debug, Clone, Default)]
pub struct LcInput {
    pub integrity: IntegrityState,
    pub receipt_invalid_count_short: u32,
    pub receipt_missing_count_short: u32,
    pub dlp_critical_present_short: bool,
    pub timeout_count_short: u32,
    pub deny_count_short: u32,
    pub arousal_floor: LevelClass,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LcOutput {
    pub arousal: LevelClass,
    pub hint_simulate_first: bool,
    pub hint_novelty_lock: bool,
    pub reason_codes: ReasonSet,
}

impl Default for LcOutput {
    fn default() -> Self {
        Self {
            arousal: LevelClass::Low,
            hint_simulate_first: false,
            hint_novelty_lock: false,
            reason_codes: ReasonSet::default(),
        }
    }
}

#[derive(Debug, Default)]
pub struct Lc {}

impl Lc {
    pub fn new() -> Self {
        Self {}
    }
}

impl DbmModule for Lc {
    type Input = LcInput;
    type Output = LcOutput;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        let mut reason_codes = ReasonSet::default();

        let mut arousal = if input.integrity != IntegrityState::Ok
            || input.receipt_invalid_count_short >= 1
            || input.dlp_critical_present_short
            || input.timeout_count_short >= 2
        {
            reason_codes.insert("lc_high_trigger");
            LevelClass::High
        } else if input.deny_count_short >= 2
            || input.receipt_missing_count_short >= 1
            || input.timeout_count_short == 1
        {
            reason_codes.insert("lc_med_trigger");
            LevelClass::Med
        } else {
            LevelClass::Low
        };

        fn severity(level: LevelClass) -> u8 {
            match level {
                LevelClass::Low => 0,
                LevelClass::Med => 1,
                LevelClass::High => 2,
            }
        }

        if severity(arousal) < severity(input.arousal_floor) {
            arousal = input.arousal_floor;
            reason_codes.insert("lc_floor");
        }

        let (hint_simulate_first, hint_novelty_lock) = match arousal {
            LevelClass::High => (true, true),
            LevelClass::Med => (true, false),
            LevelClass::Low => (false, false),
        };

        LcOutput {
            arousal,
            hint_simulate_first,
            hint_novelty_lock,
            reason_codes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_input() -> LcInput {
        LcInput {
            integrity: IntegrityState::Ok,
            arousal_floor: LevelClass::Low,
            ..Default::default()
        }
    }

    #[test]
    fn receipt_invalid_forces_high() {
        let mut lc = Lc::new();
        let output = lc.tick(&LcInput {
            receipt_invalid_count_short: 1,
            ..base_input()
        });

        assert_eq!(output.arousal, LevelClass::High);
        assert!(output.hint_simulate_first);
        assert!(output.hint_novelty_lock);
    }

    #[test]
    fn timeout_burst_forces_high() {
        let mut lc = Lc::new();
        let output = lc.tick(&LcInput {
            timeout_count_short: 2,
            ..base_input()
        });

        assert_eq!(output.arousal, LevelClass::High);
    }

    #[test]
    fn deny_threshold_sets_med() {
        let mut lc = Lc::new();
        let output = lc.tick(&LcInput {
            deny_count_short: 2,
            ..base_input()
        });

        assert_eq!(output.arousal, LevelClass::Med);
        assert!(output.hint_simulate_first);
        assert!(!output.hint_novelty_lock);
    }

    #[test]
    fn floor_applies() {
        let mut lc = Lc::new();
        let output = lc.tick(&LcInput {
            arousal_floor: LevelClass::Med,
            ..base_input()
        });

        assert_eq!(output.arousal, LevelClass::Med);
        assert!(output.reason_codes.codes.contains(&"lc_floor".to_string()));
    }
}
