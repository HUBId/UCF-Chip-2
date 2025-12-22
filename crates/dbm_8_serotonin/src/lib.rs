#![forbid(unsafe_code)]

use dbm_core::{CooldownClass, DbmModule, IntegrityState, LevelClass, ReasonSet};

#[derive(Debug, Clone, Default)]
pub struct SerInput {
    pub integrity: IntegrityState,
    pub replay_mismatch_present: bool,
    pub receipt_invalid_count_medium: u32,
    pub dlp_critical_count_medium: u32,
    pub flapping_count_medium: u32,
    pub unlock_present: bool,
    pub stability_floor: LevelClass,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SerOutput {
    pub stability: LevelClass,
    pub cooldown_class: CooldownClass,
    pub deescalation_lock: bool,
    pub reason_codes: ReasonSet,
}

impl Default for SerOutput {
    fn default() -> Self {
        Self {
            stability: LevelClass::Low,
            cooldown_class: CooldownClass::Base,
            deescalation_lock: false,
            reason_codes: ReasonSet::default(),
        }
    }
}

#[derive(Debug, Default)]
pub struct Serotonin {
    stability_current: LevelClass,
    stable_windows: u32,
    last_was_high: bool,
}

impl Serotonin {
    pub fn new() -> Self {
        Self::default()
    }
}

impl DbmModule for Serotonin {
    type Input = SerInput;
    type Output = SerOutput;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        let mut reason_codes = ReasonSet::default();
        let mut desired = if input.integrity != IntegrityState::Ok
            || input.replay_mismatch_present
            || input.receipt_invalid_count_medium >= 1
            || input.dlp_critical_count_medium >= 5
            || input.flapping_count_medium >= 6
        {
            reason_codes.insert("ser_high_trigger");
            LevelClass::High
        } else if input.flapping_count_medium >= 2 || input.dlp_critical_count_medium >= 1 {
            reason_codes.insert("ser_med_trigger");
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

        if severity(desired) < severity(input.stability_floor) {
            desired = input.stability_floor;
            reason_codes.insert("ser_floor");
        }

        let mut stability = self.stability_current;

        if stability == LevelClass::High && desired != LevelClass::High {
            if self.stable_windows >= 3 {
                stability = LevelClass::Med;
            }
        } else if stability == LevelClass::Med && desired == LevelClass::Low {
            if self.stable_windows >= 5 {
                stability = LevelClass::Low;
            }
        } else {
            stability = desired;
        }

        if desired == LevelClass::Low
            && input.integrity == IntegrityState::Ok
            && input.receipt_invalid_count_medium == 0
            && !input.replay_mismatch_present
        {
            self.stable_windows = self.stable_windows.saturating_add(1);
        } else {
            self.stable_windows = 0;
        }

        self.stability_current = stability.max(desired);
        self.last_was_high = desired == LevelClass::High;

        let cooldown_class =
            if self.stability_current == LevelClass::High || input.flapping_count_medium >= 2 {
                CooldownClass::Longer
            } else {
                CooldownClass::Base
            };

        let mut deescalation_lock = self.stability_current == LevelClass::High;
        if !input.unlock_present && input.integrity == IntegrityState::Fail {
            deescalation_lock = true;
            reason_codes.insert("ser_forensic_latch");
        }

        SerOutput {
            stability: self.stability_current,
            cooldown_class,
            deescalation_lock,
            reason_codes,
        }
    }
}

trait LevelClassExt {
    fn max(self, other: Self) -> Self;
}

impl LevelClassExt for LevelClass {
    fn max(self, other: Self) -> Self {
        use LevelClass::*;
        match (self, other) {
            (High, _) | (_, High) => High,
            (Med, _) | (_, Med) => Med,
            _ => Low,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_input() -> SerInput {
        SerInput {
            integrity: IntegrityState::Ok,
            stability_floor: LevelClass::Low,
            ..Default::default()
        }
    }

    #[test]
    fn replay_mismatch_forces_high() {
        let mut ser = Serotonin::new();
        let output = ser.tick(&SerInput {
            replay_mismatch_present: true,
            ..base_input()
        });

        assert_eq!(output.stability, LevelClass::High);
    }

    #[test]
    fn hysteresis_requires_multiple_windows() {
        let mut ser = Serotonin::new();
        let _ = ser.tick(&SerInput {
            replay_mismatch_present: true,
            ..base_input()
        });

        let mut output = SerOutput::default();
        for _ in 0..2 {
            output = ser.tick(&base_input());
            assert_eq!(output.stability, LevelClass::High);
        }

        for _ in 0..3 {
            output = ser.tick(&base_input());
        }

        assert_eq!(output.stability, LevelClass::Med);
    }

    #[test]
    fn cooldown_longer_on_flapping() {
        let mut ser = Serotonin::new();
        let output = ser.tick(&SerInput {
            flapping_count_medium: 3,
            ..base_input()
        });

        assert_eq!(output.stability, LevelClass::Med);
        assert_eq!(output.cooldown_class, CooldownClass::Longer);
    }
}
