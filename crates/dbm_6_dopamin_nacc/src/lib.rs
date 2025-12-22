#![forbid(unsafe_code)]

use dbm_core::{DbmModule, IntegrityState, LevelClass, ReasonSet};

#[derive(Debug, Clone, Default)]
pub struct DopaInput {
    pub integrity: IntegrityState,
    pub threat: LevelClass,
    pub policy_pressure: LevelClass,
    pub receipt_invalid_present: bool,
    pub dlp_critical_present: bool,
    pub replay_mismatch_present: bool,
    pub exec_success_count_medium: u32,
    pub exec_failure_count_medium: u32,
    pub deny_count_medium: u32,
    pub budget_stress: LevelClass,
    pub macro_finalized_count_long: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DopaOutput {
    pub progress: LevelClass,
    pub incentive_focus_hint: LevelClass,
    pub replay_hint: bool,
    pub reason_codes: ReasonSet,
}

impl Default for DopaOutput {
    fn default() -> Self {
        Self {
            progress: LevelClass::Low,
            incentive_focus_hint: LevelClass::Low,
            replay_hint: false,
            reason_codes: ReasonSet::default(),
        }
    }
}

#[derive(Debug, Default)]
pub struct DopaminNacc {
    progress_current: LevelClass,
    utility_score: i32,
    trend_counter: u32,
    diminishing_returns_counter: u32,
    reward_block: bool,
}

impl DopaminNacc {
    const UTILITY_MIN: i32 = -100;
    const UTILITY_MAX: i32 = 100;

    pub fn new() -> Self {
        Self::default()
    }

    fn clamp_utility(score: i32) -> i32 {
        score.clamp(Self::UTILITY_MIN, Self::UTILITY_MAX)
    }

    fn compute_delta(input: &DopaInput) -> i32 {
        let mut delta =
            input.exec_success_count_medium as i32 - input.exec_failure_count_medium as i32;
        delta -= (input.deny_count_medium as i32) / 5;

        if input.budget_stress == LevelClass::High {
            delta -= 2;
        }

        delta
    }

    fn reward_block(input: &DopaInput) -> bool {
        input.integrity != IntegrityState::Ok
            || input.threat == LevelClass::High
            || input.policy_pressure == LevelClass::High
            || input.receipt_invalid_present
            || input.dlp_critical_present
            || input.replay_mismatch_present
    }

    fn map_progress(utility_score: i32, reward_block: bool) -> LevelClass {
        if reward_block {
            if utility_score >= 1 {
                LevelClass::Med
            } else {
                LevelClass::Low
            }
        } else if utility_score >= 5 {
            LevelClass::High
        } else if utility_score >= 1 {
            LevelClass::Med
        } else {
            LevelClass::Low
        }
    }

    fn incentive_hint(progress: LevelClass, reward_block: bool) -> LevelClass {
        match (progress, reward_block) {
            (LevelClass::High, false) => LevelClass::High,
            (LevelClass::Med, false) => LevelClass::Med,
            _ => LevelClass::Low,
        }
    }
}

impl DbmModule for DopaminNacc {
    type Input = DopaInput;
    type Output = DopaOutput;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        let mut reason_codes = ReasonSet::default();

        self.reward_block = Self::reward_block(input);
        if self.reward_block {
            reason_codes.insert("RC.GV.PROGRESS.REWARD_BLOCKED");
        }

        let delta = Self::compute_delta(input);
        if delta > 0 {
            self.trend_counter = self.trend_counter.saturating_add(1);
            self.diminishing_returns_counter = 0;
        } else {
            self.trend_counter = 0;
            self.diminishing_returns_counter = self.diminishing_returns_counter.saturating_add(1);
        }

        self.utility_score = Self::clamp_utility(self.utility_score + delta);
        let mut progress = Self::map_progress(self.utility_score, self.reward_block);
        if self.reward_block && progress == LevelClass::High {
            progress = LevelClass::Med;
        }
        self.progress_current = progress;

        let replay_hint = self.diminishing_returns_counter >= 3;
        if replay_hint {
            reason_codes.insert("RC.GV.REPLAY.DIMINISHING_RETURNS");
        }

        let incentive_focus_hint = Self::incentive_hint(progress, self.reward_block);

        DopaOutput {
            progress,
            incentive_focus_hint,
            replay_hint,
            reason_codes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_input() -> DopaInput {
        DopaInput {
            ..Default::default()
        }
    }

    #[test]
    fn reward_block_caps_progress_and_incentive() {
        let mut module = DopaminNacc::new();
        let output = module.tick(&DopaInput {
            threat: LevelClass::High,
            exec_success_count_medium: 10,
            ..base_input()
        });

        assert_eq!(output.progress, LevelClass::Med);
        assert_eq!(output.incentive_focus_hint, LevelClass::Low);
        assert!(output
            .reason_codes
            .codes
            .contains(&"RC.GV.PROGRESS.REWARD_BLOCKED".to_string()));
    }

    #[test]
    fn positive_deltas_raise_utility_score() {
        let mut module = DopaminNacc::new();
        let _ = module.tick(&DopaInput {
            exec_success_count_medium: 3,
            exec_failure_count_medium: 1,
            ..base_input()
        });
        let output = module.tick(&DopaInput {
            exec_success_count_medium: 4,
            exec_failure_count_medium: 1,
            ..base_input()
        });

        assert_eq!(module.utility_score, 5);
        assert_eq!(output.progress, LevelClass::High);
    }

    #[test]
    fn replay_hint_after_three_nonpositive_deltas() {
        let mut module = DopaminNacc::new();
        for _ in 0..2 {
            let _ = module.tick(&DopaInput {
                exec_failure_count_medium: 1,
                ..base_input()
            });
        }

        let output = module.tick(&DopaInput {
            exec_failure_count_medium: 1,
            ..base_input()
        });

        assert!(output.replay_hint);
        assert!(output
            .reason_codes
            .codes
            .contains(&"RC.GV.REPLAY.DIMINISHING_RETURNS".to_string()));
    }

    #[test]
    fn deterministic_outputs() {
        let mut module_a = DopaminNacc::new();
        let mut module_b = DopaminNacc::new();
        let input = DopaInput {
            exec_success_count_medium: 2,
            exec_failure_count_medium: 1,
            deny_count_medium: 5,
            ..base_input()
        };

        let out_a = module_a.tick(&input);
        let out_b = module_b.tick(&input);

        assert_eq!(out_a, out_b);
    }
}
