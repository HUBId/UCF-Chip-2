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

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DopaState {
    pub utility_score: i32,
    pub non_positive_streak: u32,
    pub reward_block: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DopaOutput {
    pub progress: LevelClass,
    pub incentive_focus: LevelClass,
    pub replay_hint: bool,
    pub reward_block: bool,
    pub reason_codes: ReasonSet,
}

impl Default for DopaOutput {
    fn default() -> Self {
        Self {
            progress: LevelClass::Low,
            incentive_focus: LevelClass::Low,
            replay_hint: false,
            reward_block: false,
            reason_codes: ReasonSet::default(),
        }
    }
}

#[derive(Debug, Default)]
pub struct DopaminNacc {
    state: DopaState,
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
        let budget_penalty = match input.budget_stress {
            LevelClass::Low => 0,
            LevelClass::Med => 1,
            LevelClass::High => 2,
        };

        input.exec_success_count_medium as i32
            - input.exec_failure_count_medium as i32
            - (input.deny_count_medium as i32) / 5
            - budget_penalty
    }

    fn reward_block(input: &DopaInput) -> bool {
        input.integrity != IntegrityState::Ok
            || input.threat == LevelClass::High
            || input.policy_pressure == LevelClass::High
            || input.receipt_invalid_present
            || input.dlp_critical_present
            || input.replay_mismatch_present
    }

    fn map_progress(utility_score: i32) -> LevelClass {
        if utility_score >= 5 {
            LevelClass::High
        } else if utility_score >= 1 {
            LevelClass::Med
        } else {
            LevelClass::Low
        }
    }

    fn incentive_focus(progress: LevelClass, reward_block: bool) -> LevelClass {
        if reward_block {
            LevelClass::Low
        } else {
            progress
        }
    }
}

impl DbmModule for DopaminNacc {
    type Input = DopaInput;
    type Output = DopaOutput;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        let mut reason_codes = ReasonSet::default();

        self.state.reward_block = Self::reward_block(input);
        if self.state.reward_block {
            reason_codes.insert("RC.GV.PROGRESS.REWARD_BLOCKED");
        }

        let delta = Self::compute_delta(input);
        if delta > 0 {
            self.state.non_positive_streak = 0;
        } else {
            self.state.non_positive_streak = self.state.non_positive_streak.saturating_add(1);
        }

        self.state.utility_score = Self::clamp_utility(self.state.utility_score + delta);

        let mut progress = Self::map_progress(self.state.utility_score);
        if self.state.reward_block && progress == LevelClass::High {
            progress = LevelClass::Med;
        }

        let replay_hint = self.state.non_positive_streak >= 3;
        if replay_hint {
            reason_codes.insert("RC.GV.REPLAY.DIMINISHING_RETURNS");
        }

        let incentive_focus = Self::incentive_focus(progress, self.state.reward_block);

        DopaOutput {
            progress,
            incentive_focus,
            replay_hint,
            reward_block: self.state.reward_block,
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
        assert_eq!(output.incentive_focus, LevelClass::Low);
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

        assert_eq!(module.state.utility_score, 5);
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
    fn reason_codes_ordering_is_deterministic() {
        let mut module = DopaminNacc::new();
        let output = module.tick(&DopaInput {
            threat: LevelClass::High,
            replay_mismatch_present: true,
            ..base_input()
        });

        let mut sorted = output.reason_codes.codes.clone();
        sorted.sort();
        assert_eq!(sorted, output.reason_codes.codes);
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
