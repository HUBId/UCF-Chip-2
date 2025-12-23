#![forbid(unsafe_code)]

use dbm_core::{DbmModule, IntegrityState, LevelClass, ReasonSet};
use microcircuit_core::{digest_config, digest_meta, CircuitConfig, MicrocircuitBackend};
use microcircuit_dopamin_stub::{DopaInput, DopaOutput};

#[derive(Debug, Clone, Default)]
struct DopaAttractorState {
    utility: i32,
    motivation: i32,
    reward_block_latch: u8,
    no_progress_streak: u8,
    step_count: u64,
}

#[derive(Debug, Clone)]
pub struct DopaAttractorMicrocircuit {
    config: CircuitConfig,
    state: DopaAttractorState,
}

impl DopaAttractorMicrocircuit {
    pub fn new(config: CircuitConfig) -> Self {
        Self {
            config,
            state: DopaAttractorState::default(),
        }
    }

    fn compute_delta(input: &DopaInput) -> i32 {
        input.exec_success_count_medium as i32
            - input.exec_failure_count_medium as i32
            - (input.deny_count_medium as i32) / 5
    }

    fn compute_risk(input: &DopaInput) -> u8 {
        let mut risk: i32 = 0;

        if input.integrity != IntegrityState::Ok {
            risk += 40;
        }
        if input.threat == LevelClass::High {
            risk += 30;
        }
        if input.policy_pressure == LevelClass::High {
            risk += 20;
        }
        if input.receipt_invalid_present {
            risk += 25;
        }
        if input.dlp_critical_present {
            risk += 25;
        }
        if input.replay_mismatch_present {
            risk += 30;
        }

        risk.clamp(0, 100) as u8
    }
}

impl MicrocircuitBackend<DopaInput, DopaOutput> for DopaAttractorMicrocircuit {
    fn step(&mut self, input: &DopaInput, _now_ms: u64) -> DopaOutput {
        self.state.step_count = self.state.step_count.saturating_add(1);

        let delta = Self::compute_delta(input);
        let risk = Self::compute_risk(input);
        let reward_block = risk >= 25;

        if reward_block {
            self.state.reward_block_latch = (self.state.reward_block_latch + 2).min(10);
        } else if self.state.reward_block_latch > 0 {
            self.state.reward_block_latch -= 1;
        }

        let suppressed = reward_block || self.state.reward_block_latch > 0;

        self.state.utility = (self.state.utility + delta).clamp(-100, 100);

        let mut motivation_delta = if delta > 0 { 3 } else { -2 };
        if suppressed {
            motivation_delta -= 5;
        }
        self.state.motivation = (self.state.motivation + motivation_delta).clamp(0, 100);

        if delta <= 0 {
            self.state.no_progress_streak = (self.state.no_progress_streak + 1).min(10);
        } else {
            self.state.no_progress_streak = 0;
        }

        let progress = if suppressed {
            if self.state.utility >= 5 {
                LevelClass::Med
            } else {
                LevelClass::Low
            }
        } else if self.state.utility >= 10 {
            LevelClass::High
        } else if self.state.utility >= 2 {
            LevelClass::Med
        } else {
            LevelClass::Low
        };

        let incentive_focus = if suppressed {
            LevelClass::Low
        } else if self.state.motivation >= 70 && progress == LevelClass::High {
            LevelClass::High
        } else if self.state.motivation >= 40 {
            LevelClass::Med
        } else {
            LevelClass::Low
        };

        let replay_hint = self.state.no_progress_streak >= 3 || self.state.utility < -10;

        let mut reason_codes = ReasonSet::default();
        if reward_block {
            reason_codes.insert("RC.GV.PROGRESS.REWARD_BLOCKED");
        }
        if replay_hint {
            reason_codes.insert("RC.GV.REPLAY.DIMINISHING_RETURNS");
        }

        DopaOutput {
            progress,
            incentive_focus,
            replay_hint,
            reward_block: suppressed,
            reason_codes,
        }
    }

    fn snapshot_digest(&self) -> [u8; 32] {
        let mut bytes = Vec::new();
        bytes.extend(self.state.utility.to_le_bytes());
        bytes.extend(self.state.motivation.to_le_bytes());
        bytes.push(self.state.reward_block_latch);
        bytes.push(self.state.no_progress_streak);
        bytes.extend(self.state.step_count.to_le_bytes());

        digest_meta("UCF:MC:DOPA", &bytes)
    }

    fn config_digest(&self) -> [u8; 32] {
        digest_config("UCF:MC:DOPA:CFG", &self.config)
    }
}

impl DbmModule for DopaAttractorMicrocircuit {
    type Input = DopaInput;
    type Output = DopaOutput;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        self.step(input, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use microcircuit_dopamin_stub::DopaRules;

    fn base_input() -> DopaInput {
        DopaInput {
            integrity: IntegrityState::Ok,
            threat: LevelClass::Low,
            policy_pressure: LevelClass::Low,
            ..Default::default()
        }
    }

    fn assert_not_more_permissive(micro: &DopaOutput, rules: &DopaOutput) {
        fn severity(level: LevelClass) -> u8 {
            match level {
                LevelClass::Low => 0,
                LevelClass::Med => 1,
                LevelClass::High => 2,
            }
        }

        assert!(
            severity(micro.progress) <= severity(rules.progress),
            "micro progress {:?} > rules {:?}",
            micro.progress,
            rules.progress
        );
        assert!(
            severity(micro.incentive_focus) <= severity(rules.incentive_focus),
            "micro incentive {:?} > rules {:?}",
            micro.incentive_focus,
            rules.incentive_focus
        );
    }

    #[test]
    fn determinism_for_repeated_sequence() {
        let mut circuit_a = DopaAttractorMicrocircuit::new(CircuitConfig::default());
        let mut circuit_b = DopaAttractorMicrocircuit::new(CircuitConfig::default());
        let inputs = vec![
            DopaInput {
                exec_success_count_medium: 2,
                exec_failure_count_medium: 1,
                ..base_input()
            },
            DopaInput {
                exec_failure_count_medium: 2,
                ..base_input()
            },
            DopaInput {
                deny_count_medium: 5,
                ..base_input()
            },
        ];

        for input in inputs {
            let out_a = circuit_a.step(&input, 0);
            let out_b = circuit_b.step(&input, 0);
            assert_eq!(out_a, out_b);
        }
    }

    #[test]
    fn reward_block_latch_sustains_suppression() {
        let mut circuit = DopaAttractorMicrocircuit::new(CircuitConfig::default());

        let output_block = circuit.step(
            &DopaInput {
                integrity: IntegrityState::Fail,
                ..base_input()
            },
            0,
        );
        assert!(output_block.reward_block);

        let output_latch = circuit.step(&base_input(), 0);
        assert!(output_latch.reward_block);

        let output_cleared = circuit.step(&base_input(), 0);
        assert!(!output_cleared.reward_block);
    }

    #[test]
    fn replay_hint_triggers_after_three_no_progress_ticks() {
        let mut circuit = DopaAttractorMicrocircuit::new(CircuitConfig::default());

        for _ in 0..2 {
            let output = circuit.step(
                &DopaInput {
                    exec_failure_count_medium: 1,
                    ..base_input()
                },
                0,
            );
            assert!(!output.replay_hint);
        }

        let output = circuit.step(
            &DopaInput {
                exec_failure_count_medium: 1,
                ..base_input()
            },
            0,
        );
        assert!(output.replay_hint);
    }

    #[test]
    fn invariants_hold_for_critical_inputs() {
        let cases = vec![
            DopaInput {
                integrity: IntegrityState::Fail,
                ..base_input()
            },
            DopaInput {
                threat: LevelClass::High,
                ..base_input()
            },
            DopaInput {
                receipt_invalid_present: true,
                ..base_input()
            },
            DopaInput {
                dlp_critical_present: true,
                ..base_input()
            },
            DopaInput {
                replay_mismatch_present: true,
                ..base_input()
            },
        ];

        for input in cases {
            let mut rules = DopaRules::new();
            let mut micro = DopaAttractorMicrocircuit::new(CircuitConfig::default());

            let rules_output = rules.tick(&input);
            let micro_output = micro.step(&input, 0);

            assert_not_more_permissive(&micro_output, &rules_output);
            if rules_output.reward_block {
                assert!(micro_output.reward_block);
            }
        }
    }
}
