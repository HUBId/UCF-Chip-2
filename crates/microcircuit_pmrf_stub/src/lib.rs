#![forbid(unsafe_code)]

use dbm_core::{DbmModule, LevelClass, ReasonSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SequenceMode {
    #[default]
    Normal,
    Slow,
    SplitRequired,
}

#[derive(Debug, Clone, Default)]
pub struct PmrfInput {
    pub divergence: LevelClass,
    pub policy_pressure: LevelClass,
    pub stability: LevelClass,
    pub hold_active: bool,
    pub budget_stress: LevelClass,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PmrfOutput {
    pub sequence_mode: SequenceMode,
    pub chain_tightening: bool,
    pub checkpoint_required: bool,
    pub reason_codes: ReasonSet,
}

impl Default for PmrfOutput {
    fn default() -> Self {
        Self {
            sequence_mode: SequenceMode::Normal,
            chain_tightening: false,
            checkpoint_required: false,
            reason_codes: ReasonSet::default(),
        }
    }
}

#[derive(Debug, Default)]
pub struct PmrfRules {}

impl PmrfRules {
    pub fn new() -> Self {
        Self {}
    }
}

impl DbmModule for PmrfRules {
    type Input = PmrfInput;
    type Output = PmrfOutput;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        let mut reason_codes = ReasonSet::default();

        if input.divergence == LevelClass::High {
            reason_codes.insert("RC.GV.SEQUENCE.SPLIT_REQUIRED");
            return PmrfOutput {
                sequence_mode: SequenceMode::SplitRequired,
                chain_tightening: true,
                checkpoint_required: true,
                reason_codes,
            };
        }

        let slow_path = input.hold_active
            || input.policy_pressure == LevelClass::High
            || input.budget_stress == LevelClass::High;

        if slow_path {
            reason_codes.insert("RC.GV.SEQUENCE.SLOW");
            return PmrfOutput {
                sequence_mode: SequenceMode::Slow,
                chain_tightening: true,
                checkpoint_required: false,
                reason_codes,
            };
        }

        PmrfOutput {
            sequence_mode: SequenceMode::Normal,
            chain_tightening: false,
            checkpoint_required: false,
            reason_codes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_input() -> PmrfInput {
        PmrfInput {
            divergence: LevelClass::Low,
            policy_pressure: LevelClass::Low,
            stability: LevelClass::Low,
            hold_active: false,
            budget_stress: LevelClass::Low,
        }
    }

    #[test]
    fn divergence_high_triggers_split() {
        let mut pmrf = PmrfRules::new();
        let output = pmrf.tick(&PmrfInput {
            divergence: LevelClass::High,
            ..base_input()
        });

        assert_eq!(output.sequence_mode, SequenceMode::SplitRequired);
        assert!(output.chain_tightening);
        assert!(output.checkpoint_required);
    }

    #[test]
    fn hold_active_triggers_slow_mode() {
        let mut pmrf = PmrfRules::new();
        let output = pmrf.tick(&PmrfInput {
            hold_active: true,
            ..base_input()
        });

        assert_eq!(output.sequence_mode, SequenceMode::Slow);
        assert!(output.chain_tightening);
        assert!(!output.checkpoint_required);
    }

    #[test]
    fn normal_mode_without_triggers() {
        let mut pmrf = PmrfRules::new();
        let output = pmrf.tick(&base_input());

        assert_eq!(output.sequence_mode, SequenceMode::Normal);
        assert!(!output.chain_tightening);
        assert!(!output.checkpoint_required);
        assert!(output.reason_codes.codes.is_empty());
    }
}
