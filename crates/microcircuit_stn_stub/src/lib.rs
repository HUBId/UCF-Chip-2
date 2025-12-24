#![forbid(unsafe_code)]

use biophys_core::ModulatorField;
use dbm_core::{DbmModule, IntegrityState, LevelClass, ReasonSet};

#[derive(Debug, Clone, Default)]
pub struct StnInput {
    pub policy_pressure: LevelClass,
    pub arousal: LevelClass,
    pub threat: LevelClass,
    pub receipt_invalid_present: bool,
    pub dlp_critical_present: bool,
    pub integrity: IntegrityState,
    pub tool_side_effects_present: bool,
    pub cerebellum_divergence: LevelClass,
    pub modulators: ModulatorField,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StnOutput {
    pub hold_active: bool,
    pub hold_reason_codes: ReasonSet,
    pub hint_simulate_first: bool,
    pub hint_novelty_lock: bool,
    pub hint_export_lock: bool,
}

#[derive(Debug, Default)]
pub struct StnRules {}

impl StnRules {
    pub fn new() -> Self {
        Self {}
    }
}

impl DbmModule for StnRules {
    type Input = StnInput;
    type Output = StnOutput;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        let mut hold_reason_codes = ReasonSet::default();

        let arousal_threat_lock =
            input.arousal == LevelClass::High && level_at_least(input.threat, LevelClass::Med);

        let integrity_block = input.integrity != IntegrityState::Ok;
        let tool_side_effects_hold = (input.tool_side_effects_present
            || input.cerebellum_divergence == LevelClass::High)
            && (level_at_least(input.policy_pressure, LevelClass::Med)
                || input.arousal == LevelClass::High);

        let hold_active = input.receipt_invalid_present
            || input.dlp_critical_present
            || integrity_block
            || input.policy_pressure == LevelClass::High
            || arousal_threat_lock
            || tool_side_effects_hold;

        if hold_active {
            hold_reason_codes.insert("RC.GV.HOLD.ON");
        }

        let hint_simulate_first = hold_active;
        let hint_novelty_lock =
            input.policy_pressure == LevelClass::High || input.arousal == LevelClass::High;
        let hint_export_lock = input.dlp_critical_present;

        StnOutput {
            hold_active,
            hold_reason_codes,
            hint_simulate_first,
            hint_novelty_lock,
            hint_export_lock,
        }
    }
}

fn level_at_least(value: LevelClass, threshold: LevelClass) -> bool {
    (value as i32) >= (threshold as i32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use biophys_core::ModulatorField;

    fn base_input() -> StnInput {
        StnInput {
            policy_pressure: LevelClass::Low,
            arousal: LevelClass::Low,
            threat: LevelClass::Low,
            receipt_invalid_present: false,
            dlp_critical_present: false,
            integrity: IntegrityState::Ok,
            tool_side_effects_present: false,
            cerebellum_divergence: LevelClass::Low,
            modulators: ModulatorField::default(),
        }
    }

    #[test]
    fn receipt_invalid_triggers_hold() {
        let mut stn = StnRules::new();
        let output = stn.tick(&StnInput {
            receipt_invalid_present: true,
            ..base_input()
        });

        assert!(output.hold_active);
        assert!(output.hint_simulate_first);
        assert!(output
            .hold_reason_codes
            .codes
            .contains(&"RC.GV.HOLD.ON".to_string()));
    }

    #[test]
    fn policy_pressure_high_triggers_hold_and_novelty_lock() {
        let mut stn = StnRules::new();
        let output = stn.tick(&StnInput {
            policy_pressure: LevelClass::High,
            ..base_input()
        });

        assert!(output.hold_active);
        assert!(output.hint_novelty_lock);
        assert!(output.hint_simulate_first);
    }

    #[test]
    fn arousal_and_threat_trigger_hold_without_receipt_invalid() {
        let mut stn = StnRules::new();
        let output = stn.tick(&StnInput {
            arousal: LevelClass::High,
            threat: LevelClass::Med,
            ..base_input()
        });

        assert!(output.hold_active);
    }

    #[test]
    fn tool_side_effects_hold_with_policy_pressure() {
        let mut stn = StnRules::new();
        let output = stn.tick(&StnInput {
            tool_side_effects_present: true,
            policy_pressure: LevelClass::Med,
            ..base_input()
        });

        assert!(output.hold_active);
        assert!(output.hint_simulate_first);
    }
}
