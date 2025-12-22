#![forbid(unsafe_code)]

use dbm_core::{
    DbmModule, DwmMode, EmotionField, IntegrityState, IsvSnapshot, LevelClass, NoiseClass,
    OverlaySet, PriorityClass, ProfileState, ReasonSet, RecursionDepthClass,
};
use dbm_pag::DefensePattern;

#[derive(Debug, Clone)]
pub struct EmotionFieldInput {
    pub isv: IsvSnapshot,
    pub dwm: DwmMode,
    pub profile: ProfileState,
    pub overlays: OverlaySet,
    pub reward_block: bool,
    pub defense_pattern: Option<DefensePattern>,
    pub replay_hint: bool,
}

impl Default for EmotionFieldInput {
    fn default() -> Self {
        Self {
            isv: IsvSnapshot::default(),
            dwm: DwmMode::ExecPlan,
            profile: ProfileState::M0,
            overlays: OverlaySet::default(),
            reward_block: false,
            defense_pattern: None,
            replay_hint: false,
        }
    }
}

#[derive(Debug, Default)]
pub struct EmotionFieldModule {}

impl EmotionFieldModule {
    pub fn new() -> Self {
        Self {}
    }

    fn noise_class(isv: &IsvSnapshot, reasons: &mut ReasonSet) -> NoiseClass {
        if isv.integrity != IntegrityState::Ok
            || isv.threat == LevelClass::High
            || isv.stability == LevelClass::High
        {
            if isv.threat == LevelClass::High {
                reasons.insert("RC.RG.STATE.THREAT_UP");
            }

            if isv.stability == LevelClass::High {
                reasons.insert("RC.RG.STATE.STABILITY_UP");
            }

            return NoiseClass::High;
        }

        if matches!(isv.policy_pressure, LevelClass::Med | LevelClass::High)
            || matches!(isv.arousal, LevelClass::Med | LevelClass::High)
        {
            if isv.arousal == LevelClass::High {
                reasons.insert("RC.RG.STATE.AROUSAL_UP");
            }

            return NoiseClass::Med;
        }

        NoiseClass::Low
    }

    fn priority_class(input: &EmotionFieldInput, reasons: &mut ReasonSet) -> PriorityClass {
        let defense_priority = matches!(
            input.defense_pattern,
            Some(DefensePattern::DP2_QUARANTINE | DefensePattern::DP3_FORENSIC)
        );

        if input.isv.threat == LevelClass::High || input.replay_hint || defense_priority {
            if input.isv.threat == LevelClass::High {
                reasons.insert("RC.RG.STATE.THREAT_UP");
            }

            return PriorityClass::High;
        }

        if input.reward_block {
            reasons.insert("RC.GV.PROGRESS.REWARD_BLOCKED");
        }

        PriorityClass::Med
    }

    fn recursion_depth_class(isv: &IsvSnapshot, reasons: &mut ReasonSet) -> RecursionDepthClass {
        if isv.integrity != IntegrityState::Ok
            || isv.threat == LevelClass::High
            || isv.arousal == LevelClass::High
        {
            if isv.threat == LevelClass::High {
                reasons.insert("RC.RG.STATE.THREAT_UP");
            }

            if isv.arousal == LevelClass::High {
                reasons.insert("RC.RG.STATE.AROUSAL_UP");
            }

            return RecursionDepthClass::Low;
        }

        if isv.stability == LevelClass::High {
            reasons.insert("RC.RG.STATE.STABILITY_UP");
            return RecursionDepthClass::Med;
        }

        RecursionDepthClass::High
    }
}

impl DbmModule for EmotionFieldModule {
    type Input = EmotionFieldInput;
    type Output = EmotionField;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        let mut reasons = ReasonSet::default();

        let noise = Self::noise_class(&input.isv, &mut reasons);
        let priority = Self::priority_class(input, &mut reasons);
        let recursion_depth = Self::recursion_depth_class(&input.isv, &mut reasons);

        if input.reward_block {
            reasons.insert("RC.GV.PROGRESS.REWARD_BLOCKED");
        }

        EmotionField {
            noise,
            priority,
            recursion_depth,
            dwm: input.dwm,
            profile: input.profile,
            overlays: input.overlays.clone(),
            reason_codes: reasons,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_input() -> EmotionFieldInput {
        EmotionFieldInput {
            isv: IsvSnapshot::default(),
            dwm: DwmMode::ExecPlan,
            profile: ProfileState::M0,
            overlays: OverlaySet::default(),
            reward_block: false,
            defense_pattern: None,
            replay_hint: false,
        }
    }

    #[test]
    fn integrity_fail_drives_noise_high_and_recursion_low() {
        let mut module = EmotionFieldModule::new();
        let input = EmotionFieldInput {
            isv: IsvSnapshot {
                integrity: IntegrityState::Fail,
                ..IsvSnapshot::default()
            },
            ..base_input()
        };

        let output = module.tick(&input);
        assert_eq!(output.noise, NoiseClass::High);
        assert_eq!(output.recursion_depth, RecursionDepthClass::Low);
    }

    #[test]
    fn high_threat_escalates_priority_and_noise_and_recursion() {
        let mut module = EmotionFieldModule::new();
        let input = EmotionFieldInput {
            isv: IsvSnapshot {
                threat: LevelClass::High,
                ..IsvSnapshot::default()
            },
            ..base_input()
        };

        let output = module.tick(&input);
        assert_eq!(output.noise, NoiseClass::High);
        assert_eq!(output.priority, PriorityClass::High);
        assert_eq!(output.recursion_depth, RecursionDepthClass::Low);
        assert!(output
            .reason_codes
            .codes
            .contains(&"RC.RG.STATE.THREAT_UP".to_string()));
    }

    #[test]
    fn reward_block_raises_priority_floor() {
        let mut module = EmotionFieldModule::new();
        let input = EmotionFieldInput {
            reward_block: true,
            ..base_input()
        };

        let output = module.tick(&input);
        assert!(matches!(
            output.priority,
            PriorityClass::Med | PriorityClass::High
        ));
        assert!(output
            .reason_codes
            .codes
            .contains(&"RC.GV.PROGRESS.REWARD_BLOCKED".to_string()));
    }

    #[test]
    fn reason_codes_are_deterministic_and_sorted() {
        let mut module = EmotionFieldModule::new();
        let input = EmotionFieldInput {
            isv: IsvSnapshot {
                threat: LevelClass::High,
                arousal: LevelClass::High,
                stability: LevelClass::High,
                ..IsvSnapshot::default()
            },
            reward_block: true,
            ..base_input()
        };

        let output = module.tick(&input);
        let mut expected = vec![
            "RC.GV.PROGRESS.REWARD_BLOCKED".to_string(),
            "RC.RG.STATE.AROUSAL_UP".to_string(),
            "RC.RG.STATE.STABILITY_UP".to_string(),
            "RC.RG.STATE.THREAT_UP".to_string(),
        ];
        expected.sort();
        assert_eq!(output.reason_codes.codes, expected);
    }

    #[test]
    fn deterministic_output_for_same_inputs() {
        let mut module = EmotionFieldModule::new();
        let input = EmotionFieldInput {
            isv: IsvSnapshot {
                stability: LevelClass::High,
                ..IsvSnapshot::default()
            },
            ..base_input()
        };

        let output_a = module.tick(&input);
        let output_b = module.tick(&input);
        assert_eq!(output_a, output_b);
    }
}
