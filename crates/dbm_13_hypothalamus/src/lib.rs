#![forbid(unsafe_code)]

use dbm_core::{
    DbmModule, IntegrityState, IsvSnapshot, LevelClass, OverlaySet, ProfileState, ReasonSet,
    ThreatVector,
};

#[derive(Debug, Clone, Default)]
pub struct HypothalamusInput {
    pub isv: IsvSnapshot,
    pub export_lock_bias: bool,
    pub simulate_first_bias: bool,
    pub approval_strict: bool,
    pub novelty_lock_bias: bool,
    pub cerebellum_divergence: LevelClass,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlDecision {
    pub profile_state: ProfileState,
    pub overlays: OverlaySet,
    pub deescalation_lock: bool,
    pub cooldown_class: LevelClass,
    pub reason_codes: ReasonSet,
}

impl Default for ControlDecision {
    fn default() -> Self {
        Self {
            profile_state: ProfileState::M0,
            overlays: OverlaySet::default(),
            deescalation_lock: false,
            cooldown_class: LevelClass::Low,
            reason_codes: ReasonSet::default(),
        }
    }
}

#[derive(Debug, Default)]
pub struct Hypothalamus {}

impl Hypothalamus {
    pub fn new() -> Self {
        Self {}
    }
}

impl DbmModule for Hypothalamus {
    type Input = HypothalamusInput;
    type Output = ControlDecision;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        let mut decision = ControlDecision {
            reason_codes: input.isv.dominant_reason_codes.clone(),
            ..ControlDecision::default()
        };

        if input.export_lock_bias {
            decision.overlays.export_lock = true;
            decision
                .reason_codes
                .insert("baseline_export_lock".to_string());
        }

        if input.simulate_first_bias {
            decision.overlays.simulate_first = true;
            decision
                .reason_codes
                .insert("baseline_chain_conservatism".to_string());
        }

        if input.novelty_lock_bias {
            decision.overlays.novelty_lock = true;
            decision
                .reason_codes
                .insert("baseline_novelty_dampening".to_string());
        }

        if input.approval_strict {
            decision
                .reason_codes
                .insert("baseline_approval_strict".to_string());
        }

        if input.isv.integrity == IntegrityState::Fail {
            decision.profile_state = ProfileState::M3;
            decision.overlays = OverlaySet::all_enabled();
            decision.deescalation_lock = true;
            decision.cooldown_class = LevelClass::High;
            decision.reason_codes.insert("integrity_fail".to_string());
            return decision;
        }

        if input.isv.threat == LevelClass::High {
            decision.profile_state = ProfileState::M2;
            decision.overlays = OverlaySet::all_enabled();
            decision.deescalation_lock = true;
            decision.cooldown_class = LevelClass::High;
            decision.reason_codes.insert("threat_high".to_string());
            return decision;
        }

        if input.isv.policy_pressure == LevelClass::High {
            decision.profile_state = ProfileState::M1;
            decision.overlays = OverlaySet::all_enabled();
            decision.deescalation_lock = true;
            decision.cooldown_class = LevelClass::Med;
            decision
                .reason_codes
                .insert("policy_pressure_high".to_string());
            return decision;
        }

        let tool_side_effects = input
            .isv
            .threat_vectors
            .as_ref()
            .is_some_and(|vectors| vectors.contains(&ThreatVector::ToolSideEffects));

        if tool_side_effects || input.cerebellum_divergence == LevelClass::High {
            if !matches!(decision.profile_state, ProfileState::M2 | ProfileState::M3) {
                decision.profile_state = ProfileState::M1;
            }
            decision.overlays.simulate_first = true;
            decision
                .reason_codes
                .insert("RC.TH.TOOL_SIDE_EFFECTS".to_string());
            return decision;
        }

        decision.reason_codes.insert("baseline".to_string());
        decision
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_isv() -> IsvSnapshot {
        IsvSnapshot::default()
    }

    #[test]
    fn integrity_fail_forces_m3() {
        let mut module = Hypothalamus::new();
        let isv = IsvSnapshot {
            integrity: IntegrityState::Fail,
            ..IsvSnapshot::default()
        };
        let decision = module.tick(&HypothalamusInput {
            isv,
            ..Default::default()
        });

        assert_eq!(decision.profile_state, ProfileState::M3);
        assert!(decision.overlays.simulate_first);
        assert!(decision.deescalation_lock);
    }

    #[test]
    fn threat_high_pushes_m2() {
        let mut module = Hypothalamus::new();
        let isv = IsvSnapshot {
            threat: LevelClass::High,
            ..base_isv()
        };
        let decision = module.tick(&HypothalamusInput {
            isv,
            ..Default::default()
        });

        assert_eq!(decision.profile_state, ProfileState::M2);
        assert!(decision.overlays.export_lock);
    }

    #[test]
    fn policy_pressure_high_sets_m1() {
        let mut module = Hypothalamus::new();
        let isv = IsvSnapshot {
            policy_pressure: LevelClass::High,
            ..base_isv()
        };

        let decision = module.tick(&HypothalamusInput {
            isv,
            ..Default::default()
        });

        assert_eq!(decision.profile_state, ProfileState::M1);
        assert!(decision.deescalation_lock);
    }

    #[test]
    fn calm_state_remains_m0() {
        let mut module = Hypothalamus::new();
        let decision = module.tick(&HypothalamusInput {
            isv: base_isv(),
            ..Default::default()
        });

        assert_eq!(decision.profile_state, ProfileState::M0);
        assert!(!decision.overlays.export_lock);
    }

    #[test]
    fn tool_side_effects_enforce_simulate_first() {
        let mut module = Hypothalamus::new();
        let mut isv = base_isv();
        isv.threat_vectors = Some(vec![ThreatVector::ToolSideEffects]);

        let decision = module.tick(&HypothalamusInput {
            isv,
            ..Default::default()
        });

        assert!(decision.overlays.simulate_first);
        assert!(decision.profile_state == ProfileState::M1);
    }
}
