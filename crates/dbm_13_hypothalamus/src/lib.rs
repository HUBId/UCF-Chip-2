#![forbid(unsafe_code)]

use dbm_core::{
    BaselineVector, DbmModule, IntegrityState, IsvSnapshot, LevelClass, OverlaySet, ProfileState,
    ReasonSet, ThreatVector,
};
#[cfg(any(
    feature = "microcircuit-hypothalamus-setpoint",
    feature = "biophys-l4-hypothalamus"
))]
use microcircuit_core::CircuitConfig;
use microcircuit_core::MicrocircuitBackend;
pub use microcircuit_hypothalamus_setpoint::{HypoInput, HypoOutput};
use microcircuit_pag_stub::DefensePattern;
use microcircuit_pmrf_stub::SequenceMode;
use std::fmt;

#[derive(Debug, Clone, Default)]
pub struct HypothalamusInput {
    pub isv: IsvSnapshot,
    pub pag_pattern: Option<DefensePattern>,
    pub stn_hold_active: bool,
    pub pmrf_sequence_mode: SequenceMode,
    pub baseline: BaselineVector,
    pub unlock_present: bool,
    pub unlock_ready: bool,
    pub now_ms: u64,
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

pub enum HypothalamusBackend {
    Rules(HypothalamusRules),
    Micro(Box<dyn MicrocircuitBackend<HypoInput, HypoOutput>>),
}

impl fmt::Debug for HypothalamusBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HypothalamusBackend::Rules(_) => f.write_str("HypothalamusBackend::Rules"),
            HypothalamusBackend::Micro(_) => f.write_str("HypothalamusBackend::Micro"),
        }
    }
}

impl HypothalamusBackend {
    fn tick(&mut self, input: &HypothalamusInput) -> ControlDecision {
        match self {
            HypothalamusBackend::Rules(rules) => rules.tick(input),
            HypothalamusBackend::Micro(backend) => {
                let hypo_input = HypoInput {
                    isv: input.isv.clone(),
                    pag_pattern: input.pag_pattern,
                    stn_hold_active: input.stn_hold_active,
                    pmrf_sequence_mode: input.pmrf_sequence_mode,
                    baseline: input.baseline.clone(),
                    unlock_present: input.unlock_present,
                    unlock_ready: input.unlock_ready,
                    now_ms: input.now_ms,
                };
                backend.step(&hypo_input, input.now_ms).into()
            }
        }
    }
}

#[derive(Debug)]
pub struct Hypothalamus {
    backend: HypothalamusBackend,
}

impl Hypothalamus {
    pub fn new() -> Self {
        #[cfg(feature = "biophys-l4-hypothalamus")]
        {
            use microcircuit_hypothalamus_l4::HypothalamusL4Microcircuit;

            return Self {
                backend: HypothalamusBackend::Micro(Box::new(
                    HypothalamusL4Microcircuit::new(CircuitConfig::default()),
                )),
            };
        }

        #[cfg(all(
            not(feature = "biophys-l4-hypothalamus"),
            feature = "microcircuit-hypothalamus-setpoint"
        ))]
        {
            use microcircuit_hypothalamus_setpoint::HypothalamusSetpointMicrocircuit;

            return Self {
                backend: HypothalamusBackend::Micro(Box::new(
                    HypothalamusSetpointMicrocircuit::new(CircuitConfig::default()),
                )),
            };
        }

        Self {
            backend: HypothalamusBackend::Rules(HypothalamusRules::new()),
        }
    }

    #[cfg(feature = "microcircuit-hypothalamus-setpoint")]
    pub fn new_micro(config: CircuitConfig) -> Self {
        use microcircuit_hypothalamus_setpoint::HypothalamusSetpointMicrocircuit;

        Self {
            backend: HypothalamusBackend::Micro(Box::new(HypothalamusSetpointMicrocircuit::new(
                config,
            ))),
        }
    }

    pub fn snapshot_digest(&self) -> Option<[u8; 32]> {
        match &self.backend {
            HypothalamusBackend::Micro(backend) => Some(backend.snapshot_digest()),
            HypothalamusBackend::Rules(_) => None,
        }
    }

    pub fn config_digest(&self) -> Option<[u8; 32]> {
        match &self.backend {
            HypothalamusBackend::Micro(backend) => Some(backend.config_digest()),
            HypothalamusBackend::Rules(_) => None,
        }
    }
}

impl Default for Hypothalamus {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Default)]
pub struct HypothalamusRules {}

impl HypothalamusRules {
    pub fn new() -> Self {
        Self {}
    }
}

impl DbmModule for HypothalamusRules {
    type Input = HypothalamusInput;
    type Output = ControlDecision;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        let mut decision = ControlDecision {
            reason_codes: input.isv.dominant_reason_codes.clone(),
            ..ControlDecision::default()
        };

        if input.baseline.export_strictness == LevelClass::High {
            decision.overlays.export_lock = true;
            decision
                .reason_codes
                .insert("baseline_export_lock".to_string());
        }

        if input.baseline.chain_conservatism == LevelClass::High {
            decision.overlays.simulate_first = true;
            decision
                .reason_codes
                .insert("baseline_chain_conservatism".to_string());
        }

        if input.baseline.novelty_dampening == LevelClass::High {
            decision.overlays.novelty_lock = true;
            decision
                .reason_codes
                .insert("baseline_novelty_dampening".to_string());
        }

        if input.baseline.approval_strictness == LevelClass::High {
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

impl DbmModule for Hypothalamus {
    type Input = HypothalamusInput;
    type Output = ControlDecision;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        self.backend.tick(input)
    }
}

impl From<HypoOutput> for ControlDecision {
    fn from(output: HypoOutput) -> Self {
        Self {
            profile_state: output.profile_state,
            overlays: output.overlays,
            deescalation_lock: output.deescalation_lock,
            cooldown_class: output.cooldown_class,
            reason_codes: output.reason_codes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_isv() -> IsvSnapshot {
        IsvSnapshot::default()
    }

    fn base_input() -> HypothalamusInput {
        HypothalamusInput {
            isv: base_isv(),
            ..HypothalamusInput::default()
        }
    }

    #[test]
    fn integrity_fail_forces_m3() {
        let mut module = HypothalamusRules::new();
        let isv = IsvSnapshot {
            integrity: IntegrityState::Fail,
            ..IsvSnapshot::default()
        };
        let decision = module.tick(&HypothalamusInput {
            isv,
            ..base_input()
        });

        assert_eq!(decision.profile_state, ProfileState::M3);
        assert!(decision.overlays.simulate_first);
        assert!(decision.deescalation_lock);
    }

    #[test]
    fn threat_high_pushes_m2() {
        let mut module = HypothalamusRules::new();
        let isv = IsvSnapshot {
            threat: LevelClass::High,
            ..base_isv()
        };
        let decision = module.tick(&HypothalamusInput {
            isv,
            ..base_input()
        });

        assert_eq!(decision.profile_state, ProfileState::M2);
        assert!(decision.overlays.export_lock);
    }

    #[test]
    fn policy_pressure_high_sets_m1() {
        let mut module = HypothalamusRules::new();
        let isv = IsvSnapshot {
            policy_pressure: LevelClass::High,
            ..base_isv()
        };

        let decision = module.tick(&HypothalamusInput {
            isv,
            ..base_input()
        });

        assert_eq!(decision.profile_state, ProfileState::M1);
        assert!(decision.deescalation_lock);
    }

    #[test]
    fn calm_state_remains_m0() {
        let mut module = HypothalamusRules::new();
        let decision = module.tick(&HypothalamusInput { ..base_input() });

        assert_eq!(decision.profile_state, ProfileState::M0);
        assert!(!decision.overlays.export_lock);
    }

    #[test]
    fn tool_side_effects_enforce_simulate_first() {
        let mut module = HypothalamusRules::new();
        let mut isv = base_isv();
        isv.threat_vectors = Some(vec![ThreatVector::ToolSideEffects]);

        let decision = module.tick(&HypothalamusInput {
            isv,
            ..base_input()
        });

        assert!(decision.overlays.simulate_first);
        assert!(decision.profile_state == ProfileState::M1);
    }

    #[cfg(feature = "microcircuit-hypothalamus-setpoint")]
    mod microcircuit_invariants {
        use super::*;
        use microcircuit_core::CircuitConfig;
        use microcircuit_pag_stub::DefensePattern;

        fn profile_rank(profile: ProfileState) -> u8 {
            match profile {
                ProfileState::M0 => 0,
                ProfileState::M1 => 1,
                ProfileState::M2 => 2,
                ProfileState::M3 => 3,
            }
        }

        fn base_input() -> HypothalamusInput {
            HypothalamusInput {
                isv: IsvSnapshot::default(),
                baseline: BaselineVector::default(),
                ..HypothalamusInput::default()
            }
        }

        fn assert_micro_not_less_strict(input: HypothalamusInput) {
            let mut rules = HypothalamusRules::new();
            let mut micro = Hypothalamus::new_micro(CircuitConfig::default());

            let rules_output = rules.tick(&input);
            let micro_output = micro.tick(&input);

            assert!(
                profile_rank(micro_output.profile_state)
                    >= profile_rank(rules_output.profile_state)
            );
        }

        #[test]
        fn integrity_fail_always_m3() {
            let input = HypothalamusInput {
                isv: IsvSnapshot {
                    integrity: IntegrityState::Fail,
                    ..IsvSnapshot::default()
                },
                ..base_input()
            };
            let mut micro = Hypothalamus::new_micro(CircuitConfig::default());
            let output = micro.tick(&input);
            assert_eq!(output.profile_state, ProfileState::M3);
        }

        #[test]
        fn threat_high_with_exfil_stays_m2_or_higher() {
            let input = HypothalamusInput {
                isv: IsvSnapshot {
                    threat: LevelClass::High,
                    threat_vectors: Some(vec![ThreatVector::Exfil]),
                    ..IsvSnapshot::default()
                },
                ..base_input()
            };
            let mut micro = Hypothalamus::new_micro(CircuitConfig::default());
            let output = micro.tick(&input);
            assert!(matches!(
                output.profile_state,
                ProfileState::M2 | ProfileState::M3
            ));
        }

        #[test]
        fn dp3_forces_m3_profile() {
            let input = HypothalamusInput {
                pag_pattern: Some(DefensePattern::DP3_FORENSIC),
                ..base_input()
            };
            let mut micro = Hypothalamus::new_micro(CircuitConfig::default());
            let output = micro.tick(&input);
            assert_eq!(output.profile_state, ProfileState::M3);
        }

        #[test]
        fn micro_not_less_strict_under_critical_conditions() {
            assert_micro_not_less_strict(HypothalamusInput {
                isv: IsvSnapshot {
                    integrity: IntegrityState::Fail,
                    ..IsvSnapshot::default()
                },
                ..base_input()
            });
            assert_micro_not_less_strict(HypothalamusInput {
                isv: IsvSnapshot {
                    threat: LevelClass::High,
                    ..IsvSnapshot::default()
                },
                ..base_input()
            });
        }

        #[test]
        fn unlock_ready_allows_m3_to_m1_not_m0() {
            let mut micro = Hypothalamus::new_micro(CircuitConfig::default());
            let fail_input = HypothalamusInput {
                isv: IsvSnapshot {
                    integrity: IntegrityState::Fail,
                    ..IsvSnapshot::default()
                },
                ..base_input()
            };
            let output = micro.tick(&fail_input);
            assert_eq!(output.profile_state, ProfileState::M3);

            let unlock_input = HypothalamusInput {
                unlock_present: true,
                unlock_ready: true,
                isv: IsvSnapshot::default(),
                ..base_input()
            };
            let output = micro.tick(&unlock_input);
            assert_eq!(output.profile_state, ProfileState::M1);
        }
    }
}
