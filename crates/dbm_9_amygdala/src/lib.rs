#![forbid(unsafe_code)]

use dbm_core::DbmModule;
pub use microcircuit_amygdala_stub::{AmyInput, AmyOutput, AmyRules};
#[cfg(any(feature = "microcircuit-amygdala-pop", feature = "biophys-amygdala"))]
use microcircuit_core::CircuitConfig;
use microcircuit_core::MicrocircuitBackend;
use std::fmt;

pub enum AmyBackend {
    Rules(AmyRules),
    Micro(Box<dyn MicrocircuitBackend<AmyInput, AmyOutput>>),
}

impl fmt::Debug for AmyBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AmyBackend::Rules(_) => f.write_str("AmyBackend::Rules"),
            AmyBackend::Micro(_) => f.write_str("AmyBackend::Micro"),
        }
    }
}

impl AmyBackend {
    fn tick(&mut self, input: &AmyInput) -> AmyOutput {
        match self {
            AmyBackend::Rules(rules) => rules.tick(input),
            AmyBackend::Micro(backend) => backend.step(input, 0),
        }
    }
}

#[derive(Debug)]
pub struct Amygdala {
    backend: AmyBackend,
}

impl Amygdala {
    pub fn new() -> Self {
        #[cfg(feature = "biophys-amygdala")]
        {
            Self::new_biophys(CircuitConfig::default())
        }
        #[cfg(all(
            feature = "microcircuit-amygdala-pop",
            not(feature = "biophys-amygdala")
        ))]
        {
            Self::new_micro(CircuitConfig::default())
        }
        #[cfg(all(
            not(feature = "microcircuit-amygdala-pop"),
            not(feature = "biophys-amygdala")
        ))]
        {
            Self {
                backend: AmyBackend::Rules(AmyRules::new()),
            }
        }
    }

    #[cfg(feature = "microcircuit-amygdala-pop")]
    pub fn new_micro(config: CircuitConfig) -> Self {
        use microcircuit_amygdala_pop::AmygdalaPopMicrocircuit;

        Self {
            backend: AmyBackend::Micro(Box::new(AmygdalaPopMicrocircuit::new(config))),
        }
    }

    #[cfg(feature = "biophys-amygdala")]
    pub fn new_biophys(config: CircuitConfig) -> Self {
        use microcircuit_amygdala_biophys::AmygdalaBiophysMicrocircuit;

        Self {
            backend: AmyBackend::Micro(Box::new(AmygdalaBiophysMicrocircuit::new(config))),
        }
    }

    pub fn snapshot_digest(&self) -> Option<[u8; 32]> {
        match &self.backend {
            AmyBackend::Micro(backend) => Some(backend.snapshot_digest()),
            AmyBackend::Rules(_) => None,
        }
    }

    pub fn config_digest(&self) -> Option<[u8; 32]> {
        match &self.backend {
            AmyBackend::Micro(backend) => Some(backend.config_digest()),
            AmyBackend::Rules(_) => None,
        }
    }
}

impl Default for Amygdala {
    fn default() -> Self {
        Self::new()
    }
}

impl DbmModule for Amygdala {
    type Input = AmyInput;
    type Output = AmyOutput;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        self.backend.tick(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dbm_core::{IntegrityState, LevelClass, ThreatVector};

    fn base_input() -> AmyInput {
        AmyInput {
            integrity: IntegrityState::Ok,
            replay_mismatch_present: false,
            dlp_secret_present: false,
            dlp_obfuscation_present: false,
            dlp_stegano_present: false,
            dlp_critical_count_med: 0,
            receipt_invalid_medium: 0,
            policy_pressure: LevelClass::Low,
            deny_storm_present: false,
            sealed: None,
            tool_anomaly_present: false,
            cerebellum_tool_anomaly_present: None,
            tool_anomalies: Vec::new(),
            divergence: LevelClass::Low,
        }
    }

    #[test]
    fn dlp_secret_triggers_exfil_and_high_threat() {
        let mut module = Amygdala::new();
        let output = module.tick(&AmyInput {
            dlp_secret_present: true,
            ..base_input()
        });

        assert_eq!(output.threat, LevelClass::High);
        assert!(output.vectors.contains(&ThreatVector::Exfil));
    }

    #[test]
    fn replay_mismatch_sets_integrity_compromise_and_high_threat() {
        let mut module = Amygdala::new();
        let output = module.tick(&AmyInput {
            replay_mismatch_present: true,
            ..base_input()
        });

        assert_eq!(output.threat, LevelClass::High);
        assert!(output.vectors.contains(&ThreatVector::IntegrityCompromise));
    }

    #[test]
    fn policy_pressure_high_marks_probing_and_med_threat() {
        let mut module = Amygdala::new();
        let output = module.tick(&AmyInput {
            policy_pressure: LevelClass::High,
            ..base_input()
        });

        assert_eq!(output.threat, LevelClass::Med);
        assert!(output.vectors.contains(&ThreatVector::Probing));
    }

    #[test]
    fn vector_order_is_deterministic() {
        let mut module = Amygdala::new();
        let output = module.tick(&AmyInput {
            replay_mismatch_present: true,
            dlp_secret_present: true,
            policy_pressure: LevelClass::High,
            ..base_input()
        });

        #[cfg(feature = "biophys-amygdala")]
        assert_eq!(
            output.vectors,
            vec![
                ThreatVector::IntegrityCompromise,
                ThreatVector::Exfil,
                ThreatVector::Probing,
            ]
        );
        #[cfg(not(feature = "biophys-amygdala"))]
        assert_eq!(
            output.vectors,
            vec![
                ThreatVector::Exfil,
                ThreatVector::Probing,
                ThreatVector::IntegrityCompromise,
            ]
        );
    }

    #[test]
    fn tool_side_effects_raise_threat_and_vector() {
        let mut module = Amygdala::new();
        let output = module.tick(&AmyInput {
            tool_anomaly_present: true,
            policy_pressure: LevelClass::Med,
            ..base_input()
        });

        assert_eq!(output.threat, LevelClass::Med);
        assert!(output.vectors.contains(&ThreatVector::ToolSideEffects));
        assert!(output
            .reason_codes
            .codes
            .contains(&"RC.TH.TOOL_SIDE_EFFECTS".to_string()));
    }

    #[cfg(feature = "microcircuit-amygdala-pop")]
    mod invariants {
        use super::*;
        use microcircuit_core::CircuitConfig;

        fn threat_rank(level: LevelClass) -> u8 {
            match level {
                LevelClass::Low => 0,
                LevelClass::Med => 1,
                LevelClass::High => 2,
            }
        }

        fn assert_not_less(micro: &AmyOutput, rules: &AmyOutput) {
            assert!(threat_rank(micro.threat) >= threat_rank(rules.threat));
        }

        #[test]
        fn dlp_critical_not_less_than_rules() {
            let mut rules = Amygdala::new();
            let mut micro = Amygdala::new_micro(CircuitConfig::default());
            let input = AmyInput {
                dlp_secret_present: true,
                dlp_critical_count_med: 1,
                ..base_input()
            };

            let rules_output = rules.tick(&input);
            let micro_output = micro.tick(&input);

            assert_not_less(&micro_output, &rules_output);
            assert!(micro_output.vectors.contains(&ThreatVector::Exfil));
            assert!(rules_output.vectors.contains(&ThreatVector::Exfil));
        }

        #[test]
        fn integrity_fail_not_less_than_rules() {
            let mut rules = Amygdala::new();
            let mut micro = Amygdala::new_micro(CircuitConfig::default());
            let input = AmyInput {
                integrity: IntegrityState::Fail,
                ..base_input()
            };

            let rules_output = rules.tick(&input);
            let micro_output = micro.tick(&input);

            assert_not_less(&micro_output, &rules_output);
            assert!(micro_output
                .vectors
                .contains(&ThreatVector::IntegrityCompromise));
            assert!(rules_output
                .vectors
                .contains(&ThreatVector::IntegrityCompromise));
        }

        #[test]
        fn receipt_invalid_not_less_than_rules() {
            let mut rules = Amygdala::new();
            let mut micro = Amygdala::new_micro(CircuitConfig::default());
            let input = AmyInput {
                receipt_invalid_medium: 1,
                ..base_input()
            };

            let mut rules_output = AmyOutput::default();
            let mut micro_output = AmyOutput::default();
            for _ in 0..25 {
                rules_output = rules.tick(&input);
                micro_output = micro.tick(&input);
            }

            assert_not_less(&micro_output, &rules_output);
        }
    }
}
