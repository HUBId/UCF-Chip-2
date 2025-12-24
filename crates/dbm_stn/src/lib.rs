#![forbid(unsafe_code)]

use dbm_core::DbmModule;
#[cfg(any(feature = "biophys-stn", feature = "microcircuit-stn-hold"))]
use microcircuit_core::CircuitConfig;
use microcircuit_core::MicrocircuitBackend;
pub use microcircuit_stn_stub::{StnInput, StnOutput, StnRules};
use std::fmt;

pub enum StnBackend {
    Rules(StnRules),
    Micro(Box<dyn MicrocircuitBackend<StnInput, StnOutput>>),
}

impl fmt::Debug for StnBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StnBackend::Rules(_) => f.write_str("StnBackend::Rules"),
            StnBackend::Micro(_) => f.write_str("StnBackend::Micro"),
        }
    }
}

impl StnBackend {
    fn tick(&mut self, input: &StnInput) -> StnOutput {
        match self {
            StnBackend::Rules(rules) => rules.tick(input),
            StnBackend::Micro(backend) => backend.step(input, 0),
        }
    }
}

#[derive(Debug)]
pub struct Stn {
    backend: StnBackend,
}

impl Stn {
    #[cfg(feature = "biophys-stn")]
    pub fn new() -> Self {
        use microcircuit_stn_biophys::StnBiophysMicrocircuit;

        Self {
            backend: StnBackend::Micro(Box::new(StnBiophysMicrocircuit::new(
                CircuitConfig::default(),
            ))),
        }
    }

    #[cfg(all(not(feature = "biophys-stn"), feature = "microcircuit-stn-hold"))]
    pub fn new() -> Self {
        use microcircuit_stn_hold::StnHoldMicrocircuit;

        Self {
            backend: StnBackend::Micro(Box::new(
                StnHoldMicrocircuit::new(CircuitConfig::default()),
            )),
        }
    }

    #[cfg(all(not(feature = "biophys-stn"), not(feature = "microcircuit-stn-hold")))]
    pub fn new() -> Self {
        Self {
            backend: StnBackend::Rules(StnRules::new()),
        }
    }

    #[cfg(feature = "microcircuit-stn-hold")]
    pub fn new_micro(config: CircuitConfig) -> Self {
        use microcircuit_stn_hold::StnHoldMicrocircuit;

        Self {
            backend: StnBackend::Micro(Box::new(StnHoldMicrocircuit::new(config))),
        }
    }

    #[cfg(feature = "biophys-stn")]
    pub fn new_biophys(config: CircuitConfig) -> Self {
        use microcircuit_stn_biophys::StnBiophysMicrocircuit;

        Self {
            backend: StnBackend::Micro(Box::new(StnBiophysMicrocircuit::new(config))),
        }
    }

    pub fn snapshot_digest(&self) -> Option<[u8; 32]> {
        match &self.backend {
            StnBackend::Micro(backend) => Some(backend.snapshot_digest()),
            StnBackend::Rules(_) => None,
        }
    }

    pub fn config_digest(&self) -> Option<[u8; 32]> {
        match &self.backend {
            StnBackend::Micro(backend) => Some(backend.config_digest()),
            StnBackend::Rules(_) => None,
        }
    }
}

impl Default for Stn {
    fn default() -> Self {
        Self::new()
    }
}

impl DbmModule for Stn {
    type Input = StnInput;
    type Output = StnOutput;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        self.backend.tick(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use biophys_core::ModulatorField;
    use dbm_core::{IntegrityState, LevelClass};

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
        let mut stn = Stn::new();
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

    #[cfg(not(feature = "biophys-stn"))]
    #[test]
    fn policy_pressure_high_triggers_hold_and_novelty_lock() {
        let mut stn = Stn::new();
        let output = stn.tick(&StnInput {
            policy_pressure: LevelClass::High,
            ..base_input()
        });

        assert!(output.hold_active);
        assert!(output.hint_novelty_lock);
        assert!(output.hint_simulate_first);
    }

    #[cfg(not(feature = "biophys-stn"))]
    #[test]
    fn arousal_and_threat_trigger_hold_without_receipt_invalid() {
        let mut stn = Stn::new();
        let output = stn.tick(&StnInput {
            arousal: LevelClass::High,
            threat: LevelClass::Med,
            ..base_input()
        });

        assert!(output.hold_active);
    }

    #[cfg(not(feature = "biophys-stn"))]
    #[test]
    fn tool_side_effects_hold_with_policy_pressure() {
        let mut stn = Stn::new();
        let output = stn.tick(&StnInput {
            tool_side_effects_present: true,
            policy_pressure: LevelClass::Med,
            ..base_input()
        });

        assert!(output.hold_active);
        assert!(output.hint_simulate_first);
    }

    #[cfg(feature = "microcircuit-stn-hold")]
    mod microcircuit_invariants {
        use super::*;
        use microcircuit_core::CircuitConfig;

        fn assert_micro_superset(input: StnInput) {
            let mut rules = StnRules::new();
            let mut micro = Stn::new_micro(CircuitConfig::default());

            let rules_output = rules.tick(&input);
            let micro_output = micro.tick(&input);

            assert!(micro_output.hold_active);
            if rules_output.hint_simulate_first {
                assert!(micro_output.hint_simulate_first);
            }
            if rules_output.hint_export_lock {
                assert!(micro_output.hint_export_lock);
            }
        }

        #[test]
        fn receipt_invalid_forces_hold_active() {
            assert_micro_superset(StnInput {
                receipt_invalid_present: true,
                ..base_input()
            });
        }

        #[test]
        fn integrity_block_forces_hold_active() {
            assert_micro_superset(StnInput {
                integrity: IntegrityState::Fail,
                ..base_input()
            });
        }

        #[test]
        fn dlp_critical_forces_hold_active() {
            assert_micro_superset(StnInput {
                dlp_critical_present: true,
                ..base_input()
            });
        }
    }

    #[cfg(feature = "biophys-stn")]
    mod biophys_invariants {
        use super::*;
        use microcircuit_core::CircuitConfig;

        fn assert_micro_superset(input: StnInput) {
            let mut rules = StnRules::new();
            let mut micro = Stn::new_biophys(CircuitConfig::default());

            let rules_output = rules.tick(&input);
            let micro_output = micro.tick(&input);

            assert!(micro_output.hold_active);
            if rules_output.hint_simulate_first {
                assert!(micro_output.hint_simulate_first);
            }
            if rules_output.hint_export_lock {
                assert!(micro_output.hint_export_lock);
            }
        }

        #[test]
        fn receipt_invalid_forces_hold_active() {
            assert_micro_superset(StnInput {
                receipt_invalid_present: true,
                ..base_input()
            });
        }

        #[test]
        fn integrity_block_forces_hold_active() {
            assert_micro_superset(StnInput {
                integrity: IntegrityState::Fail,
                ..base_input()
            });
        }

        #[test]
        fn dlp_critical_forces_hold_active() {
            assert_micro_superset(StnInput {
                dlp_critical_present: true,
                ..base_input()
            });
        }
    }
}
