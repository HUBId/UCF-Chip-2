#![forbid(unsafe_code)]

use dbm_core::DbmModule;
#[cfg(feature = "microcircuit-pmrf-rhythm")]
use microcircuit_core::CircuitConfig;
use microcircuit_core::MicrocircuitBackend;
pub use microcircuit_pmrf_stub::{PmrfInput, PmrfOutput, PmrfRules, SequenceMode};
use std::fmt;

pub enum PmrfBackend {
    Rules(PmrfRules),
    Micro(Box<dyn MicrocircuitBackend<PmrfInput, PmrfOutput>>),
}

impl fmt::Debug for PmrfBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PmrfBackend::Rules(_) => f.write_str("PmrfBackend::Rules"),
            PmrfBackend::Micro(_) => f.write_str("PmrfBackend::Micro"),
        }
    }
}

impl PmrfBackend {
    fn tick(&mut self, input: &PmrfInput) -> PmrfOutput {
        match self {
            PmrfBackend::Rules(rules) => rules.tick(input),
            PmrfBackend::Micro(backend) => backend.step(input, 0),
        }
    }
}

#[derive(Debug)]
pub struct Pmrf {
    backend: PmrfBackend,
}

impl Pmrf {
    pub fn new() -> Self {
        Self {
            backend: PmrfBackend::Rules(PmrfRules::new()),
        }
    }

    #[cfg(feature = "microcircuit-pmrf-rhythm")]
    pub fn new_micro(config: CircuitConfig) -> Self {
        use microcircuit_pmrf_rhythm::PmrfRhythmMicrocircuit;

        Self {
            backend: PmrfBackend::Micro(Box::new(PmrfRhythmMicrocircuit::new(config))),
        }
    }

    pub fn snapshot_digest(&self) -> Option<[u8; 32]> {
        match &self.backend {
            PmrfBackend::Micro(backend) => Some(backend.snapshot_digest()),
            PmrfBackend::Rules(_) => None,
        }
    }

    pub fn config_digest(&self) -> Option<[u8; 32]> {
        match &self.backend {
            PmrfBackend::Micro(backend) => Some(backend.config_digest()),
            PmrfBackend::Rules(_) => None,
        }
    }
}

impl Default for Pmrf {
    fn default() -> Self {
        Self::new()
    }
}

impl DbmModule for Pmrf {
    type Input = PmrfInput;
    type Output = PmrfOutput;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        self.backend.tick(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dbm_core::LevelClass;

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
        let mut pmrf = Pmrf::new();
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
        let mut pmrf = Pmrf::new();
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
        let mut pmrf = Pmrf::new();
        let output = pmrf.tick(&base_input());

        assert_eq!(output.sequence_mode, SequenceMode::Normal);
        assert!(!output.chain_tightening);
        assert!(!output.checkpoint_required);
        assert!(output.reason_codes.codes.is_empty());
    }

    #[cfg(feature = "microcircuit-pmrf-rhythm")]
    mod microcircuit_invariants {
        use super::*;
        use microcircuit_core::CircuitConfig;

        fn assert_micro_not_less_strict(input: PmrfInput) {
            let mut rules = PmrfRules::new();
            let mut micro = Pmrf::new_micro(CircuitConfig::default());

            let rules_output = rules.tick(&input);
            let micro_output = micro.tick(&input);

            match rules_output.sequence_mode {
                SequenceMode::SplitRequired => {
                    assert_eq!(micro_output.sequence_mode, SequenceMode::SplitRequired);
                }
                SequenceMode::Slow => {
                    assert!(micro_output.sequence_mode != SequenceMode::Normal);
                }
                SequenceMode::Normal => {}
            }
        }

        #[test]
        fn divergence_high_forces_split_required() {
            assert_micro_not_less_strict(PmrfInput {
                divergence: LevelClass::High,
                ..base_input()
            });
        }

        #[test]
        fn hold_active_forces_slow_or_split() {
            assert_micro_not_less_strict(PmrfInput {
                hold_active: true,
                ..base_input()
            });
        }

        #[test]
        fn policy_pressure_high_forces_slow_or_split() {
            assert_micro_not_less_strict(PmrfInput {
                policy_pressure: LevelClass::High,
                ..base_input()
            });
        }
    }
}
