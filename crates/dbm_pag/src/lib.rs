#![forbid(unsafe_code)]

use dbm_core::DbmModule;
#[cfg(feature = "microcircuit-pag-attractor")]
use microcircuit_core::CircuitConfig;
use microcircuit_core::MicrocircuitBackend;
pub use microcircuit_pag_stub::{DefensePattern, PagInput, PagOutput, PagRules};
use std::fmt;

pub enum PagBackend {
    Rules(PagRules),
    Micro(Box<dyn MicrocircuitBackend<PagInput, PagOutput>>),
}

impl fmt::Debug for PagBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PagBackend::Rules(_) => f.write_str("PagBackend::Rules"),
            PagBackend::Micro(_) => f.write_str("PagBackend::Micro"),
        }
    }
}

impl PagBackend {
    fn tick(&mut self, input: &PagInput) -> PagOutput {
        match self {
            PagBackend::Rules(rules) => rules.tick(input),
            PagBackend::Micro(backend) => backend.step(input, 0),
        }
    }
}

#[derive(Debug)]
pub struct Pag {
    backend: PagBackend,
}

impl Pag {
    pub fn new() -> Self {
        Self {
            backend: PagBackend::Rules(PagRules::new()),
        }
    }

    #[cfg(feature = "microcircuit-pag-attractor")]
    pub fn new_micro(config: CircuitConfig) -> Self {
        #[cfg(feature = "biophys-pag")]
        {
            use microcircuit_pag_biophys::PagBiophysMicrocircuit;

            return Self {
                backend: PagBackend::Micro(Box::new(PagBiophysMicrocircuit::new(config))),
            };
        }

        #[cfg(not(feature = "biophys-pag"))]
        {
            use microcircuit_pag_attractor::PagAttractorMicrocircuit;

            Self {
                backend: PagBackend::Micro(Box::new(PagAttractorMicrocircuit::new(config))),
            }
        }
    }

    pub fn snapshot_digest(&self) -> Option<[u8; 32]> {
        match &self.backend {
            PagBackend::Micro(backend) => Some(backend.snapshot_digest()),
            PagBackend::Rules(_) => None,
        }
    }

    pub fn config_digest(&self) -> Option<[u8; 32]> {
        match &self.backend {
            PagBackend::Micro(backend) => Some(backend.config_digest()),
            PagBackend::Rules(_) => None,
        }
    }
}

impl Default for Pag {
    fn default() -> Self {
        Self::new()
    }
}

impl DbmModule for Pag {
    type Input = PagInput;
    type Output = PagOutput;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        self.backend.tick(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use biophys_core::ModulatorField;
    use dbm_core::{CooldownClass, IntegrityState, LevelClass, ThreatVector};

    fn base_input() -> PagInput {
        PagInput {
            integrity: IntegrityState::Ok,
            threat: LevelClass::Low,
            vectors: Vec::new(),
            unlock_present: false,
            stability: LevelClass::Low,
            serotonin_cooldown: CooldownClass::Base,
            modulators: ModulatorField::default(),
        }
    }

    #[test]
    fn integrity_fail_drives_forensic() {
        let mut module = Pag::new();
        let output = module.tick(&PagInput {
            integrity: IntegrityState::Fail,
            ..base_input()
        });

        assert_eq!(output.pattern, DefensePattern::DP3_FORENSIC);
        assert!(output.pattern_latched);
    }

    #[test]
    fn exfil_high_triggers_forensic() {
        let mut module = Pag::new();
        let output = module.tick(&PagInput {
            threat: LevelClass::High,
            vectors: vec![ThreatVector::Exfil],
            ..base_input()
        });

        assert_eq!(output.pattern, DefensePattern::DP3_FORENSIC);
        assert!(output.pattern_latched);
    }

    #[test]
    fn high_threat_without_integrity_sets_quarantine() {
        let mut module = Pag::new();
        let output = module.tick(&PagInput {
            threat: LevelClass::High,
            ..base_input()
        });

        assert_eq!(output.pattern, DefensePattern::DP2_QUARANTINE);
        assert!(output.pattern_latched);
    }

    #[test]
    fn probing_threat_results_in_contained_continue() {
        let mut module = Pag::new();
        let output = module.tick(&PagInput {
            threat: LevelClass::Med,
            ..base_input()
        });

        assert_eq!(output.pattern, DefensePattern::DP4_CONTAINED_CONTINUE);
        assert!(!output.pattern_latched);
    }
}
