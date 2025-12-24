#![forbid(unsafe_code)]

use dbm_core::DbmModule;
#[cfg(feature = "microcircuit-sc-attractor")]
use microcircuit_core::CircuitConfig;
use microcircuit_core::MicrocircuitBackend;
pub use microcircuit_sc_stub::{ScInput, ScOutput, ScRules};
use std::fmt;

pub enum ScBackend {
    Rules(ScRules),
    Micro(Box<dyn MicrocircuitBackend<ScInput, ScOutput>>),
}

impl fmt::Debug for ScBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScBackend::Rules(_) => f.write_str("ScBackend::Rules"),
            ScBackend::Micro(_) => f.write_str("ScBackend::Micro"),
        }
    }
}

impl ScBackend {
    fn tick(&mut self, input: &ScInput) -> ScOutput {
        match self {
            ScBackend::Rules(rules) => rules.tick(input),
            ScBackend::Micro(backend) => backend.step(input, 0),
        }
    }
}

#[derive(Debug)]
pub struct Sc {
    backend: ScBackend,
}

impl Sc {
    pub fn new() -> Self {
        Self {
            backend: ScBackend::Rules(ScRules::new()),
        }
    }

    #[cfg(feature = "microcircuit-sc-attractor")]
    pub fn new_micro(config: CircuitConfig) -> Self {
        use microcircuit_sc_attractor::ScAttractorMicrocircuit;

        Self {
            backend: ScBackend::Micro(Box::new(ScAttractorMicrocircuit::new(config))),
        }
    }

    pub fn snapshot_digest(&self) -> Option<[u8; 32]> {
        match &self.backend {
            ScBackend::Micro(backend) => Some(backend.snapshot_digest()),
            ScBackend::Rules(_) => None,
        }
    }

    pub fn config_digest(&self) -> Option<[u8; 32]> {
        match &self.backend {
            ScBackend::Micro(backend) => Some(backend.config_digest()),
            ScBackend::Rules(_) => None,
        }
    }
}

impl Default for Sc {
    fn default() -> Self {
        Self::new()
    }
}

impl DbmModule for Sc {
    type Input = ScInput;
    type Output = ScOutput;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        self.backend.tick(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dbm_core::{
        IntegrityState, IsvSnapshot, LevelClass, SalienceItem, SalienceSource, ThreatVector,
    };

    fn base_input() -> ScInput {
        ScInput {
            salience_items: vec![SalienceItem::new(
                SalienceSource::Threat,
                LevelClass::Low,
                vec![],
            )],
            ..Default::default()
        }
    }

    #[test]
    fn integrity_fail_forces_report() {
        let mut sc = Sc::new();
        let input = ScInput {
            integrity: IntegrityState::Fail,
            ..base_input()
        };

        let output = sc.tick(&input);

        assert_eq!(output.target, dbm_core::OrientTarget::Integrity);
        assert_eq!(output.recommended_dwm, dbm_core::DwmMode::Report);
        assert!(output
            .reason_codes
            .codes
            .contains(&"RC.GV.ORIENT.TARGET_INTEGRITY".to_string()));
    }

    #[test]
    fn exfil_threat_triggers_dlp_target() {
        let mut sc = Sc::new();
        let input = ScInput {
            isv: IsvSnapshot {
                threat: LevelClass::High,
                threat_vectors: Some(vec![ThreatVector::Exfil]),
                ..Default::default()
            },
            ..base_input()
        };

        let output = sc.tick(&input);

        assert_eq!(output.target, dbm_core::OrientTarget::Dlp);
        assert_eq!(output.recommended_dwm, dbm_core::DwmMode::Stabilize);
    }

    #[test]
    fn unlock_takes_precedence_over_replay() {
        let mut sc = Sc::new();
        let input = ScInput {
            unlock_present: true,
            replay_planned_present: true,
            ..base_input()
        };

        let output = sc.tick(&input);

        assert_eq!(output.target, dbm_core::OrientTarget::Recovery);
        assert_eq!(output.urgency, dbm_core::UrgencyClass::Med);
        assert_eq!(output.recommended_dwm, dbm_core::DwmMode::Report);
    }
}
