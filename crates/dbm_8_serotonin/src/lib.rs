#![forbid(unsafe_code)]

use dbm_core::DbmModule;
#[cfg(feature = "microcircuit-serotonin")]
use microcircuit_core::CircuitConfig;
use microcircuit_core::MicrocircuitBackend;
pub use microcircuit_serotonin_stub::{SerInput, SerOutput, SerRules};
use std::fmt;

pub enum SerBackend {
    Rules(SerRules),
    Micro(Box<dyn MicrocircuitBackend<SerInput, SerOutput>>),
}

impl fmt::Debug for SerBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SerBackend::Rules(_) => f.write_str("SerBackend::Rules"),
            SerBackend::Micro(_) => f.write_str("SerBackend::Micro"),
        }
    }
}

impl SerBackend {
    fn tick(&mut self, input: &SerInput) -> SerOutput {
        match self {
            SerBackend::Rules(rules) => rules.tick(input),
            SerBackend::Micro(backend) => backend.step(input, 0),
        }
    }
}

#[derive(Debug)]
pub struct Serotonin {
    backend: SerBackend,
}

impl Serotonin {
    pub fn new() -> Self {
        Self {
            backend: SerBackend::Rules(SerRules::new()),
        }
    }

    #[cfg(feature = "microcircuit-serotonin-attractor")]
    pub fn new_micro(config: CircuitConfig) -> Self {
        use microcircuit_serotonin_attractor::SerAttractorMicrocircuit;

        Self {
            backend: SerBackend::Micro(Box::new(SerAttractorMicrocircuit::new(config))),
        }
    }

    #[cfg(all(
        feature = "microcircuit-serotonin",
        not(feature = "microcircuit-serotonin-attractor")
    ))]
    pub fn new_micro(config: CircuitConfig) -> Self {
        use microcircuit_serotonin_stub::SerMicrocircuit;

        Self {
            backend: SerBackend::Micro(Box::new(SerMicrocircuit::new(config))),
        }
    }

    pub fn snapshot_digest(&self) -> Option<[u8; 32]> {
        match &self.backend {
            SerBackend::Micro(backend) => Some(backend.snapshot_digest()),
            SerBackend::Rules(_) => None,
        }
    }

    pub fn config_digest(&self) -> Option<[u8; 32]> {
        match &self.backend {
            SerBackend::Micro(backend) => Some(backend.config_digest()),
            SerBackend::Rules(_) => None,
        }
    }
}

impl Default for Serotonin {
    fn default() -> Self {
        Self::new()
    }
}

impl DbmModule for Serotonin {
    type Input = SerInput;
    type Output = SerOutput;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        self.backend.tick(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dbm_core::{CooldownClass, IntegrityState, LevelClass};

    fn base_input() -> SerInput {
        SerInput {
            integrity: IntegrityState::Ok,
            stability_floor: LevelClass::Low,
            ..Default::default()
        }
    }

    #[test]
    fn replay_mismatch_forces_high() {
        let mut ser = Serotonin::new();
        let output = ser.tick(&SerInput {
            replay_mismatch_present: true,
            ..base_input()
        });

        assert_eq!(output.stability, LevelClass::High);
    }

    #[test]
    fn hysteresis_requires_multiple_windows() {
        let mut ser = Serotonin::new();
        let _ = ser.tick(&SerInput {
            replay_mismatch_present: true,
            ..base_input()
        });

        let mut output = SerOutput::default();
        for _ in 0..2 {
            output = ser.tick(&base_input());
            assert_eq!(output.stability, LevelClass::High);
        }

        for _ in 0..3 {
            output = ser.tick(&base_input());
        }

        assert_eq!(output.stability, LevelClass::Med);
    }

    #[test]
    fn cooldown_longer_on_flapping() {
        let mut ser = Serotonin::new();
        let output = ser.tick(&SerInput {
            flapping_count_medium: 3,
            ..base_input()
        });

        assert_eq!(output.stability, LevelClass::Med);
        assert_eq!(output.cooldown_class, CooldownClass::Longer);
    }
}
