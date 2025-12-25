#![forbid(unsafe_code)]

use dbm_core::DbmModule;
#[cfg(any(feature = "microcircuit-insula-fusion", feature = "biophys-l4-insula"))]
use microcircuit_core::CircuitConfig;
use microcircuit_core::MicrocircuitBackend;
pub use microcircuit_insula_stub::{InsulaInput, InsulaRules};
use std::fmt;

pub enum InsulaBackend {
    Rules(InsulaRules),
    Micro(Box<dyn MicrocircuitBackend<InsulaInput, dbm_core::IsvSnapshot>>),
}

impl fmt::Debug for InsulaBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InsulaBackend::Rules(_) => f.write_str("InsulaBackend::Rules"),
            InsulaBackend::Micro(_) => f.write_str("InsulaBackend::Micro"),
        }
    }
}

impl InsulaBackend {
    fn tick(&mut self, input: &InsulaInput) -> dbm_core::IsvSnapshot {
        match self {
            InsulaBackend::Rules(rules) => rules.tick(input),
            InsulaBackend::Micro(backend) => backend.step(input, 0),
        }
    }
}

#[derive(Debug)]
pub struct Insula {
    backend: InsulaBackend,
}

impl Insula {
    pub fn new() -> Self {
        #[cfg(feature = "biophys-l4-insula")]
        {
            Self::new_l4(CircuitConfig::default())
        }
        #[cfg(all(
            feature = "microcircuit-insula-fusion",
            not(feature = "biophys-l4-insula")
        ))]
        {
            Self::new_micro(CircuitConfig::default())
        }
        #[cfg(all(
            not(feature = "microcircuit-insula-fusion"),
            not(feature = "biophys-l4-insula")
        ))]
        {
            Self {
                backend: InsulaBackend::Rules(InsulaRules::new()),
            }
        }
    }

    #[cfg(feature = "microcircuit-insula-fusion")]
    pub fn new_micro(config: CircuitConfig) -> Self {
        use microcircuit_insula_fusion::InsulaFusionMicrocircuit;

        Self {
            backend: InsulaBackend::Micro(Box::new(InsulaFusionMicrocircuit::new(config))),
        }
    }

    #[cfg(feature = "biophys-l4-insula")]
    pub fn new_l4(config: CircuitConfig) -> Self {
        use microcircuit_insula_l4::InsulaL4Microcircuit;

        Self {
            backend: InsulaBackend::Micro(Box::new(InsulaL4Microcircuit::new(config))),
        }
    }

    pub fn snapshot_digest(&self) -> Option<[u8; 32]> {
        match &self.backend {
            InsulaBackend::Micro(backend) => Some(backend.snapshot_digest()),
            InsulaBackend::Rules(_) => None,
        }
    }

    pub fn config_digest(&self) -> Option<[u8; 32]> {
        match &self.backend {
            InsulaBackend::Micro(backend) => Some(backend.config_digest()),
            InsulaBackend::Rules(_) => None,
        }
    }
}

impl Default for Insula {
    fn default() -> Self {
        Self::new()
    }
}

impl DbmModule for Insula {
    type Input = InsulaInput;
    type Output = dbm_core::IsvSnapshot;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        self.backend.tick(input)
    }
}
