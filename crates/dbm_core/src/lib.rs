#![forbid(unsafe_code)]

//! Platzhalter-Typen und Traits für DBM-Komponenten.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LevelClass {
    Low,
    Med,
    High,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntegrityState {
    Ok,
    Degraded,
    Fail,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThreatVector {
    Exfil,
    Probing,
    Integrity,
    Availability,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DwmMode {
    Simulate,
    ExecPlan,
    Stabilize,
    Report,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ReasonCodeSet {
    pub codes: Vec<String>,
}

impl ReasonCodeSet {
    pub fn new(codes: Vec<String>) -> Self {
        // TODO: enforce bounds and sorting once business rules are defined.
        Self { codes }
    }
}

/// Trait-Platzhalter für DBM-Komponenten.
pub trait DbmComponent {
    fn mode(&self) -> DwmMode;
    fn integrity(&self) -> IntegrityState;
}
