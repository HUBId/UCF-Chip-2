#![forbid(unsafe_code)]

//! Kern-Typen und Traits für deterministische DBM-Komponenten.

use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LevelClass {
    #[default]
    Low,
    Med,
    High,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CooldownClass {
    #[default]
    Base,
    Longer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UrgencyClass {
    #[default]
    Low,
    Med,
    High,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IntegrityState {
    #[default]
    Ok,
    Degraded,
    Fail,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OrientTarget {
    Integrity,
    Dlp,
    Recovery,
    #[default]
    Approval,
    Replay,
    PolicyPressure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DwmMode {
    Simulate,
    #[default]
    ExecPlan,
    Stabilize,
    Report,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SalienceSource {
    Integrity,
    Threat,
    PolicyPressure,
    Receipt,
    Dlp,
    ExecReliability,
    Recovery,
    Replay,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SalienceItem {
    pub source: SalienceSource,
    pub intensity: LevelClass,
    pub rcs: Vec<String>,
}

impl SalienceItem {
    pub const MAX_RCS: usize = 4;

    pub fn new(source: SalienceSource, intensity: LevelClass, rcs: Vec<String>) -> Self {
        let mut rcs = rcs;
        rcs.sort();
        rcs.dedup();
        if rcs.len() > Self::MAX_RCS {
            rcs.truncate(Self::MAX_RCS);
        }

        Self {
            source,
            intensity,
            rcs,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreatVector {
    Exfil,
    Probing,
    IntegrityCompromise,
    RuntimeEscape,
    ToolSideEffects,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OverlaySet {
    pub simulate_first: bool,
    pub export_lock: bool,
    pub novelty_lock: bool,
}

impl OverlaySet {
    pub fn all_enabled() -> Self {
        Self {
            simulate_first: true,
            export_lock: true,
            novelty_lock: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileState {
    M0,
    M1,
    M2,
    M3,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IsvSnapshot {
    pub arousal: LevelClass,
    pub threat: LevelClass,
    pub stability: LevelClass,
    pub policy_pressure: LevelClass,
    pub progress: LevelClass,
    pub integrity: IntegrityState,
    pub dominant_reason_codes: ReasonSet,
    pub threat_vectors: Option<Vec<ThreatVector>>,
    pub replay_hint: bool,
}

impl Default for IsvSnapshot {
    fn default() -> Self {
        Self {
            arousal: LevelClass::Low,
            threat: LevelClass::Low,
            stability: LevelClass::Low,
            policy_pressure: LevelClass::Low,
            progress: LevelClass::Low,
            integrity: IntegrityState::Ok,
            dominant_reason_codes: ReasonSet::default(),
            threat_vectors: None,
            replay_hint: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RsvSnapshot {
    pub arousal: LevelClass,
    pub threat: LevelClass,
    pub stability: LevelClass,
    pub policy_pressure: LevelClass,
    pub integrity: IntegrityState,
    pub dominant_reason_codes: ReasonSet,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasonSet {
    pub codes: Vec<String>,
    max_len: usize,
}

impl ReasonSet {
    pub const DEFAULT_MAX_LEN: usize = 8;

    pub fn new(max_len: usize) -> Self {
        Self {
            codes: Vec::new(),
            max_len,
        }
    }

    pub fn insert(&mut self, code: impl Into<String>) {
        self.codes.push(code.into());
        self.normalize();
    }

    pub fn extend<I: IntoIterator<Item = String>>(&mut self, iter: I) {
        self.codes.extend(iter);
        self.normalize();
    }

    fn normalize(&mut self) {
        self.codes.sort_by(|a, b| {
            let ord = a.cmp(b);
            if ord == Ordering::Equal {
                Ordering::Equal
            } else {
                ord
            }
        });
        self.codes.dedup();
        if self.codes.len() > self.max_len {
            self.codes.truncate(self.max_len);
        }
    }
}

impl Default for ReasonSet {
    fn default() -> Self {
        Self::new(Self::DEFAULT_MAX_LEN)
    }
}

/// Gemeinsames Trait für deterministische DBM-Module.
pub trait DbmModule {
    type Input;
    type Output;

    fn tick(&mut self, input: &Self::Input) -> Self::Output;
}

/// Abwärtskompatibles Alias für Legacy-Module.
pub trait DbmComponent {
    fn mode(&self) -> DwmMode {
        DwmMode::Simulate
    }

    fn integrity(&self) -> IntegrityState {
        IntegrityState::Ok
    }
}
