#![forbid(unsafe_code)]

//! Kern-Typen und Traits für deterministische DBM-Komponenten.

pub mod limits;

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
pub enum NoiseClass {
    #[default]
    Low,
    Med,
    High,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PriorityClass {
    #[default]
    Low,
    Med,
    High,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RecursionDepthClass {
    #[default]
    Low,
    Med,
    High,
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
    Progress,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceKind {
    LcMicroSnapshot,
    SnMicroSnapshot,
    RulesetDigest,
    CbvDigest,
    PevDigest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvidenceRef {
    pub kind: EvidenceKind,
    pub digest: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ToolKey {
    pub tool_id: String,
    pub action_id: String,
}

impl ToolKey {
    pub const MAX_LEN: usize = 128;

    pub fn new(tool_id: impl Into<String>, action_id: impl Into<String>) -> Self {
        let mut key = Self {
            tool_id: tool_id.into(),
            action_id: action_id.into(),
        };
        key.normalize();
        key
    }

    pub fn normalized(mut self) -> Self {
        self.normalize();
        self
    }

    fn normalize(&mut self) {
        if self.tool_id.len() > Self::MAX_LEN {
            self.tool_id.truncate(Self::MAX_LEN);
        }
        if self.action_id.len() > Self::MAX_LEN {
            self.action_id.truncate(Self::MAX_LEN);
        }
    }
}

impl From<(String, String)> for ToolKey {
    fn from(value: (String, String)) -> Self {
        Self::new(value.0, value.1)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SuspendRecommendation {
    pub tool: ToolKey,
    pub severity: LevelClass,
    pub reason_codes: ReasonSet,
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

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BaselineVector {
    pub caution_floor: LevelClass,
    pub novelty_dampening: LevelClass,
    pub approval_strictness: LevelClass,
    pub export_strictness: LevelClass,
    pub chain_conservatism: LevelClass,
    pub cooldown_bias: CooldownClass,
    pub reward_block_bias: LevelClass,
    pub reason_codes: ReasonSet,
}

impl BaselineVector {
    pub fn add_reason_code(&mut self, code: impl Into<String>) {
        self.reason_codes.insert(code);
    }

    pub fn extend_reason_codes<I: IntoIterator<Item = String>>(&mut self, iter: I) {
        self.reason_codes.extend(iter);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmotionField {
    pub noise: NoiseClass,
    pub priority: PriorityClass,
    pub recursion_depth: RecursionDepthClass,
    pub dwm: DwmMode,
    pub profile: ProfileState,
    pub overlays: OverlaySet,
    pub reason_codes: ReasonSet,
}

impl Default for EmotionField {
    fn default() -> Self {
        Self {
            noise: NoiseClass::Low,
            priority: PriorityClass::Low,
            recursion_depth: RecursionDepthClass::High,
            dwm: DwmMode::ExecPlan,
            profile: ProfileState::M0,
            overlays: OverlaySet::default(),
            reason_codes: ReasonSet::default(),
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_vector_normalizes_reason_codes() {
        let mut baseline = BaselineVector::default();

        baseline.add_reason_code("b");
        baseline.add_reason_code("a");
        baseline.add_reason_code("a");
        baseline.extend_reason_codes(
            ["c", "d", "e", "f", "g", "h", "i", "j"]
                .into_iter()
                .map(String::from),
        );

        assert_eq!(
            baseline.reason_codes.codes.len(),
            ReasonSet::DEFAULT_MAX_LEN
        );
        assert_eq!(
            baseline.reason_codes.codes,
            vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
                "e".to_string(),
                "f".to_string(),
                "g".to_string(),
                "h".to_string(),
            ]
        );
    }

    #[test]
    fn tool_key_truncates() {
        let long = "a".repeat(ToolKey::MAX_LEN + 10);
        let key = ToolKey::new(long.clone(), long.clone());

        assert_eq!(key.tool_id.len(), ToolKey::MAX_LEN);
        assert_eq!(key.action_id.len(), ToolKey::MAX_LEN);
    }
}
