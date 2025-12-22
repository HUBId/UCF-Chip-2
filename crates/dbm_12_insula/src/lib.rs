#![forbid(unsafe_code)]

use dbm_core::{DbmModule, IntegrityState, IsvSnapshot, LevelClass, ReasonSet};

#[derive(Debug, Clone)]
pub struct InsulaInput {
    pub policy_pressure: LevelClass,
    pub receipt_failures: LevelClass,
    pub receipt_invalid_present: bool,
    pub exec_reliability: LevelClass,
    pub integrity: IntegrityState,
    pub timeout_burst: bool,
    pub cbv_present: bool,
    pub pev_present: bool,
    pub hbv_present: bool,
    pub dominant_reason_codes: Vec<String>,
}

impl Default for InsulaInput {
    fn default() -> Self {
        Self {
            policy_pressure: LevelClass::Low,
            receipt_failures: LevelClass::Low,
            receipt_invalid_present: false,
            exec_reliability: LevelClass::Low,
            integrity: IntegrityState::Ok,
            timeout_burst: false,
            cbv_present: false,
            pev_present: false,
            hbv_present: false,
            dominant_reason_codes: Vec::new(),
        }
    }
}

#[derive(Debug, Default)]
pub struct Insula {}

impl Insula {
    pub fn new() -> Self {
        Self {}
    }
}

impl DbmModule for Insula {
    type Input = InsulaInput;
    type Output = IsvSnapshot;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        let mut snapshot = IsvSnapshot {
            integrity: input.integrity,
            policy_pressure: input.policy_pressure,
            ..IsvSnapshot::default()
        };

        if matches!(input.integrity, IntegrityState::Fail) {
            snapshot.threat = LevelClass::High;
            snapshot.arousal = LevelClass::High;
            snapshot.stability = LevelClass::High;
        }

        if input.receipt_invalid_present {
            snapshot.threat = LevelClass::High;
        }

        if input.policy_pressure == LevelClass::High {
            snapshot.policy_pressure = LevelClass::High;
            snapshot.arousal = LevelClass::Med.max(snapshot.arousal);
        }

        if input.timeout_burst || input.exec_reliability == LevelClass::High {
            snapshot.arousal = LevelClass::High;
        }

        if input.integrity != IntegrityState::Ok {
            snapshot.stability = LevelClass::High;
        }

        let mut reasons = ReasonSet::default();
        reasons.extend(input.dominant_reason_codes.clone());
        snapshot.dominant_reason_codes = reasons;

        snapshot
    }
}

trait LevelClassExt {
    fn max(self, other: Self) -> Self;
}

impl LevelClassExt for LevelClass {
    fn max(self, other: Self) -> Self {
        use LevelClass::*;
        match (self, other) {
            (High, _) | (_, High) => High,
            (Med, _) | (_, Med) => Med,
            _ => Low,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reason_codes() -> Vec<String> {
        vec!["a".to_string(), "b".to_string(), "c".to_string()]
    }

    #[test]
    fn integrity_fail_drives_high_values() {
        let mut module = Insula::new();
        let input = InsulaInput {
            integrity: IntegrityState::Fail,
            receipt_invalid_present: true,
            dominant_reason_codes: reason_codes(),
            ..Default::default()
        };

        let snapshot = module.tick(&input);
        assert_eq!(snapshot.threat, LevelClass::High);
        assert_eq!(snapshot.arousal, LevelClass::High);
        assert_eq!(snapshot.stability, LevelClass::High);
        assert_eq!(snapshot.dominant_reason_codes.codes.len(), 3);
    }

    #[test]
    fn policy_pressure_and_timeouts_raise_arousal() {
        let mut module = Insula::new();
        let input = InsulaInput {
            policy_pressure: LevelClass::High,
            exec_reliability: LevelClass::High,
            timeout_burst: true,
            ..Default::default()
        };

        let snapshot = module.tick(&input);
        assert_eq!(snapshot.policy_pressure, LevelClass::High);
        assert_eq!(snapshot.arousal, LevelClass::High);
    }

    #[test]
    fn integrity_degraded_sets_stability_high() {
        let mut module = Insula::new();
        let input = InsulaInput {
            integrity: IntegrityState::Degraded,
            ..Default::default()
        };

        let snapshot = module.tick(&input);
        assert_eq!(snapshot.stability, LevelClass::High);
    }
}
