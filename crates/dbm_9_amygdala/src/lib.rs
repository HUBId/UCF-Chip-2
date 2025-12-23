#![forbid(unsafe_code)]

use dbm_core::{DbmModule, IntegrityState, LevelClass, ReasonSet, ThreatVector, ToolKey};

#[derive(Debug, Clone, Default)]
pub struct AmyInput {
    pub integrity: IntegrityState,
    pub replay_mismatch_present: bool,
    pub dlp_secret_present: bool,
    pub dlp_obfuscation_present: bool,
    pub dlp_stegano_present: bool,
    pub receipt_invalid_medium: u32,
    pub policy_pressure: LevelClass,
    pub sealed: Option<bool>,
    pub tool_anomaly_present: bool,
    pub tool_anomalies: Vec<(ToolKey, LevelClass)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AmyOutput {
    pub threat: LevelClass,
    pub vectors: Vec<ThreatVector>,
    pub reason_codes: ReasonSet,
}

impl Default for AmyOutput {
    fn default() -> Self {
        Self {
            threat: LevelClass::Low,
            vectors: Vec::new(),
            reason_codes: ReasonSet::default(),
        }
    }
}

#[derive(Debug, Default)]
pub struct Amygdala {}

impl Amygdala {
    pub fn new() -> Self {
        Self {}
    }
}

impl DbmModule for Amygdala {
    type Input = AmyInput;
    type Output = AmyOutput;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        let mut output = AmyOutput::default();
        let mut reason_codes = ReasonSet::default();

        let sealed = input
            .sealed
            .unwrap_or(matches!(input.integrity, IntegrityState::Fail));

        let integrity_compromise = sealed
            || matches!(input.integrity, IntegrityState::Fail)
            || input.replay_mismatch_present;
        let exfil_present =
            input.dlp_secret_present || input.dlp_obfuscation_present || input.dlp_stegano_present;
        let probing_present = input.policy_pressure == LevelClass::High;
        let tool_side_effects = input.tool_anomaly_present
            || input
                .tool_anomalies
                .iter()
                .any(|(_, level)| matches!(level, LevelClass::High));

        let mut vectors: Vec<ThreatVector> = Vec::new();
        for vector in [
            ThreatVector::Exfil,
            ThreatVector::Probing,
            ThreatVector::IntegrityCompromise,
            ThreatVector::RuntimeEscape,
            ThreatVector::ToolSideEffects,
        ] {
            let should_add = match vector {
                ThreatVector::Exfil => exfil_present,
                ThreatVector::Probing => probing_present,
                ThreatVector::IntegrityCompromise => integrity_compromise,
                ThreatVector::RuntimeEscape => false,
                ThreatVector::ToolSideEffects => tool_side_effects,
            };

            if should_add {
                vectors.push(vector);
            }
        }

        if integrity_compromise {
            reason_codes.insert("RcThIntegrityCompromise");
        }

        if exfil_present {
            reason_codes.insert("ThExfilHighConfidence");
        }

        if probing_present {
            reason_codes.insert("ThPolicyProbing");
        }
        if tool_side_effects {
            reason_codes.insert("RC.TH.TOOL_SIDE_EFFECTS");
        }

        let threat = if integrity_compromise || exfil_present || input.receipt_invalid_medium >= 1 {
            LevelClass::High
        } else if probing_present {
            LevelClass::Med
        } else if tool_side_effects {
            if input.policy_pressure == LevelClass::High {
                LevelClass::High
            } else {
                LevelClass::Med
            }
        } else {
            LevelClass::Low
        };

        output.threat = threat;
        output.vectors = vectors;
        output.reason_codes = reason_codes;
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_input() -> AmyInput {
        AmyInput {
            integrity: IntegrityState::Ok,
            replay_mismatch_present: false,
            dlp_secret_present: false,
            dlp_obfuscation_present: false,
            dlp_stegano_present: false,
            receipt_invalid_medium: 0,
            policy_pressure: LevelClass::Low,
            sealed: None,
            tool_anomaly_present: false,
            tool_anomalies: Vec::new(),
        }
    }

    #[test]
    fn dlp_secret_triggers_exfil_and_high_threat() {
        let mut module = Amygdala::new();
        let output = module.tick(&AmyInput {
            dlp_secret_present: true,
            ..base_input()
        });

        assert_eq!(output.threat, LevelClass::High);
        assert!(output.vectors.contains(&ThreatVector::Exfil));
    }

    #[test]
    fn replay_mismatch_sets_integrity_compromise_and_high_threat() {
        let mut module = Amygdala::new();
        let output = module.tick(&AmyInput {
            replay_mismatch_present: true,
            ..base_input()
        });

        assert_eq!(output.threat, LevelClass::High);
        assert!(output.vectors.contains(&ThreatVector::IntegrityCompromise));
    }

    #[test]
    fn policy_pressure_high_marks_probing_and_med_threat() {
        let mut module = Amygdala::new();
        let output = module.tick(&AmyInput {
            policy_pressure: LevelClass::High,
            ..base_input()
        });

        assert_eq!(output.threat, LevelClass::Med);
        assert!(output.vectors.contains(&ThreatVector::Probing));
    }

    #[test]
    fn vector_order_is_deterministic() {
        let mut module = Amygdala::new();
        let output = module.tick(&AmyInput {
            replay_mismatch_present: true,
            dlp_secret_present: true,
            policy_pressure: LevelClass::High,
            ..base_input()
        });

        assert_eq!(
            output.vectors,
            vec![
                ThreatVector::Exfil,
                ThreatVector::Probing,
                ThreatVector::IntegrityCompromise,
            ]
        );
    }

    #[test]
    fn tool_side_effects_raise_threat_and_vector() {
        let mut module = Amygdala::new();
        let output = module.tick(&AmyInput {
            tool_anomaly_present: true,
            policy_pressure: LevelClass::Med,
            ..base_input()
        });

        assert_eq!(output.threat, LevelClass::Med);
        assert!(output.vectors.contains(&ThreatVector::ToolSideEffects));
        assert!(output
            .reason_codes
            .codes
            .contains(&"RC.TH.TOOL_SIDE_EFFECTS".to_string()));
    }
}
