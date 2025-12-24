#![forbid(unsafe_code)]

use biophys_core::ModulatorField;
use dbm_core::{IntegrityState, LevelClass, ReasonSet, ThreatVector, ToolKey};

#[derive(Debug, Clone, Default)]
pub struct AmyInput {
    pub integrity: IntegrityState,
    pub replay_mismatch_present: bool,
    pub dlp_secret_present: bool,
    pub dlp_obfuscation_present: bool,
    pub dlp_stegano_present: bool,
    pub dlp_critical_count_med: u32,
    pub receipt_invalid_medium: u32,
    pub policy_pressure: LevelClass,
    pub deny_storm_present: bool,
    pub sealed: Option<bool>,
    pub tool_anomaly_present: bool,
    pub cerebellum_tool_anomaly_present: Option<bool>,
    pub tool_anomalies: Vec<(ToolKey, LevelClass)>,
    pub divergence: LevelClass,
    pub modulators: ModulatorField,
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
pub struct AmyRules {}

impl AmyRules {
    pub fn new() -> Self {
        Self {}
    }

    pub fn tick(&mut self, input: &AmyInput) -> AmyOutput {
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
            || input.cerebellum_tool_anomaly_present.unwrap_or(false)
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
