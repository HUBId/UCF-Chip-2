#![forbid(unsafe_code)]

use biophys_core::ModulatorField;
use dbm_core::{CooldownClass, DbmModule, IntegrityState, LevelClass, ReasonSet, ThreatVector};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum DefensePattern {
    DP1_FREEZE,
    DP2_QUARANTINE,
    DP3_FORENSIC,
    DP4_CONTAINED_CONTINUE,
}

#[derive(Debug, Clone, Default)]
pub struct PagInput {
    pub integrity: IntegrityState,
    pub threat: LevelClass,
    pub vectors: Vec<ThreatVector>,
    pub unlock_present: bool,
    pub stability: LevelClass,
    pub serotonin_cooldown: CooldownClass,
    pub modulators: ModulatorField,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagOutput {
    pub pattern: DefensePattern,
    pub pattern_latched: bool,
    pub reason_codes: ReasonSet,
}

impl Default for PagOutput {
    fn default() -> Self {
        Self {
            pattern: DefensePattern::DP4_CONTAINED_CONTINUE,
            pattern_latched: false,
            reason_codes: ReasonSet::default(),
        }
    }
}

#[derive(Debug, Default)]
pub struct PagRules {}

impl PagRules {
    pub fn new() -> Self {
        Self {}
    }
}

impl DbmModule for PagRules {
    type Input = PagInput;
    type Output = PagOutput;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        let mut output = PagOutput::default();

        let integrity_fail = matches!(input.integrity, IntegrityState::Fail);
        let exfil_high =
            input.vectors.contains(&ThreatVector::Exfil) && input.threat == LevelClass::High;

        if integrity_fail {
            output.pattern = DefensePattern::DP3_FORENSIC;
            output.pattern_latched = true;
            output
                .reason_codes
                .insert("RC.RX.ACTION.FORENSIC".to_string());
            output.reason_codes.insert("RcRxActionForensic".to_string());
            return output;
        }

        if exfil_high {
            output.pattern = DefensePattern::DP3_FORENSIC;
            output.pattern_latched = true;
            output
                .reason_codes
                .insert("RC.RX.ACTION.FORENSIC".to_string());
            output.reason_codes.insert("RcRxActionForensic".to_string());
            return output;
        }

        if input.threat == LevelClass::High {
            output.pattern = DefensePattern::DP2_QUARANTINE;
            output.pattern_latched = true;
            output
                .reason_codes
                .insert("RC.RX.ACTION.QUARANTINE".to_string());
            return output;
        }

        if input.threat == LevelClass::Med {
            output.pattern = DefensePattern::DP4_CONTAINED_CONTINUE;
            return output;
        }

        output.pattern = DefensePattern::DP4_CONTAINED_CONTINUE;
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use biophys_core::ModulatorField;

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
        let mut module = PagRules::new();
        let output = module.tick(&PagInput {
            integrity: IntegrityState::Fail,
            ..base_input()
        });

        assert_eq!(output.pattern, DefensePattern::DP3_FORENSIC);
        assert!(output.pattern_latched);
    }

    #[test]
    fn exfil_high_triggers_forensic() {
        let mut module = PagRules::new();
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
        let mut module = PagRules::new();
        let output = module.tick(&PagInput {
            threat: LevelClass::High,
            ..base_input()
        });

        assert_eq!(output.pattern, DefensePattern::DP2_QUARANTINE);
        assert!(output.pattern_latched);
    }

    #[test]
    fn probing_threat_results_in_contained_continue() {
        let mut module = PagRules::new();
        let output = module.tick(&PagInput {
            threat: LevelClass::Med,
            ..base_input()
        });

        assert_eq!(output.pattern, DefensePattern::DP4_CONTAINED_CONTINUE);
        assert!(!output.pattern_latched);
    }
}
