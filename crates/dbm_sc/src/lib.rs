#![forbid(unsafe_code)]

use dbm_core::{
    DbmModule, DwmMode, IntegrityState, IsvSnapshot, OrientTarget, ReasonSet, SalienceItem,
    ThreatVector, UrgencyClass,
};

#[derive(Debug, Clone, Default)]
pub struct ScInput {
    pub isv: IsvSnapshot,
    pub salience_items: Vec<SalienceItem>,
    pub unlock_present: bool,
    pub replay_planned_present: bool,
    pub integrity: IntegrityState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScOutput {
    pub target: OrientTarget,
    pub urgency: UrgencyClass,
    pub recommended_dwm: DwmMode,
    pub reason_codes: ReasonSet,
}

impl Default for ScOutput {
    fn default() -> Self {
        Self {
            target: OrientTarget::Approval,
            urgency: UrgencyClass::Low,
            recommended_dwm: DwmMode::ExecPlan,
            reason_codes: ReasonSet::default(),
        }
    }
}

#[derive(Debug, Default)]
pub struct Sc {}

impl Sc {
    pub fn new() -> Self {
        Self {}
    }

    fn dlp_critical(isv: &IsvSnapshot) -> bool {
        isv.threat == dbm_core::LevelClass::High
            && isv
                .threat_vectors
                .as_ref()
                .map(|vectors| vectors.contains(&ThreatVector::Exfil))
                .unwrap_or(false)
    }

    fn orient_reason(target: OrientTarget) -> &'static str {
        match target {
            OrientTarget::Integrity => "RC.GV.ORIENT.TARGET_INTEGRITY",
            OrientTarget::Dlp => "RC.GV.ORIENT.TARGET_DLP",
            OrientTarget::Recovery => "RC.GV.ORIENT.TARGET_RECOVERY",
            OrientTarget::Approval => "RC.GV.ORIENT.TARGET_APPROVAL",
            OrientTarget::Replay => "RC.GV.ORIENT.TARGET_REPLAY",
            OrientTarget::PolicyPressure => "RC.GV.ORIENT.TARGET_POLICY_PRESSURE",
        }
    }
}

impl DbmModule for Sc {
    type Input = ScInput;
    type Output = ScOutput;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        let mut output = ScOutput::default();

        if input.integrity == IntegrityState::Fail {
            output.target = OrientTarget::Integrity;
            output.urgency = UrgencyClass::High;
            output.recommended_dwm = DwmMode::Report;
        } else if Self::dlp_critical(&input.isv) {
            output.target = OrientTarget::Dlp;
            output.urgency = UrgencyClass::High;
            output.recommended_dwm = DwmMode::Stabilize;
        } else if input.unlock_present {
            output.target = OrientTarget::Recovery;
            output.urgency = UrgencyClass::Med;
            output.recommended_dwm = DwmMode::Report;
        } else if input.replay_planned_present {
            output.target = OrientTarget::Replay;
            output.urgency = UrgencyClass::Med;
            output.recommended_dwm = DwmMode::Simulate;
        } else if input.isv.policy_pressure == dbm_core::LevelClass::High {
            output.target = OrientTarget::PolicyPressure;
            output.urgency = UrgencyClass::Med;
            output.recommended_dwm = DwmMode::Simulate;
        }

        output
            .reason_codes
            .insert(Self::orient_reason(output.target));

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dbm_core::{LevelClass, SalienceSource};

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

        assert_eq!(output.target, OrientTarget::Integrity);
        assert_eq!(output.recommended_dwm, DwmMode::Report);
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

        assert_eq!(output.target, OrientTarget::Dlp);
        assert_eq!(output.recommended_dwm, DwmMode::Stabilize);
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

        assert_eq!(output.target, OrientTarget::Recovery);
        assert_eq!(output.urgency, UrgencyClass::Med);
        assert_eq!(output.recommended_dwm, DwmMode::Report);
    }
}
