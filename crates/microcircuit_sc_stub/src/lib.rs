#![forbid(unsafe_code)]

use dbm_core::{
    DbmModule, DwmMode, IntegrityState, IsvSnapshot, OrientTarget, ReasonSet, SalienceItem,
};

#[derive(Debug, Clone, Default)]
pub struct ScInput {
    pub isv: IsvSnapshot,
    pub salience_items: Vec<SalienceItem>,
    pub unlock_present: bool,
    pub replay_planned_present: bool,
    pub integrity: IntegrityState,
    pub dlp_critical_present: bool,
    pub replay_mismatch_present: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScOutput {
    pub target: OrientTarget,
    pub urgency: dbm_core::UrgencyClass,
    pub recommended_dwm: DwmMode,
    pub reason_codes: ReasonSet,
}

impl Default for ScOutput {
    fn default() -> Self {
        Self {
            target: OrientTarget::Approval,
            urgency: dbm_core::UrgencyClass::Low,
            recommended_dwm: DwmMode::ExecPlan,
            reason_codes: ReasonSet::default(),
        }
    }
}

#[derive(Debug, Default)]
pub struct ScRules {}

impl ScRules {
    pub fn new() -> Self {
        Self {}
    }
}

impl DbmModule for ScRules {
    type Input = ScInput;
    type Output = ScOutput;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        let mut output = ScOutput::default();

        let dlp_critical = input.dlp_critical_present
            || (input.isv.threat == dbm_core::LevelClass::High
                && input
                    .isv
                    .threat_vectors
                    .as_ref()
                    .map(|vectors| vectors.contains(&dbm_core::ThreatVector::Exfil))
                    .unwrap_or(false));

        if input.integrity == IntegrityState::Fail {
            output.target = OrientTarget::Integrity;
            output.urgency = dbm_core::UrgencyClass::High;
            output.recommended_dwm = DwmMode::Report;
        } else if dlp_critical {
            output.target = OrientTarget::Dlp;
            output.urgency = dbm_core::UrgencyClass::High;
            output.recommended_dwm = DwmMode::Stabilize;
        } else if input.unlock_present {
            output.target = OrientTarget::Recovery;
            output.urgency = dbm_core::UrgencyClass::Med;
            output.recommended_dwm = DwmMode::Report;
        } else if input.replay_planned_present {
            output.target = OrientTarget::Replay;
            output.urgency = dbm_core::UrgencyClass::Med;
            output.recommended_dwm = DwmMode::Simulate;
        } else if input.isv.policy_pressure == dbm_core::LevelClass::High {
            output.target = OrientTarget::PolicyPressure;
            output.urgency = dbm_core::UrgencyClass::Med;
            output.recommended_dwm = DwmMode::Simulate;
        }

        output.reason_codes.insert(match output.target {
            OrientTarget::Integrity => "RC.GV.ORIENT.TARGET_INTEGRITY",
            OrientTarget::Dlp => "RC.GV.ORIENT.TARGET_DLP",
            OrientTarget::Recovery => "RC.GV.ORIENT.TARGET_RECOVERY",
            OrientTarget::Approval => "RC.GV.ORIENT.TARGET_APPROVAL",
            OrientTarget::Replay => "RC.GV.ORIENT.TARGET_REPLAY",
            OrientTarget::PolicyPressure => "RC.GV.ORIENT.TARGET_POLICY_PRESSURE",
        });

        output
    }
}
