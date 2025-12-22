#![forbid(unsafe_code)]

use dbm_core::{
    DbmModule, DwmMode, IntegrityState, IsvSnapshot, LevelClass, ReasonSet, SalienceItem,
    SalienceSource,
};

#[derive(Debug, Clone, Default)]
pub struct SnInput {
    pub isv: IsvSnapshot,
    pub cooldown_class: Option<LevelClass>,
    pub current_dwm: Option<DwmMode>,
    pub replay_hint: bool,
    pub reward_block: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnOutput {
    pub dwm: DwmMode,
    pub salience_items: Vec<SalienceItem>,
    pub reason_codes: ReasonSet,
}

impl Default for SnOutput {
    fn default() -> Self {
        Self {
            dwm: DwmMode::ExecPlan,
            salience_items: Vec::new(),
            reason_codes: ReasonSet::default(),
        }
    }
}

#[derive(Debug, Default)]
pub struct SubstantiaNigra {}

impl SubstantiaNigra {
    pub fn new() -> Self {
        Self {}
    }

    fn suggested_mode(input: &SnInput) -> DwmMode {
        if input.isv.integrity == IntegrityState::Fail {
            return DwmMode::Report;
        }

        if input.isv.threat == LevelClass::High {
            return DwmMode::Stabilize;
        }

        if input.isv.policy_pressure == LevelClass::High || input.isv.arousal == LevelClass::High {
            return DwmMode::Simulate;
        }

        if input.replay_hint {
            return DwmMode::Simulate;
        }

        DwmMode::ExecPlan
    }

    fn apply_hysteresis(current: Option<DwmMode>, suggested: DwmMode) -> DwmMode {
        match current {
            Some(mode) if Self::is_stricter(mode, suggested) => mode,
            _ => suggested,
        }
    }

    fn is_stricter(lhs: DwmMode, rhs: DwmMode) -> bool {
        Self::severity(lhs) > Self::severity(rhs)
    }

    fn severity(mode: DwmMode) -> u8 {
        match mode {
            DwmMode::Report => 3,
            DwmMode::Stabilize => 2,
            DwmMode::Simulate => 1,
            DwmMode::ExecPlan => 0,
        }
    }

    fn build_salience(input: &SnInput) -> Vec<SalienceItem> {
        let mut items = Vec::new();

        if input.isv.integrity == IntegrityState::Fail {
            items.push(SalienceItem::new(
                SalienceSource::Integrity,
                LevelClass::High,
                input.isv.dominant_reason_codes.codes.clone(),
            ));
        }

        if input.isv.threat == LevelClass::High {
            items.push(SalienceItem::new(
                SalienceSource::Threat,
                LevelClass::High,
                input.isv.dominant_reason_codes.codes.clone(),
            ));
        }

        if input.isv.policy_pressure == LevelClass::High {
            items.push(SalienceItem::new(
                SalienceSource::PolicyPressure,
                LevelClass::High,
                input.isv.dominant_reason_codes.codes.clone(),
            ));
        }

        if input.replay_hint {
            let intensity = if input.isv.progress == LevelClass::High {
                LevelClass::High
            } else {
                LevelClass::Med
            };

            items.push(SalienceItem::new(
                SalienceSource::Replay,
                intensity,
                input.isv.dominant_reason_codes.codes.clone(),
            ));
        }

        if input.isv.progress == LevelClass::High && !input.reward_block {
            items.push(SalienceItem::new(
                SalienceSource::Progress,
                LevelClass::Med,
                input.isv.dominant_reason_codes.codes.clone(),
            ));
        }

        if input.reward_block {
            items.push(SalienceItem::new(
                SalienceSource::Integrity,
                LevelClass::Med,
                input.isv.dominant_reason_codes.codes.clone(),
            ));
        }

        let receipt_rcs: Vec<String> = input
            .isv
            .dominant_reason_codes
            .codes
            .iter()
            .filter(|code| code.to_lowercase().contains("insula"))
            .cloned()
            .collect();
        if !receipt_rcs.is_empty() {
            items.push(SalienceItem::new(
                SalienceSource::Receipt,
                LevelClass::Med,
                receipt_rcs,
            ));
        }

        items.sort_by(|a, b| {
            let intensity_ord =
                Self::severity_level(a.intensity).cmp(&Self::severity_level(b.intensity));
            if intensity_ord != std::cmp::Ordering::Equal {
                return intensity_ord.reverse();
            }

            let source_ord = (a.source as u8).cmp(&(b.source as u8));
            if source_ord != std::cmp::Ordering::Equal {
                return source_ord;
            }

            let a_first = a.rcs.first();
            let b_first = b.rcs.first();
            a_first.cmp(&b_first)
        });

        if items.len() > 16 {
            items.truncate(16);
        }

        items
    }

    fn severity_level(level: LevelClass) -> u8 {
        match level {
            LevelClass::High => 2,
            LevelClass::Med => 1,
            LevelClass::Low => 0,
        }
    }
}

impl DbmModule for SubstantiaNigra {
    type Input = SnInput;
    type Output = SnOutput;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        let suggested = Self::suggested_mode(input);
        let dwm = Self::apply_hysteresis(input.current_dwm, suggested);

        let mut reason_codes = ReasonSet::default();
        reason_codes.insert(match dwm {
            DwmMode::Report => "RC.GV.DWM.REPORT",
            DwmMode::Stabilize => "RC.GV.DWM.STABILIZE",
            DwmMode::Simulate => "RC.GV.DWM.SIMULATE",
            DwmMode::ExecPlan => "RC.GV.DWM.EXEC_PLAN",
        });

        let salience_items = Self::build_salience(input);

        SnOutput {
            dwm,
            salience_items,
            reason_codes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_isv() -> IsvSnapshot {
        IsvSnapshot::default()
    }

    #[test]
    fn integrity_fail_forces_report() {
        let mut sn = SubstantiaNigra::new();
        let isv = IsvSnapshot {
            integrity: IntegrityState::Fail,
            ..base_isv()
        };

        let output = sn.tick(&SnInput {
            isv,
            ..Default::default()
        });

        assert_eq!(output.dwm, DwmMode::Report);
    }

    #[test]
    fn threat_high_pushes_stabilize() {
        let mut sn = SubstantiaNigra::new();
        let isv = IsvSnapshot {
            threat: LevelClass::High,
            ..base_isv()
        };

        let output = sn.tick(&SnInput {
            isv,
            ..Default::default()
        });

        assert_eq!(output.dwm, DwmMode::Stabilize);
    }

    #[test]
    fn policy_pressure_high_simulates() {
        let mut sn = SubstantiaNigra::new();
        let isv = IsvSnapshot {
            policy_pressure: LevelClass::High,
            ..base_isv()
        };

        let output = sn.tick(&SnInput {
            isv,
            ..Default::default()
        });

        assert_eq!(output.dwm, DwmMode::Simulate);
    }

    #[test]
    fn calm_state_executes_plan() {
        let mut sn = SubstantiaNigra::new();
        let output = sn.tick(&SnInput {
            isv: base_isv(),
            ..Default::default()
        });

        assert_eq!(output.dwm, DwmMode::ExecPlan);
    }

    #[test]
    fn hysteresis_keeps_stricter_mode() {
        let mut sn = SubstantiaNigra::new();
        let output = sn.tick(&SnInput {
            isv: base_isv(),
            current_dwm: Some(DwmMode::Report),
            ..Default::default()
        });

        assert_eq!(output.dwm, DwmMode::Report);
    }

    #[test]
    fn salience_items_bounded_and_sorted() {
        let mut sn = SubstantiaNigra::new();
        let mut reasons = ReasonSet::default();
        for idx in 0..20 {
            reasons.insert(format!("InsulaReason{:02}", idx));
        }
        let isv = IsvSnapshot {
            integrity: IntegrityState::Fail,
            threat: LevelClass::High,
            policy_pressure: LevelClass::High,
            dominant_reason_codes: reasons,
            ..base_isv()
        };

        let output = sn.tick(&SnInput {
            isv,
            ..Default::default()
        });

        assert!(output.salience_items.len() <= 16);
        assert!(output.salience_items.windows(2).all(|pair| {
            let left = &pair[0];
            let right = &pair[1];
            SubstantiaNigra::severity_level(left.intensity)
                > SubstantiaNigra::severity_level(right.intensity)
                || (SubstantiaNigra::severity_level(left.intensity)
                    == SubstantiaNigra::severity_level(right.intensity)
                    && (left.source as u8) <= (right.source as u8))
        }));

        if let Some(receipt_item) = output
            .salience_items
            .iter()
            .find(|item| item.source == SalienceSource::Receipt)
        {
            assert!(receipt_item.rcs.len() <= SalienceItem::MAX_RCS);
        }
    }

    #[test]
    fn replay_hint_and_progress_generate_salience() {
        let mut sn = SubstantiaNigra::new();
        let isv = IsvSnapshot {
            progress: LevelClass::High,
            ..base_isv()
        };

        let output = sn.tick(&SnInput {
            isv,
            replay_hint: true,
            ..Default::default()
        });

        assert!(output
            .salience_items
            .iter()
            .any(|item| item.source == SalienceSource::Replay));
        assert!(output
            .salience_items
            .iter()
            .any(|item| item.source == SalienceSource::Progress));
    }

    #[test]
    fn deterministic_outputs() {
        let mut sn = SubstantiaNigra::new();
        let isv = IsvSnapshot {
            policy_pressure: LevelClass::High,
            ..base_isv()
        };

        let out_a = sn.tick(&SnInput {
            isv: isv.clone(),
            ..Default::default()
        });
        let out_b = sn.tick(&SnInput {
            isv,
            ..Default::default()
        });

        assert_eq!(out_a, out_b);
    }
}
