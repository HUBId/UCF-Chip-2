#![forbid(unsafe_code)]

use dbm_core::{
    DbmModule, DwmMode, IntegrityState, LevelClass, ReasonSet, SalienceItem, SalienceSource,
};
use microcircuit_core::{digest_meta, CircuitConfig, CircuitStateMeta, MicrocircuitBackend};

#[derive(Debug, Clone, Default)]
pub struct SnInput {
    pub isv: dbm_core::IsvSnapshot,
    pub cooldown_class: Option<dbm_core::LevelClass>,
    pub current_dwm: Option<dbm_core::DwmMode>,
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

#[derive(Debug, Clone, Default)]
pub struct SnMicrocircuit {
    config: CircuitConfig,
    meta: CircuitStateMeta,
}

impl SnMicrocircuit {
    pub fn new(config: CircuitConfig) -> Self {
        Self {
            config,
            meta: CircuitStateMeta::default(),
        }
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

impl MicrocircuitBackend<SnInput, SnOutput> for SnMicrocircuit {
    fn step(&mut self, input: &SnInput, now_ms: u64) -> SnOutput {
        self.meta.last_step_ms = now_ms;
        self.meta.step_count = self.meta.step_count.saturating_add(1);

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

    fn snapshot_digest(&self) -> [u8; 32] {
        let mut bytes = Vec::new();
        bytes.extend(self.config.version.to_le_bytes());
        bytes.extend(self.config.seed.to_le_bytes());
        bytes.extend(self.config.max_neurons.to_le_bytes());
        bytes.extend(self.meta.last_step_ms.to_le_bytes());
        bytes.extend(self.meta.step_count.to_le_bytes());

        digest_meta("sn_stub", &bytes)
    }
}

impl DbmModule for SnMicrocircuit {
    type Input = SnInput;
    type Output = SnOutput;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        self.step(input, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_isv() -> dbm_core::IsvSnapshot {
        dbm_core::IsvSnapshot::default()
    }

    #[test]
    fn deterministic_digest_for_same_state() {
        let circuit = SnMicrocircuit::new(CircuitConfig::default());
        let digest_a = circuit.snapshot_digest();
        let digest_b = circuit.snapshot_digest();
        assert_eq!(digest_a, digest_b);
    }

    #[test]
    fn digest_changes_with_progress() {
        let mut circuit = SnMicrocircuit::new(CircuitConfig::default());
        let digest_a = circuit.snapshot_digest();
        circuit.step(&SnInput::default(), 42);
        let digest_b = circuit.snapshot_digest();
        assert_ne!(digest_a, digest_b);
    }

    #[test]
    fn deterministic_outputs_across_steps() {
        let mut circuit = SnMicrocircuit::new(CircuitConfig::default());
        let input = SnInput {
            isv: dbm_core::IsvSnapshot {
                policy_pressure: LevelClass::High,
                ..base_isv()
            },
            replay_hint: true,
            ..Default::default()
        };

        let first = circuit.step(&input, 100);
        let second = circuit.step(&input, 200);

        assert_eq!(first, second);
    }
}
