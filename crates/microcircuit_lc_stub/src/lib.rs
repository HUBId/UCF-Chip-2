#![forbid(unsafe_code)]

use biophys_core::ModulatorField;
use dbm_core::{DbmModule, IntegrityState, LevelClass, ReasonSet};
use microcircuit_core::{
    digest_config, digest_meta, CircuitConfig, CircuitStateMeta, MicrocircuitBackend,
};

#[derive(Debug, Clone, Default)]
pub struct LcInput {
    pub integrity: IntegrityState,
    pub receipt_invalid_count_short: u32,
    pub receipt_missing_count_short: u32,
    pub dlp_critical_present_short: bool,
    pub timeout_count_short: u32,
    pub deny_count_short: u32,
    pub arousal_floor: LevelClass,
    pub modulators: ModulatorField,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LcOutput {
    pub arousal: LevelClass,
    pub hint_simulate_first: bool,
    pub hint_novelty_lock: bool,
    pub reason_codes: ReasonSet,
}

impl Default for LcOutput {
    fn default() -> Self {
        Self {
            arousal: LevelClass::Low,
            hint_simulate_first: false,
            hint_novelty_lock: false,
            reason_codes: ReasonSet::default(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct LcMicrocircuit {
    config: CircuitConfig,
    meta: CircuitStateMeta,
}

impl LcMicrocircuit {
    pub fn new(config: CircuitConfig) -> Self {
        Self {
            config,
            meta: CircuitStateMeta::default(),
        }
    }

    fn severity(level: LevelClass) -> u8 {
        match level {
            LevelClass::Low => 0,
            LevelClass::Med => 1,
            LevelClass::High => 2,
        }
    }
}

impl MicrocircuitBackend<LcInput, LcOutput> for LcMicrocircuit {
    fn step(&mut self, input: &LcInput, now_ms: u64) -> LcOutput {
        self.meta.last_step_ms = now_ms;
        self.meta.step_count = self.meta.step_count.saturating_add(1);

        let mut reason_codes = ReasonSet::default();

        let mut arousal = if input.integrity != IntegrityState::Ok
            || input.receipt_invalid_count_short >= 1
            || input.dlp_critical_present_short
            || input.timeout_count_short >= 2
        {
            reason_codes.insert("lc_high_trigger");
            LevelClass::High
        } else if input.deny_count_short >= 2
            || input.receipt_missing_count_short >= 1
            || input.timeout_count_short == 1
        {
            reason_codes.insert("lc_med_trigger");
            LevelClass::Med
        } else {
            LevelClass::Low
        };

        if Self::severity(arousal) < Self::severity(input.arousal_floor) {
            arousal = input.arousal_floor;
            reason_codes.insert("lc_floor");
        }

        let (hint_simulate_first, hint_novelty_lock) = match arousal {
            LevelClass::High => (true, true),
            LevelClass::Med => (true, false),
            LevelClass::Low => (false, false),
        };

        LcOutput {
            arousal,
            hint_simulate_first,
            hint_novelty_lock,
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

        digest_meta("lc_stub", &bytes)
    }

    fn config_digest(&self) -> [u8; 32] {
        digest_config("lc_stub_config", &self.config)
    }
}

impl DbmModule for LcMicrocircuit {
    type Input = LcInput;
    type Output = LcOutput;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        self.step(input, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_input() -> LcInput {
        LcInput {
            integrity: IntegrityState::Ok,
            arousal_floor: LevelClass::Low,
            ..Default::default()
        }
    }

    #[test]
    fn digest_changes_with_progress() {
        let mut circuit = LcMicrocircuit::new(CircuitConfig::default());
        let digest_a = circuit.snapshot_digest();
        circuit.step(&LcInput::default(), 10);
        let digest_b = circuit.snapshot_digest();
        assert_ne!(digest_a, digest_b);
    }

    #[test]
    fn parity_with_rule_logic() {
        let mut circuit = LcMicrocircuit::new(CircuitConfig::default());
        let output = circuit.step(
            &LcInput {
                receipt_invalid_count_short: 1,
                ..base_input()
            },
            0,
        );

        assert_eq!(output.arousal, LevelClass::High);
        assert!(output.hint_simulate_first);
        assert!(output.hint_novelty_lock);
    }
}
