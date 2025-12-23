#![forbid(unsafe_code)]

use dbm_core::{CooldownClass, DbmModule, IntegrityState, LevelClass, ReasonSet};
use microcircuit_core::{
    digest_config, digest_meta, CircuitConfig, CircuitStateMeta, MicrocircuitBackend,
};

#[derive(Debug, Clone, Default)]
pub struct SerInput {
    pub integrity: IntegrityState,
    pub replay_mismatch_present: bool,
    pub receipt_invalid_count_medium: u32,
    pub dlp_critical_count_medium: u32,
    pub flapping_count_medium: u32,
    pub unlock_present: bool,
    pub stability_floor: LevelClass,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SerOutput {
    pub stability: LevelClass,
    pub cooldown_class: CooldownClass,
    pub deescalation_lock: bool,
    pub reason_codes: ReasonSet,
}

impl Default for SerOutput {
    fn default() -> Self {
        Self {
            stability: LevelClass::Low,
            cooldown_class: CooldownClass::Base,
            deescalation_lock: false,
            reason_codes: ReasonSet::default(),
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct SerRules {
    stability_current: LevelClass,
    stable_windows: u32,
    last_was_high: bool,
}

impl SerRules {
    pub fn new() -> Self {
        Self::default()
    }

    fn severity(level: LevelClass) -> u8 {
        match level {
            LevelClass::Low => 0,
            LevelClass::Med => 1,
            LevelClass::High => 2,
        }
    }

    pub fn tick(&mut self, input: &SerInput) -> SerOutput {
        let mut reason_codes = ReasonSet::default();
        let mut desired = if input.integrity != IntegrityState::Ok
            || input.replay_mismatch_present
            || input.receipt_invalid_count_medium >= 1
            || input.dlp_critical_count_medium >= 5
            || input.flapping_count_medium >= 6
        {
            reason_codes.insert("ser_high_trigger");
            LevelClass::High
        } else if input.flapping_count_medium >= 2 || input.dlp_critical_count_medium >= 1 {
            reason_codes.insert("ser_med_trigger");
            LevelClass::Med
        } else {
            LevelClass::Low
        };

        if Self::severity(desired) < Self::severity(input.stability_floor) {
            desired = input.stability_floor;
            reason_codes.insert("ser_floor");
        }

        let mut stability = self.stability_current;

        if stability == LevelClass::High && desired != LevelClass::High {
            if self.stable_windows >= 3 {
                stability = LevelClass::Med;
            }
        } else if stability == LevelClass::Med && desired == LevelClass::Low {
            if self.stable_windows >= 5 {
                stability = LevelClass::Low;
            }
        } else {
            stability = desired;
        }

        if desired == LevelClass::Low
            && input.integrity == IntegrityState::Ok
            && input.receipt_invalid_count_medium == 0
            && !input.replay_mismatch_present
        {
            self.stable_windows = self.stable_windows.saturating_add(1);
        } else {
            self.stable_windows = 0;
        }

        self.stability_current = stability.max(desired);
        self.last_was_high = desired == LevelClass::High;

        let cooldown_class =
            if self.stability_current == LevelClass::High || input.flapping_count_medium >= 2 {
                CooldownClass::Longer
            } else {
                CooldownClass::Base
            };

        let mut deescalation_lock = self.stability_current == LevelClass::High;
        if !input.unlock_present && input.integrity == IntegrityState::Fail {
            deescalation_lock = true;
            reason_codes.insert("ser_forensic_latch");
        }

        SerOutput {
            stability: self.stability_current,
            cooldown_class,
            deescalation_lock,
            reason_codes,
        }
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

#[derive(Debug, Clone, Default)]
pub struct SerMicrocircuit {
    config: CircuitConfig,
    meta: CircuitStateMeta,
    rules: SerRules,
}

impl SerMicrocircuit {
    pub fn new(config: CircuitConfig) -> Self {
        Self {
            config,
            meta: CircuitStateMeta::default(),
            rules: SerRules::default(),
        }
    }
}

impl MicrocircuitBackend<SerInput, SerOutput> for SerMicrocircuit {
    fn step(&mut self, input: &SerInput, now_ms: u64) -> SerOutput {
        self.meta.last_step_ms = now_ms;
        self.meta.step_count = self.meta.step_count.saturating_add(1);

        self.rules.tick(input)
    }

    fn snapshot_digest(&self) -> [u8; 32] {
        let mut bytes = Vec::new();
        bytes.extend(self.config.version.to_le_bytes());
        bytes.extend(self.config.seed.to_le_bytes());
        bytes.extend(self.config.max_neurons.to_le_bytes());
        bytes.extend(self.meta.last_step_ms.to_le_bytes());
        bytes.extend(self.meta.step_count.to_le_bytes());
        bytes.push(self.rules.stability_current as u8);
        bytes.extend(self.rules.stable_windows.to_le_bytes());
        bytes.push(self.rules.last_was_high as u8);

        digest_meta("UCF:MC:SER:STUB", &bytes)
    }

    fn config_digest(&self) -> [u8; 32] {
        digest_config("UCF:MC:SER:STUB:CONFIG", &self.config)
    }
}

impl DbmModule for SerMicrocircuit {
    type Input = SerInput;
    type Output = SerOutput;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        self.step(input, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_input() -> SerInput {
        SerInput {
            integrity: IntegrityState::Ok,
            stability_floor: LevelClass::Low,
            ..Default::default()
        }
    }

    #[test]
    fn digest_changes_with_progress() {
        let mut circuit = SerMicrocircuit::new(CircuitConfig::default());
        let digest_a = circuit.snapshot_digest();
        circuit.step(&SerInput::default(), 10);
        let digest_b = circuit.snapshot_digest();
        assert_ne!(digest_a, digest_b);
    }

    #[test]
    fn parity_with_rule_logic() {
        let mut circuit = SerMicrocircuit::new(CircuitConfig::default());
        let output = circuit.step(
            &SerInput {
                replay_mismatch_present: true,
                ..base_input()
            },
            0,
        );

        assert_eq!(output.stability, LevelClass::High);
    }
}
