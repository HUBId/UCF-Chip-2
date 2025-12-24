#![forbid(unsafe_code)]

use dbm_core::{IntegrityState, LevelClass, ReasonSet};
use microcircuit_core::{digest_config, digest_meta, CircuitConfig, MicrocircuitBackend};
use microcircuit_stn_stub::{StnInput, StnOutput};

const INHIBIT_MIN: i32 = 0;
const INHIBIT_MAX: i32 = 100;
const LATCH_MAX: u8 = 10;

#[derive(Debug, Clone, Default)]
struct StnHoldState {
    inhibit: i32,
    hold_latch_steps: u8,
    step_count: u64,
}

#[derive(Debug, Clone)]
pub struct StnHoldMicrocircuit {
    config: CircuitConfig,
    state: StnHoldState,
}

impl StnHoldMicrocircuit {
    pub fn new(config: CircuitConfig) -> Self {
        Self {
            config,
            state: StnHoldState::default(),
        }
    }

    fn encode_drive(input: &StnInput) -> i32 {
        let mut drive = 0;

        if input.integrity != IntegrityState::Ok {
            drive += 60;
        }
        if input.receipt_invalid_present {
            drive += 50;
        }
        if input.dlp_critical_present {
            drive += 50;
        }
        if input.policy_pressure == LevelClass::High {
            drive += 30;
        }
        if input.arousal == LevelClass::High {
            drive += 20;
        }
        match input.threat {
            LevelClass::High => drive += 30,
            LevelClass::Med => drive += 10,
            LevelClass::Low => {}
        }

        drive.clamp(0, 100)
    }

    fn push_integrity_reason(reason_codes: &mut ReasonSet, integrity: IntegrityState) {
        match integrity {
            IntegrityState::Degraded => {
                reason_codes.insert("RC.RE.INTEGRITY.DEGRADED");
            }
            IntegrityState::Fail => {
                reason_codes.insert("RC.RE.INTEGRITY.FAIL");
            }
            IntegrityState::Ok => {}
        }
    }
}

impl MicrocircuitBackend<StnInput, StnOutput> for StnHoldMicrocircuit {
    fn step(&mut self, input: &StnInput, _now_ms: u64) -> StnOutput {
        self.state.step_count = self.state.step_count.saturating_add(1);

        let drive = Self::encode_drive(input);
        let delta = (drive - 30) / 3;
        self.state.inhibit = (self.state.inhibit + delta).clamp(INHIBIT_MIN, INHIBIT_MAX);

        if drive >= 50 {
            self.state.hold_latch_steps = (self.state.hold_latch_steps + 2).min(LATCH_MAX);
        } else if self.state.hold_latch_steps > 0 {
            self.state.hold_latch_steps -= 1;
        }

        if drive < 20 {
            self.state.inhibit = (self.state.inhibit - 5).max(INHIBIT_MIN);
        }

        let hold_active = self.state.inhibit >= 50 || self.state.hold_latch_steps > 0;
        let hint_simulate_first = hold_active;
        let hint_novelty_lock = self.state.inhibit >= 70
            || input.policy_pressure == LevelClass::High
            || input.arousal == LevelClass::High;
        let hint_export_lock = input.dlp_critical_present;

        let mut hold_reason_codes = ReasonSet::default();
        if hold_active {
            hold_reason_codes.insert("RC.GV.HOLD.ON");
        }
        if input.integrity != IntegrityState::Ok {
            Self::push_integrity_reason(&mut hold_reason_codes, input.integrity);
        }
        if input.receipt_invalid_present {
            hold_reason_codes.insert("RC.GE.EXEC.DISPATCH_BLOCKED");
        }
        if input.dlp_critical_present {
            hold_reason_codes.insert("RC.CD.DLP.SECRET_PATTERN");
        }

        StnOutput {
            hold_active,
            hold_reason_codes,
            hint_simulate_first,
            hint_novelty_lock,
            hint_export_lock,
        }
    }

    fn snapshot_digest(&self) -> [u8; 32] {
        let mut bytes = Vec::new();
        bytes.extend(self.state.inhibit.to_le_bytes());
        bytes.push(self.state.hold_latch_steps);
        bytes.extend(self.state.step_count.to_le_bytes());

        digest_meta("UCF:MC:STN", &bytes)
    }

    fn config_digest(&self) -> [u8; 32] {
        digest_config("UCF:MC:STN:CFG", &self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use biophys_core::ModulatorField;

    fn base_input() -> StnInput {
        StnInput {
            policy_pressure: LevelClass::Low,
            arousal: LevelClass::Low,
            threat: LevelClass::Low,
            receipt_invalid_present: false,
            dlp_critical_present: false,
            integrity: IntegrityState::Ok,
            tool_side_effects_present: false,
            cerebellum_divergence: LevelClass::Low,
            modulators: ModulatorField::default(),
        }
    }

    #[test]
    fn determinism_same_sequence_same_outputs() {
        let mut left = StnHoldMicrocircuit::new(CircuitConfig::default());
        let mut right = StnHoldMicrocircuit::new(CircuitConfig::default());
        let sequence = [
            base_input(),
            StnInput {
                receipt_invalid_present: true,
                ..base_input()
            },
            StnInput {
                threat: LevelClass::High,
                ..base_input()
            },
            base_input(),
        ];

        let outputs_left: Vec<StnOutput> =
            sequence.iter().map(|input| left.step(input, 0)).collect();
        let outputs_right: Vec<StnOutput> =
            sequence.iter().map(|input| right.step(input, 0)).collect();

        assert_eq!(outputs_left, outputs_right);
    }

    #[test]
    fn receipt_invalid_triggers_hold() {
        let mut micro = StnHoldMicrocircuit::new(CircuitConfig::default());
        let output = micro.step(
            &StnInput {
                receipt_invalid_present: true,
                ..base_input()
            },
            0,
        );

        assert!(output.hold_active);
        assert!(output.hint_simulate_first);
        assert!(output
            .hold_reason_codes
            .codes
            .contains(&"RC.GE.EXEC.DISPATCH_BLOCKED".to_string()));
    }

    #[test]
    fn latch_holds_for_followup_step() {
        let mut micro = StnHoldMicrocircuit::new(CircuitConfig::default());
        let _ = micro.step(
            &StnInput {
                receipt_invalid_present: true,
                ..base_input()
            },
            0,
        );

        let follow = micro.step(&base_input(), 0);
        assert!(follow.hold_active);

        let after = micro.step(&base_input(), 0);
        assert!(!after.hold_active);
    }

    #[test]
    fn novelty_lock_triggers_on_high_inhibit() {
        let mut micro = StnHoldMicrocircuit::new(CircuitConfig::default());
        let high_drive = StnInput {
            integrity: IntegrityState::Fail,
            dlp_critical_present: true,
            receipt_invalid_present: true,
            ..base_input()
        };

        for _ in 0..3 {
            micro.step(&high_drive, 0);
        }
        let output = micro.step(&high_drive, 0);

        assert!(output.hint_novelty_lock);
    }
}
