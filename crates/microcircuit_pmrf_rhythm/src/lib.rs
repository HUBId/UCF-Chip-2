#![forbid(unsafe_code)]

use dbm_core::{LevelClass, ReasonSet};
use microcircuit_core::{digest_config, digest_meta, CircuitConfig, MicrocircuitBackend};
use microcircuit_pmrf_stub::{PmrfInput, PmrfOutput, SequenceMode};

const TEMPO_MIN: i32 = 0;
const TEMPO_MAX: i32 = 100;
const TEMPO_RECOVERY: i32 = 2;
const TEMPO_SLOW_DIVISOR: i32 = 10;
const SPLIT_GATE_MAX: u8 = 10;
const CHECKPOINT_GATE_MAX: u8 = 10;

#[derive(Debug, Clone)]
struct PmrfRhythmState {
    tempo: i32,
    split_gate: u8,
    checkpoint_gate: u8,
    step_count: u64,
}

impl Default for PmrfRhythmState {
    fn default() -> Self {
        Self {
            tempo: 60,
            split_gate: 0,
            checkpoint_gate: 0,
            step_count: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PmrfRhythmMicrocircuit {
    config: CircuitConfig,
    state: PmrfRhythmState,
}

impl PmrfRhythmMicrocircuit {
    pub fn new(config: CircuitConfig) -> Self {
        Self {
            config,
            state: PmrfRhythmState::default(),
        }
    }

    fn slow_drive(input: &PmrfInput) -> i32 {
        let mut drive = 0;

        match input.divergence {
            LevelClass::High => drive += 70,
            LevelClass::Med => drive += 40,
            LevelClass::Low => {}
        }

        if input.hold_active {
            drive += 40;
        }

        if input.policy_pressure == LevelClass::High {
            drive += 30;
        }

        if input.stability == LevelClass::High {
            drive += 20;
        }

        if input.budget_stress == LevelClass::High {
            drive += 30;
        }

        drive.clamp(0, 100)
    }

    fn split_drive(input: &PmrfInput) -> i32 {
        let mut drive = 0;

        if input.divergence == LevelClass::High {
            drive += 80;
        }

        drive.clamp(0, 100)
    }

    fn update_tempo(&mut self, slow_drive: i32) {
        let delta = slow_drive / TEMPO_SLOW_DIVISOR;
        let updated = self.state.tempo - delta + TEMPO_RECOVERY;
        self.state.tempo = updated.clamp(TEMPO_MIN, TEMPO_MAX);
    }

    fn update_split_gate(&mut self, split_drive: i32) {
        if split_drive >= 80 {
            self.state.split_gate = (self.state.split_gate + 3).min(SPLIT_GATE_MAX);
        } else if self.state.split_gate > 0 {
            self.state.split_gate = self.state.split_gate.saturating_sub(1);
        }
    }

    fn update_checkpoint_gate(&mut self, hold_active: bool) {
        if self.state.split_gate > 0 {
            self.state.checkpoint_gate = CHECKPOINT_GATE_MAX;
        } else if hold_active {
            self.state.checkpoint_gate = (self.state.checkpoint_gate + 1).min(CHECKPOINT_GATE_MAX);
        } else if self.state.checkpoint_gate > 0 {
            self.state.checkpoint_gate = self.state.checkpoint_gate.saturating_sub(1);
        }
    }

    fn sequence_mode(&self, input: &PmrfInput) -> SequenceMode {
        if self.state.split_gate > 0 {
            SequenceMode::SplitRequired
        } else if self.state.tempo < 45
            || input.hold_active
            || input.policy_pressure == LevelClass::High
        {
            SequenceMode::Slow
        } else {
            SequenceMode::Normal
        }
    }
}

impl MicrocircuitBackend<PmrfInput, PmrfOutput> for PmrfRhythmMicrocircuit {
    fn step(&mut self, input: &PmrfInput, _now_ms: u64) -> PmrfOutput {
        self.state.step_count = self.state.step_count.saturating_add(1);

        let slow_drive = Self::slow_drive(input);
        let split_drive = Self::split_drive(input);

        self.update_tempo(slow_drive);
        self.update_split_gate(split_drive);
        self.update_checkpoint_gate(input.hold_active);

        let sequence_mode = self.sequence_mode(input);
        let chain_tightening = sequence_mode != SequenceMode::Normal;
        let checkpoint_required = self.state.split_gate > 0 || self.state.checkpoint_gate >= 5;

        let mut reason_codes = ReasonSet::default();
        match sequence_mode {
            SequenceMode::SplitRequired => {
                reason_codes.insert("RC.GV.SEQUENCE.SPLIT_REQUIRED");
            }
            SequenceMode::Slow => {
                reason_codes.insert("RC.GV.SEQUENCE.SLOW_DOWN");
            }
            SequenceMode::Normal => {}
        }

        PmrfOutput {
            sequence_mode,
            chain_tightening,
            checkpoint_required,
            reason_codes,
        }
    }

    fn snapshot_digest(&self) -> [u8; 32] {
        let mut bytes = Vec::new();
        bytes.extend(self.state.tempo.to_le_bytes());
        bytes.push(self.state.split_gate);
        bytes.push(self.state.checkpoint_gate);
        bytes.extend(self.state.step_count.to_le_bytes());

        digest_meta("UCF:MC:PMRF", &bytes)
    }

    fn config_digest(&self) -> [u8; 32] {
        digest_config("UCF:MC:PMRF:CFG", &self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use biophys_core::ModulatorField;
    use microcircuit_core::CircuitConfig;

    fn base_input() -> PmrfInput {
        PmrfInput {
            divergence: LevelClass::Low,
            policy_pressure: LevelClass::Low,
            stability: LevelClass::Low,
            hold_active: false,
            budget_stress: LevelClass::Low,
            modulators: ModulatorField::default(),
        }
    }

    #[test]
    fn determinism_same_sequence_same_outputs() {
        let mut micro_a = PmrfRhythmMicrocircuit::new(CircuitConfig::default());
        let mut micro_b = PmrfRhythmMicrocircuit::new(CircuitConfig::default());

        let inputs = vec![
            base_input(),
            PmrfInput {
                divergence: LevelClass::Med,
                ..base_input()
            },
            PmrfInput {
                hold_active: true,
                ..base_input()
            },
        ];

        for input in inputs {
            let out_a = micro_a.step(&input, 0);
            let out_b = micro_b.step(&input, 0);
            assert_eq!(out_a, out_b);
        }
    }

    #[test]
    fn divergence_high_triggers_split_required() {
        let mut micro = PmrfRhythmMicrocircuit::new(CircuitConfig::default());
        let output = micro.step(
            &PmrfInput {
                divergence: LevelClass::High,
                ..base_input()
            },
            0,
        );

        assert_eq!(output.sequence_mode, SequenceMode::SplitRequired);
        assert!(output.chain_tightening);
        assert!(output.checkpoint_required);
    }

    #[test]
    fn tempo_recovers_when_drive_low() {
        let mut micro = PmrfRhythmMicrocircuit::new(CircuitConfig::default());
        micro.state.tempo = 40;

        let output = micro.step(&base_input(), 0);

        assert!(micro.state.tempo > 40);
        assert_eq!(output.sequence_mode, SequenceMode::Slow);
    }

    #[test]
    fn hold_active_triggers_slow_mode() {
        let mut micro = PmrfRhythmMicrocircuit::new(CircuitConfig::default());
        let output = micro.step(
            &PmrfInput {
                hold_active: true,
                ..base_input()
            },
            0,
        );

        assert_eq!(output.sequence_mode, SequenceMode::Slow);
        assert!(output.chain_tightening);
    }
}
