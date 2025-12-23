#![forbid(unsafe_code)]

use dbm_core::{CooldownClass, IntegrityState, LevelClass, ReasonSet};
use microcircuit_core::{digest_config, digest_meta, CircuitConfig, MicrocircuitBackend};
use microcircuit_serotonin_stub::{SerInput, SerOutput};

#[derive(Debug, Clone, Default)]
struct SerAttractorState {
    tone: i32,
    inertia: u8,
    lock_steps: u8,
    step_count: u64,
}

#[derive(Debug, Clone)]
pub struct SerAttractorMicrocircuit {
    config: CircuitConfig,
    state: SerAttractorState,
}

impl SerAttractorMicrocircuit {
    pub fn new(config: CircuitConfig) -> Self {
        Self {
            config,
            state: SerAttractorState::default(),
        }
    }

    fn encode_drive(input: &SerInput) -> u8 {
        let mut drive: i32 = 0;

        if input.integrity != IntegrityState::Ok {
            drive += 35;
        }
        if input.replay_mismatch_present {
            drive += 30;
        }
        if input.receipt_invalid_count_medium >= 1 {
            drive += 25;
        }
        if input.dlp_critical_count_medium >= 5 {
            drive += 20;
        }
        if input.flapping_count_medium >= 6 {
            drive += 20;
        }
        if input.flapping_count_medium >= 2 {
            drive += 10;
        }
        if input.integrity == IntegrityState::Ok && input.unlock_present {
            drive -= 10;
        }

        match input.stability_floor {
            LevelClass::Med => drive += 10,
            LevelClass::High => drive += 20,
            LevelClass::Low => {}
        }

        drive.clamp(0, 100) as u8
    }

    fn stability_for_state(tone: i32, lock_steps: u8) -> LevelClass {
        if tone >= 70 || lock_steps >= 10 {
            LevelClass::High
        } else if tone >= 45 {
            LevelClass::Med
        } else {
            LevelClass::Low
        }
    }
}

impl MicrocircuitBackend<SerInput, SerOutput> for SerAttractorMicrocircuit {
    fn step(&mut self, input: &SerInput, _now_ms: u64) -> SerOutput {
        self.state.step_count = self.state.step_count.saturating_add(1);

        let drive = Self::encode_drive(input);
        let delta = (drive as i32 - 50) / 5;
        self.state.tone = (self.state.tone + delta).clamp(0, 100);

        if drive >= 70 {
            self.state.lock_steps = (self.state.lock_steps + 2).min(20);
        } else if drive <= 40 && self.state.lock_steps > 0 {
            self.state.lock_steps -= 1;
        }

        if drive >= 70 {
            self.state.inertia = (self.state.inertia + 1).min(10);
        } else if drive <= 40 {
            self.state.inertia = self.state.inertia.saturating_sub(1);
        }

        let mut stability = Self::stability_for_state(self.state.tone, self.state.lock_steps);
        let critical = input.integrity != IntegrityState::Ok
            || input.replay_mismatch_present
            || input.receipt_invalid_count_medium >= 1;
        if critical {
            stability = LevelClass::High;
        }

        let cooldown_class = if stability == LevelClass::High || input.flapping_count_medium >= 2 {
            CooldownClass::Longer
        } else {
            CooldownClass::Base
        };

        let mut deescalation_lock = stability == LevelClass::High;
        if input.integrity != IntegrityState::Ok {
            deescalation_lock = true;
        }

        let mut reason_codes = ReasonSet::default();
        if stability == LevelClass::High {
            reason_codes.insert("RC.GV.SEROTONIN.TONE_HIGH");
        }
        if input.flapping_count_medium >= 6 {
            reason_codes.insert("RC.GV.FLAPPING.PENALTY");
        }
        if input.integrity != IntegrityState::Ok {
            reason_codes.insert("RC.RE.INTEGRITY.DEGRADED/FAIL");
        }

        SerOutput {
            stability,
            cooldown_class,
            deescalation_lock,
            reason_codes,
        }
    }

    fn snapshot_digest(&self) -> [u8; 32] {
        let mut bytes = Vec::new();
        bytes.extend(self.state.tone.to_le_bytes());
        bytes.push(self.state.inertia);
        bytes.push(self.state.lock_steps);
        bytes.extend(self.state.step_count.to_le_bytes());

        digest_meta("UCF:MC:SER", &bytes)
    }

    fn config_digest(&self) -> [u8; 32] {
        digest_config("UCF:MC:SER", &self.config)
    }
}

impl dbm_core::DbmModule for SerAttractorMicrocircuit {
    type Input = SerInput;
    type Output = SerOutput;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        self.step(input, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use microcircuit_serotonin_stub::SerRules;

    fn base_input() -> SerInput {
        SerInput {
            integrity: IntegrityState::Ok,
            stability_floor: LevelClass::Low,
            ..Default::default()
        }
    }

    fn severity(level: LevelClass) -> u8 {
        match level {
            LevelClass::Low => 0,
            LevelClass::Med => 1,
            LevelClass::High => 2,
        }
    }

    fn assert_no_less_conservative(input: SerInput) {
        let mut rules = SerRules::new();
        let mut micro = SerAttractorMicrocircuit::new(CircuitConfig::default());

        let rules_output = rules.tick(&input);
        let micro_output = micro.step(&input, 0);

        let critical = input.integrity != IntegrityState::Ok
            || input.replay_mismatch_present
            || input.receipt_invalid_count_medium >= 1;

        if critical {
            assert!(
                severity(micro_output.stability) >= severity(rules_output.stability),
                "micro stability {:?} < rules {:?}",
                micro_output.stability,
                rules_output.stability
            );
            if rules_output.deescalation_lock {
                assert!(micro_output.deescalation_lock);
            }
        }
    }

    #[test]
    fn determinism_sequence_and_digests() {
        let inputs = vec![
            SerInput {
                integrity: IntegrityState::Fail,
                replay_mismatch_present: true,
                receipt_invalid_count_medium: 1,
                stability_floor: LevelClass::Low,
                ..Default::default()
            },
            SerInput {
                flapping_count_medium: 6,
                dlp_critical_count_medium: 5,
                stability_floor: LevelClass::Med,
                ..Default::default()
            },
            base_input(),
        ];

        let run_sequence = |inputs: &[SerInput]| -> Vec<(SerOutput, [u8; 32])> {
            let mut circuit = SerAttractorMicrocircuit::new(CircuitConfig::default());
            inputs
                .iter()
                .map(|input| {
                    let output = circuit.step(input, 0);
                    let digest = circuit.snapshot_digest();
                    (output, digest)
                })
                .collect()
        };

        let outputs_a = run_sequence(&inputs);
        let outputs_b = run_sequence(&inputs);

        assert_eq!(outputs_a, outputs_b);
        for (_, digest) in outputs_a {
            assert_ne!(digest, [0u8; 32]);
        }
    }

    #[test]
    fn hysteresis_lock_steps_relax_slowly() {
        let mut circuit = SerAttractorMicrocircuit::new(CircuitConfig::default());
        let high_drive = SerInput {
            integrity: IntegrityState::Fail,
            replay_mismatch_present: true,
            receipt_invalid_count_medium: 1,
            stability_floor: LevelClass::Low,
            ..Default::default()
        };
        let low_drive = base_input();

        circuit.step(&high_drive, 0);
        let first_lock = circuit.state.lock_steps;
        circuit.step(&high_drive, 0);
        let second_lock = circuit.state.lock_steps;

        assert!(second_lock > first_lock);

        circuit.step(&low_drive, 0);
        let relaxed_lock = circuit.state.lock_steps;

        assert!(relaxed_lock < second_lock);
    }

    #[test]
    fn no_less_conservative_under_critical_inputs() {
        let cases = vec![
            SerInput {
                integrity: IntegrityState::Fail,
                ..base_input()
            },
            SerInput {
                replay_mismatch_present: true,
                ..base_input()
            },
            SerInput {
                receipt_invalid_count_medium: 2,
                ..base_input()
            },
        ];

        for input in cases {
            assert_no_less_conservative(input);
        }
    }
}
