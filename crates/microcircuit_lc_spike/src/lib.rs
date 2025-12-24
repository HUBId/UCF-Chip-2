#![forbid(unsafe_code)]

use dbm_core::{DbmModule, IntegrityState, LevelClass, ReasonSet};
use microcircuit_core::{digest_config, digest_meta, CircuitConfig, MicrocircuitBackend};
use microcircuit_lc_stub::{LcInput, LcOutput};

const MEMBRANE_MIN: i32 = 0;
const MEMBRANE_MAX: i32 = 100;
const MEMBRANE_DECAY: i32 = 3;
const MEMBRANE_REFRACTORY_DECAY: i32 = 5;
const SPIKE_THRESHOLD: i32 = 60;
const SPIKE_RESET: i32 = 30;
const REFRACTORY_STEPS: u32 = 2;
const SPIKE_COUNT_MAX: u32 = 255;

#[derive(Debug, Clone, Default)]
struct LcState {
    membrane: i32,
    refractory_steps: u32,
    spike_count_short: u32,
    tonic_floor: u8,
    step_count: u64,
}

#[derive(Debug, Clone)]
pub struct LcMicrocircuit {
    config: CircuitConfig,
    state: LcState,
}

impl LcMicrocircuit {
    pub fn new(config: CircuitConfig) -> Self {
        Self {
            config,
            state: LcState::default(),
        }
    }

    fn severity(level: LevelClass) -> u8 {
        match level {
            LevelClass::Low => 0,
            LevelClass::Med => 1,
            LevelClass::High => 2,
        }
    }

    fn tonic_floor_value(level: LevelClass) -> u8 {
        match level {
            LevelClass::Low => 0,
            LevelClass::Med => 1,
            LevelClass::High => 2,
        }
    }

    fn tonic_injection(level: LevelClass) -> i32 {
        match level {
            LevelClass::Low => 0,
            LevelClass::Med => 5,
            LevelClass::High => 10,
        }
    }

    fn encode_injection(input: &LcInput) -> i32 {
        let mut inj: i32 = 0;

        if input.integrity != IntegrityState::Ok {
            inj += 30;
        }
        if input.receipt_invalid_count_short >= 1 {
            inj += 25;
        }
        if input.dlp_critical_present_short {
            inj += 25;
        }
        if input.timeout_count_short >= 2 {
            inj += 20;
        }
        if input.deny_count_short >= 2 {
            inj += 10;
        }
        if input.receipt_missing_count_short >= 1 {
            inj += 5;
        }

        inj += Self::tonic_injection(input.arousal_floor);

        inj.clamp(0, 100)
    }

    fn rules_floor(input: &LcInput) -> LevelClass {
        if input.integrity != IntegrityState::Ok
            || input.receipt_invalid_count_short >= 1
            || input.dlp_critical_present_short
            || input.timeout_count_short >= 2
        {
            LevelClass::High
        } else if input.deny_count_short >= 2
            || input.receipt_missing_count_short >= 1
            || input.timeout_count_short == 1
        {
            LevelClass::Med
        } else {
            LevelClass::Low
        }
    }

    fn max_level(a: LevelClass, b: LevelClass) -> LevelClass {
        if Self::severity(a) >= Self::severity(b) {
            a
        } else {
            b
        }
    }
}

impl MicrocircuitBackend<LcInput, LcOutput> for LcMicrocircuit {
    fn step(&mut self, input: &LcInput, _now_ms: u64) -> LcOutput {
        self.state.step_count = self.state.step_count.saturating_add(1);
        self.state.tonic_floor = Self::tonic_floor_value(input.arousal_floor);

        let inj = Self::encode_injection(input);
        let mut spiked = false;

        if self.state.refractory_steps > 0 {
            self.state.refractory_steps = self.state.refractory_steps.saturating_sub(1);
            self.state.membrane =
                (self.state.membrane - MEMBRANE_REFRACTORY_DECAY).max(MEMBRANE_MIN);
        } else {
            let updated = self.state.membrane + inj - MEMBRANE_DECAY;
            self.state.membrane = updated.clamp(MEMBRANE_MIN, MEMBRANE_MAX);

            if self.state.membrane >= SPIKE_THRESHOLD {
                spiked = true;
                self.state.spike_count_short =
                    (self.state.spike_count_short + 1).min(SPIKE_COUNT_MAX);
                self.state.refractory_steps = REFRACTORY_STEPS;
                self.state.membrane = SPIKE_RESET;
            }
        }

        let mut arousal = if self.state.spike_count_short >= 2 || self.state.membrane >= 70 {
            LevelClass::High
        } else if self.state.spike_count_short == 1 || self.state.membrane >= 40 {
            LevelClass::Med
        } else {
            LevelClass::Low
        };

        let floor_from_rules = Self::rules_floor(input);
        arousal = Self::max_level(arousal, floor_from_rules);
        arousal = Self::max_level(arousal, input.arousal_floor);

        let hint_simulate_first = arousal != LevelClass::Low;
        let hint_novelty_lock = arousal == LevelClass::High;

        let mut reason_codes = ReasonSet::default();
        if spiked {
            reason_codes.insert("RC.GV.LC.SPIKE");
        }
        if input.integrity != IntegrityState::Ok {
            match input.integrity {
                IntegrityState::Degraded => reason_codes.insert("RC.RE.INTEGRITY.DEGRADED"),
                IntegrityState::Fail => reason_codes.insert("RC.RE.INTEGRITY.FAIL"),
                IntegrityState::Ok => {}
            }
        }
        if input.receipt_invalid_count_short >= 1 {
            reason_codes.insert("RC.GE.EXEC.DISPATCH_BLOCKED");
        }
        if input.dlp_critical_present_short {
            reason_codes.insert("RC.CD.DLP.SECRET_PATTERN");
        }

        LcOutput {
            arousal,
            hint_simulate_first,
            hint_novelty_lock,
            reason_codes,
        }
    }

    fn snapshot_digest(&self) -> [u8; 32] {
        let mut bytes = Vec::new();
        bytes.extend(self.state.membrane.to_le_bytes());
        bytes.extend(self.state.refractory_steps.to_le_bytes());
        bytes.extend(self.state.spike_count_short.to_le_bytes());
        bytes.push(self.state.tonic_floor);
        bytes.extend(self.state.step_count.to_le_bytes());
        bytes.extend(self.config.version.to_le_bytes());

        digest_meta("UCF:MC:LC", &bytes)
    }

    fn config_digest(&self) -> [u8; 32] {
        digest_config("UCF:MC:LC:SPIKE:CONFIG", &self.config)
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
    use biophys_core::ModulatorField;

    fn base_input() -> LcInput {
        LcInput {
            integrity: IntegrityState::Ok,
            arousal_floor: LevelClass::Low,
            ..Default::default()
        }
    }

    #[test]
    fn determinism_and_digest_stability() {
        let inputs = vec![
            LcInput {
                receipt_invalid_count_short: 1,
                arousal_floor: LevelClass::Med,
                ..base_input()
            },
            LcInput {
                dlp_critical_present_short: true,
                timeout_count_short: 2,
                arousal_floor: LevelClass::High,
                ..base_input()
            },
            LcInput {
                deny_count_short: 2,
                receipt_missing_count_short: 1,
                ..base_input()
            },
        ];

        let run_sequence = |inputs: &[LcInput]| -> Vec<(LcOutput, [u8; 32])> {
            let mut circuit = LcMicrocircuit::new(CircuitConfig::default());
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
    fn spikes_when_injection_is_high() {
        let mut circuit = LcMicrocircuit::new(CircuitConfig::default());
        let output = circuit.step(
            &LcInput {
                integrity: IntegrityState::Fail,
                receipt_invalid_count_short: 1,
                dlp_critical_present_short: true,
                timeout_count_short: 2,
                deny_count_short: 2,
                receipt_missing_count_short: 1,
                arousal_floor: LevelClass::High,
                modulators: ModulatorField::default(),
            },
            0,
        );

        assert_eq!(circuit.state.spike_count_short, 1);
        assert!(output
            .reason_codes
            .codes
            .contains(&"RC.GV.LC.SPIKE".to_string()));
        assert!(matches!(output.arousal, LevelClass::Med | LevelClass::High));
    }

    #[test]
    fn refractory_blocks_immediate_respike() {
        let mut circuit = LcMicrocircuit::new(CircuitConfig::default());
        let input = LcInput {
            integrity: IntegrityState::Fail,
            receipt_invalid_count_short: 1,
            dlp_critical_present_short: true,
            timeout_count_short: 2,
            deny_count_short: 2,
            receipt_missing_count_short: 1,
            arousal_floor: LevelClass::High,
            modulators: ModulatorField::default(),
        };

        circuit.step(&input, 0);
        let spike_count_after_first = circuit.state.spike_count_short;
        let output = circuit.step(&input, 0);

        assert_eq!(spike_count_after_first, 1);
        assert_eq!(circuit.state.spike_count_short, spike_count_after_first);
        assert!(!output
            .reason_codes
            .codes
            .contains(&"RC.GV.LC.SPIKE".to_string()));
    }
}
