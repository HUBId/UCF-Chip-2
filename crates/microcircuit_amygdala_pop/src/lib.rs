#![forbid(unsafe_code)]

use dbm_core::{LevelClass, ReasonSet, ThreatVector};
use microcircuit_amygdala_stub::{AmyInput, AmyOutput};
use microcircuit_core::{digest_config, digest_meta, CircuitConfig, MicrocircuitBackend};

#[derive(Debug, Clone, Default)]
struct AmyPopState {
    pop_exfil: u8,
    pop_integrity: u8,
    pop_probing: u8,
    pop_tool_side_effects: u8,
    pop_escape: u8,
    latch_integrity: u8,
    latch_exfil: u8,
    step_count: u64,
}

#[derive(Debug, Clone)]
pub struct AmygdalaPopMicrocircuit {
    config: CircuitConfig,
    state: AmyPopState,
}

impl AmygdalaPopMicrocircuit {
    pub fn new(config: CircuitConfig) -> Self {
        Self {
            config,
            state: AmyPopState::default(),
        }
    }

    fn clamp_drive(value: i32) -> u8 {
        value.clamp(0, 100) as u8
    }

    fn integrity_drive(input: &AmyInput) -> u8 {
        let mut drive = 0;

        match input.integrity {
            dbm_core::IntegrityState::Fail => drive += 60,
            dbm_core::IntegrityState::Degraded => drive += 30,
            dbm_core::IntegrityState::Ok => {}
        }

        if input.replay_mismatch_present {
            drive += 40;
        }

        if input.receipt_invalid_medium >= 1 {
            drive += 30;
        }

        Self::clamp_drive(drive)
    }

    fn exfil_drive(input: &AmyInput) -> u8 {
        let mut drive = 0;

        if input.dlp_secret_present || input.dlp_obfuscation_present || input.dlp_stegano_present {
            drive += 60;
        }

        if input.dlp_critical_count_med >= 1 {
            drive += 30;
        }

        Self::clamp_drive(drive)
    }

    fn probing_drive(input: &AmyInput) -> u8 {
        let mut drive = 0;

        if input.policy_pressure == LevelClass::High {
            drive += 40;
        }

        if input.deny_storm_present {
            drive += 30;
        }

        Self::clamp_drive(drive)
    }

    fn tool_side_effects_drive(input: &AmyInput) -> u8 {
        let mut drive = 0;
        let tool_anomaly_high = input.tool_anomaly_present
            || input
                .tool_anomalies
                .iter()
                .any(|(_, level)| matches!(level, LevelClass::High));

        if tool_anomaly_high {
            drive += 40;
        }

        if input.divergence == LevelClass::High {
            drive += 30;
        }

        Self::clamp_drive(drive)
    }

    fn update_population(pop: &mut u8, drive: u8) {
        let mut value = *pop as i32;
        value += (i32::from(drive) - 20) / 4;
        value = value.clamp(0, 100);

        if drive < 20 {
            value = (value - 2).max(0);
        }

        *pop = value as u8;
    }

    fn update_latch(latch: &mut u8, drive: u8) {
        if drive >= 50 {
            *latch = (*latch + 2).min(10);
        }

        if drive < 30 {
            *latch = latch.saturating_sub(1);
        }
    }

    fn build_vectors(
        integrity_active: bool,
        exfil_active: bool,
        probing_active: bool,
        tool_active: bool,
        escape_active: bool,
    ) -> Vec<ThreatVector> {
        let mut vectors = Vec::new();
        for vector in [
            ThreatVector::IntegrityCompromise,
            ThreatVector::Exfil,
            ThreatVector::Probing,
            ThreatVector::ToolSideEffects,
            ThreatVector::RuntimeEscape,
        ] {
            let active = match vector {
                ThreatVector::IntegrityCompromise => integrity_active,
                ThreatVector::Exfil => exfil_active,
                ThreatVector::Probing => probing_active,
                ThreatVector::ToolSideEffects => tool_active,
                ThreatVector::RuntimeEscape => escape_active,
            };

            if active {
                vectors.push(vector);
            }
        }

        if vectors.len() > 8 {
            vectors.truncate(8);
        }

        vectors
    }
}

impl MicrocircuitBackend<AmyInput, AmyOutput> for AmygdalaPopMicrocircuit {
    fn step(&mut self, input: &AmyInput, _now_ms: u64) -> AmyOutput {
        let integrity_drive = Self::integrity_drive(input);
        let exfil_drive = Self::exfil_drive(input);
        let probing_drive = Self::probing_drive(input);
        let tool_drive = Self::tool_side_effects_drive(input);
        let escape_drive = 0;

        Self::update_population(&mut self.state.pop_integrity, integrity_drive);
        Self::update_population(&mut self.state.pop_exfil, exfil_drive);
        Self::update_population(&mut self.state.pop_probing, probing_drive);
        Self::update_population(&mut self.state.pop_tool_side_effects, tool_drive);
        Self::update_population(&mut self.state.pop_escape, escape_drive);

        Self::update_latch(&mut self.state.latch_integrity, integrity_drive);
        Self::update_latch(&mut self.state.latch_exfil, exfil_drive);

        self.state.step_count = self.state.step_count.saturating_add(1);

        let integrity_active = self.state.pop_integrity >= 50 || self.state.latch_integrity > 0;
        let exfil_active = self.state.pop_exfil >= 50 || self.state.latch_exfil > 0;
        let probing_active = self.state.pop_probing >= 50;
        let tool_active = self.state.pop_tool_side_effects >= 50;
        let escape_active = self.state.pop_escape >= 50;

        let threat = if integrity_active || exfil_active {
            LevelClass::High
        } else if tool_active || probing_active {
            LevelClass::Med
        } else {
            LevelClass::Low
        };

        let vectors = Self::build_vectors(
            integrity_active,
            exfil_active,
            probing_active,
            tool_active,
            escape_active,
        );

        let mut reason_codes = ReasonSet::default();
        if integrity_active {
            reason_codes.insert("RC.TH.INTEGRITY_COMPROMISE");
        }
        if exfil_active {
            reason_codes.insert("RC.TH.EXFIL.HIGH_CONFIDENCE");
        }
        if probing_active {
            reason_codes.insert("RC.TH.POLICY_PROBING");
        }
        if tool_active {
            reason_codes.insert("RC.TH.TOOL_SIDE_EFFECTS");
        }

        AmyOutput {
            threat,
            vectors,
            reason_codes,
        }
    }

    fn snapshot_digest(&self) -> [u8; 32] {
        let mut bytes = vec![
            self.state.pop_exfil,
            self.state.pop_integrity,
            self.state.pop_probing,
            self.state.pop_tool_side_effects,
            self.state.pop_escape,
            self.state.latch_integrity,
            self.state.latch_exfil,
        ];
        bytes.extend(self.state.step_count.to_le_bytes());

        digest_meta("UCF:MC:AMY", &bytes)
    }

    fn config_digest(&self) -> [u8; 32] {
        digest_config("UCF:MC:AMY:CONFIG", &self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dbm_core::IntegrityState;

    fn base_input() -> AmyInput {
        AmyInput {
            integrity: IntegrityState::Ok,
            replay_mismatch_present: false,
            dlp_secret_present: false,
            dlp_obfuscation_present: false,
            dlp_stegano_present: false,
            dlp_critical_count_med: 0,
            receipt_invalid_medium: 0,
            policy_pressure: LevelClass::Low,
            deny_storm_present: false,
            sealed: None,
            tool_anomaly_present: false,
            tool_anomalies: Vec::new(),
            divergence: LevelClass::Low,
        }
    }

    #[test]
    fn deterministic_sequence_is_repeatable() {
        let mut circuit_a = AmygdalaPopMicrocircuit::new(CircuitConfig::default());
        let mut circuit_b = AmygdalaPopMicrocircuit::new(CircuitConfig::default());
        let sequence = vec![
            AmyInput {
                dlp_secret_present: true,
                ..base_input()
            },
            AmyInput {
                policy_pressure: LevelClass::High,
                ..base_input()
            },
            base_input(),
            AmyInput {
                replay_mismatch_present: true,
                ..base_input()
            },
        ];

        for input in sequence {
            let out_a = circuit_a.step(&input, 0);
            let out_b = circuit_b.step(&input, 0);
            assert_eq!(out_a, out_b);
        }
    }

    #[test]
    fn dlp_secret_sets_exfil_and_high_threat() {
        let mut circuit = AmygdalaPopMicrocircuit::new(CircuitConfig::default());
        let output = circuit.step(
            &AmyInput {
                dlp_secret_present: true,
                ..base_input()
            },
            0,
        );

        assert_eq!(output.threat, LevelClass::High);
        assert!(output.vectors.contains(&ThreatVector::Exfil));
    }

    #[test]
    fn replay_mismatch_sets_integrity_and_high_threat() {
        let mut circuit = AmygdalaPopMicrocircuit::new(CircuitConfig::default());
        let mut output = AmyOutput::default();
        for _ in 0..10 {
            output = circuit.step(
                &AmyInput {
                    replay_mismatch_present: true,
                    ..base_input()
                },
                0,
            );
        }

        assert_eq!(output.threat, LevelClass::High);
        assert!(output.vectors.contains(&ThreatVector::IntegrityCompromise));
    }

    #[test]
    fn probing_sets_med_threat() {
        let mut circuit = AmygdalaPopMicrocircuit::new(CircuitConfig::default());
        let mut output = AmyOutput::default();
        for _ in 0..10 {
            output = circuit.step(
                &AmyInput {
                    policy_pressure: LevelClass::High,
                    ..base_input()
                },
                0,
            );
        }

        assert_eq!(output.threat, LevelClass::Med);
        assert!(output.vectors.contains(&ThreatVector::Probing));
    }

    #[test]
    fn latch_keeps_exfil_active_after_drop() {
        let mut circuit = AmygdalaPopMicrocircuit::new(CircuitConfig::default());
        let initial = circuit.step(
            &AmyInput {
                dlp_secret_present: true,
                ..base_input()
            },
            0,
        );
        let _ = circuit.step(
            &AmyInput {
                dlp_secret_present: true,
                ..base_input()
            },
            0,
        );
        assert!(initial.vectors.contains(&ThreatVector::Exfil));

        let after_drop = circuit.step(&base_input(), 0);
        assert!(after_drop.vectors.contains(&ThreatVector::Exfil));

        let after_second_drop = circuit.step(&base_input(), 0);
        assert!(after_second_drop.vectors.contains(&ThreatVector::Exfil));

        let _ = circuit.step(&base_input(), 0);
        let after_decay = circuit.step(&base_input(), 0);
        assert!(!after_decay.vectors.contains(&ThreatVector::Exfil));
    }
}
