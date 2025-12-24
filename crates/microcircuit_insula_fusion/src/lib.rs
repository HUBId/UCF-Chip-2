#![forbid(unsafe_code)]

use dbm_core::{IntegrityState, IsvSnapshot, LevelClass, ReasonSet, ThreatVector};
use microcircuit_core::{digest_config, digest_meta, CircuitConfig, MicrocircuitBackend};
use microcircuit_insula_stub::InsulaInput;
use microcircuit_pag_stub::DefensePattern;

#[derive(Debug, Clone, Default)]
struct InsulaFusionState {
    pop_arousal: i32,
    pop_threat: i32,
    pop_stability: i32,
    pop_policy_pressure: i32,
    pop_progress: i32,
    pop_integrity: i32,
    step_count: u64,
    lock_integrity: u8,
    lock_threat: u8,
}

#[derive(Debug, Clone)]
pub struct InsulaFusionMicrocircuit {
    config: CircuitConfig,
    state: InsulaFusionState,
}

impl InsulaFusionMicrocircuit {
    pub fn new(config: CircuitConfig) -> Self {
        Self {
            config,
            state: InsulaFusionState::default(),
        }
    }

    fn level_drive(level: LevelClass) -> i32 {
        match level {
            LevelClass::Low => 10,
            LevelClass::Med => 40,
            LevelClass::High => 80,
        }
    }

    fn integrity_drive(state: IntegrityState) -> i32 {
        match state {
            IntegrityState::Ok => 10,
            IntegrityState::Degraded => 60,
            IntegrityState::Fail => 100,
        }
    }

    fn attract(pop: i32, drive: i32) -> i32 {
        let delta = (drive - pop) / 4;
        (pop + delta).clamp(0, 100)
    }

    fn level_from_pop(pop: i32) -> LevelClass {
        if pop >= 70 {
            LevelClass::High
        } else if pop >= 40 {
            LevelClass::Med
        } else {
            LevelClass::Low
        }
    }

    fn integrity_from_pop(pop: i32) -> IntegrityState {
        if pop >= 90 {
            IntegrityState::Fail
        } else if pop >= 50 {
            IntegrityState::Degraded
        } else {
            IntegrityState::Ok
        }
    }

    fn severity(level: LevelClass) -> u8 {
        match level {
            LevelClass::Low => 0,
            LevelClass::Med => 1,
            LevelClass::High => 2,
        }
    }

    fn integrity_severity(state: IntegrityState) -> u8 {
        match state {
            IntegrityState::Ok => 0,
            IntegrityState::Degraded => 1,
            IntegrityState::Fail => 2,
        }
    }

    fn defense_pattern_boost(pattern: Option<DefensePattern>) -> i32 {
        match pattern {
            Some(DefensePattern::DP2_QUARANTINE | DefensePattern::DP3_FORENSIC) => 20,
            _ => 0,
        }
    }

    fn threat_reason_code(vectors: &[ThreatVector]) -> &'static str {
        if vectors.contains(&ThreatVector::Exfil) {
            "RC.TH.EXFIL.HIGH_CONFIDENCE"
        } else if vectors.contains(&ThreatVector::IntegrityCompromise) {
            "RC.TH.INTEGRITY_COMPROMISE"
        } else {
            "RC.TH.EXFIL.HIGH_CONFIDENCE"
        }
    }

    fn apply_locks(
        &mut self,
        integrity_drive: i32,
        threat_drive: i32,
        integrity_state: IntegrityState,
        threat_level: LevelClass,
    ) {
        if integrity_state == IntegrityState::Fail {
            self.state.pop_integrity = 100;
            self.state.lock_integrity = 10;
        }

        if threat_drive >= 80 || threat_level == LevelClass::High {
            self.state.pop_threat = self.state.pop_threat.max(80);
            self.state.lock_threat = (self.state.lock_threat + 2).min(10);
        }

        if self.state.lock_integrity > 0 && integrity_drive < 60 {
            self.state.lock_integrity -= 1;
        }
        if self.state.lock_integrity > 0 {
            self.state.pop_integrity = self.state.pop_integrity.max(60);
        }

        if self.state.lock_threat > 0 && threat_drive < 60 {
            self.state.lock_threat -= 1;
        }
        if self.state.lock_threat > 0 {
            self.state.pop_threat = self.state.pop_threat.max(60);
        }
    }

    fn build_reason_codes(
        integrity: IntegrityState,
        threat: LevelClass,
        policy_pressure: LevelClass,
        arousal: LevelClass,
        threat_vectors: &[ThreatVector],
    ) -> ReasonSet {
        let mut reasons = ReasonSet::default();
        if integrity != IntegrityState::Ok {
            reasons.insert("RC.RE.INTEGRITY.DEGRADED/FAIL");
        }
        if threat == LevelClass::High {
            reasons.insert(Self::threat_reason_code(threat_vectors));
        }
        if policy_pressure == LevelClass::High {
            reasons.insert("RC.TH.POLICY_PROBING");
        }
        if arousal == LevelClass::High {
            reasons.insert("RC.RG.STATE.AROUSAL_UP");
        }
        reasons
    }
}

impl MicrocircuitBackend<InsulaInput, IsvSnapshot> for InsulaFusionMicrocircuit {
    fn step(&mut self, input: &InsulaInput, _now_ms: u64) -> IsvSnapshot {
        self.state.step_count = self.state.step_count.saturating_add(1);

        let mut arousal_drive = Self::level_drive(input.arousal);
        let mut threat_drive = Self::level_drive(input.threat);
        let mut stability_drive = Self::level_drive(input.stability);
        let mut policy_drive = Self::level_drive(input.policy_pressure);
        let mut progress_drive = Self::level_drive(input.progress);
        let integrity_drive = Self::integrity_drive(input.integrity);

        let pattern_boost = Self::defense_pattern_boost(input.pag_pattern);
        if pattern_boost > 0 {
            threat_drive += pattern_boost;
            stability_drive += pattern_boost;
        }

        if input.stn_hold_active {
            arousal_drive += 10;
            stability_drive += 10;
        }

        arousal_drive = arousal_drive.clamp(0, 100);
        threat_drive = threat_drive.clamp(0, 100);
        stability_drive = stability_drive.clamp(0, 100);
        policy_drive = policy_drive.clamp(0, 100);
        progress_drive = progress_drive.clamp(0, 100);

        self.state.pop_arousal = Self::attract(self.state.pop_arousal, arousal_drive);
        self.state.pop_threat = Self::attract(self.state.pop_threat, threat_drive);
        self.state.pop_stability = Self::attract(self.state.pop_stability, stability_drive);
        self.state.pop_policy_pressure =
            Self::attract(self.state.pop_policy_pressure, policy_drive);
        self.state.pop_progress = Self::attract(self.state.pop_progress, progress_drive);
        self.state.pop_integrity = Self::attract(self.state.pop_integrity, integrity_drive);

        self.apply_locks(integrity_drive, threat_drive, input.integrity, input.threat);

        let arousal = Self::level_from_pop(self.state.pop_arousal);
        let mut threat = Self::level_from_pop(self.state.pop_threat);
        let stability = Self::level_from_pop(self.state.pop_stability);
        let policy_pressure = Self::level_from_pop(self.state.pop_policy_pressure);
        let progress = Self::level_from_pop(self.state.pop_progress);
        let mut integrity = Self::integrity_from_pop(self.state.pop_integrity);

        if Self::integrity_severity(integrity) < Self::integrity_severity(input.integrity) {
            integrity = input.integrity;
        }
        if Self::severity(threat) < Self::severity(input.threat) {
            threat = input.threat;
        }
        if threat == LevelClass::High && self.state.lock_threat == 0 {
            self.state.lock_threat = 1;
        }

        let dominant_reason_codes = Self::build_reason_codes(
            integrity,
            threat,
            policy_pressure,
            arousal,
            &input.threat_vectors,
        );

        IsvSnapshot {
            arousal,
            threat,
            stability,
            policy_pressure,
            progress,
            integrity,
            dominant_reason_codes,
            threat_vectors: if input.threat_vectors.is_empty() {
                None
            } else {
                Some(input.threat_vectors.clone())
            },
            replay_hint: false,
        }
    }

    fn snapshot_digest(&self) -> [u8; 32] {
        let mut bytes = Vec::new();
        bytes.extend(self.state.pop_arousal.to_le_bytes());
        bytes.extend(self.state.pop_threat.to_le_bytes());
        bytes.extend(self.state.pop_stability.to_le_bytes());
        bytes.extend(self.state.pop_policy_pressure.to_le_bytes());
        bytes.extend(self.state.pop_progress.to_le_bytes());
        bytes.extend(self.state.pop_integrity.to_le_bytes());
        bytes.push(self.state.lock_integrity);
        bytes.push(self.state.lock_threat);
        bytes.extend(self.state.step_count.to_le_bytes());

        digest_meta("UCF:MC:INSULA", &bytes)
    }

    fn config_digest(&self) -> [u8; 32] {
        digest_config("UCF:MC:INSULA:CFG", &self.config)
    }
}

impl dbm_core::DbmModule for InsulaFusionMicrocircuit {
    type Input = InsulaInput;
    type Output = IsvSnapshot;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        self.step(input, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dbm_core::DbmModule;
    use microcircuit_insula_stub::InsulaRules;

    fn base_input() -> InsulaInput {
        InsulaInput {
            integrity: IntegrityState::Ok,
            policy_pressure: LevelClass::Low,
            progress: LevelClass::Low,
            arousal: LevelClass::Low,
            stability: LevelClass::Low,
            threat: LevelClass::Low,
            ..Default::default()
        }
    }

    fn run_sequence(inputs: &[InsulaInput]) -> Vec<(IsvSnapshot, [u8; 32])> {
        let mut circuit = InsulaFusionMicrocircuit::new(CircuitConfig::default());
        inputs
            .iter()
            .map(|input| {
                let output = circuit.step(input, 0);
                let digest = circuit.snapshot_digest();
                (output, digest)
            })
            .collect()
    }

    #[test]
    fn determinism_sequence_and_digests() {
        let inputs = vec![
            InsulaInput {
                integrity: IntegrityState::Fail,
                threat: LevelClass::High,
                arousal: LevelClass::High,
                stability: LevelClass::Med,
                policy_pressure: LevelClass::High,
                progress: LevelClass::Med,
                threat_vectors: vec![ThreatVector::Exfil],
                pag_pattern: Some(DefensePattern::DP3_FORENSIC),
                stn_hold_active: true,
                ..Default::default()
            },
            InsulaInput {
                threat: LevelClass::Med,
                arousal: LevelClass::Med,
                stability: LevelClass::Med,
                policy_pressure: LevelClass::Low,
                progress: LevelClass::High,
                ..Default::default()
            },
            base_input(),
        ];

        let outputs_a = run_sequence(&inputs);
        let outputs_b = run_sequence(&inputs);

        assert_eq!(outputs_a, outputs_b);
        for (_, digest) in outputs_a {
            assert_ne!(digest, [0u8; 32]);
        }
    }

    #[test]
    fn integrity_fail_locks_state() {
        let mut circuit = InsulaFusionMicrocircuit::new(CircuitConfig::default());
        let input = InsulaInput {
            integrity: IntegrityState::Fail,
            ..base_input()
        };

        let output = circuit.step(&input, 0);
        assert_eq!(output.integrity, IntegrityState::Fail);
        assert_eq!(circuit.state.lock_integrity, 10);
        assert_eq!(circuit.state.pop_integrity, 100);
    }

    #[test]
    fn threat_high_locks_for_multiple_steps() {
        let mut circuit = InsulaFusionMicrocircuit::new(CircuitConfig::default());
        let high = InsulaInput {
            threat: LevelClass::High,
            ..base_input()
        };
        let low = base_input();

        let output = circuit.step(&high, 0);
        assert_eq!(output.threat, LevelClass::High);
        assert!(circuit.state.lock_threat > 0);

        let output_low = circuit.step(&low, 0);
        assert!(circuit.state.lock_threat > 0);
        assert!(matches!(
            output_low.threat,
            LevelClass::Med | LevelClass::High
        ));
    }

    #[test]
    fn conservative_clamps_do_not_downplay_inputs() {
        let mut circuit = InsulaFusionMicrocircuit::new(CircuitConfig::default());
        let input = InsulaInput {
            integrity: IntegrityState::Degraded,
            threat: LevelClass::High,
            ..base_input()
        };

        let output = circuit.step(&input, 0);
        assert!(
            InsulaFusionMicrocircuit::integrity_severity(output.integrity)
                >= InsulaFusionMicrocircuit::integrity_severity(input.integrity)
        );
        assert!(
            InsulaFusionMicrocircuit::severity(output.threat)
                >= InsulaFusionMicrocircuit::severity(input.threat)
        );
    }

    #[test]
    fn micro_is_not_less_conservative_than_rules_on_critical_inputs() {
        let cases = vec![
            InsulaInput {
                integrity: IntegrityState::Fail,
                ..base_input()
            },
            InsulaInput {
                threat: LevelClass::High,
                ..base_input()
            },
        ];

        for input in cases {
            let mut rules = InsulaRules::new();
            let mut micro = InsulaFusionMicrocircuit::new(CircuitConfig::default());

            let rules_output = rules.tick(&input);
            let micro_output = micro.step(&input, 0);

            if input.integrity != IntegrityState::Ok {
                assert!(
                    InsulaFusionMicrocircuit::integrity_severity(micro_output.integrity)
                        >= InsulaFusionMicrocircuit::integrity_severity(rules_output.integrity)
                );
            }
            if input.threat == LevelClass::High {
                assert!(
                    InsulaFusionMicrocircuit::severity(micro_output.threat)
                        >= InsulaFusionMicrocircuit::severity(rules_output.threat)
                );
            }
        }
    }

    #[test]
    fn benign_inputs_are_not_less_conservative_than_rules() {
        let input = InsulaInput {
            integrity: IntegrityState::Ok,
            threat: LevelClass::Low,
            policy_pressure: LevelClass::Low,
            ..base_input()
        };

        let mut rules = InsulaRules::new();
        let mut micro = InsulaFusionMicrocircuit::new(CircuitConfig::default());

        let rules_output = rules.tick(&input);
        let micro_output = micro.step(&input, 0);

        assert!(
            InsulaFusionMicrocircuit::integrity_severity(micro_output.integrity)
                >= InsulaFusionMicrocircuit::integrity_severity(rules_output.integrity)
        );
        assert!(
            InsulaFusionMicrocircuit::severity(micro_output.threat)
                >= InsulaFusionMicrocircuit::severity(rules_output.threat)
        );
    }
}
