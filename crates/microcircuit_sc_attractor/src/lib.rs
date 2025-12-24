#![forbid(unsafe_code)]

use dbm_core::{
    DwmMode, IntegrityState, LevelClass, OrientTarget, ReasonSet, SalienceSource, ThreatVector,
    UrgencyClass,
};
use microcircuit_core::{digest_config, digest_meta, CircuitConfig, MicrocircuitBackend};
use microcircuit_sc_stub::{ScInput, ScOutput};

const SCORE_MIN: i32 = -100;
const SCORE_MAX: i32 = 100;
const SCORE_DECAY: i32 = 5;

const IDX_INTEGRITY: usize = 0;
const IDX_DLP: usize = 1;
const IDX_RECOVERY: usize = 2;
const IDX_REPLAY: usize = 3;
const IDX_POLICY_PRESSURE: usize = 4;
const IDX_APPROVAL: usize = 5;

const IDX_INTEGRITY_U8: u8 = 0;
const IDX_DLP_U8: u8 = 1;
const IDX_RECOVERY_U8: u8 = 2;
const IDX_REPLAY_U8: u8 = 3;
const IDX_POLICY_PRESSURE_U8: u8 = 4;
const IDX_APPROVAL_U8: u8 = 5;

#[derive(Debug, Clone)]
struct ScAttractorState {
    scores: [i32; 6],
    winner: u8,
    lock_steps: u8,
    step_count: u64,
}

impl Default for ScAttractorState {
    fn default() -> Self {
        Self {
            scores: [0; 6],
            winner: IDX_APPROVAL_U8,
            lock_steps: 0,
            step_count: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScAttractorMicrocircuit {
    config: CircuitConfig,
    state: ScAttractorState,
}

impl ScAttractorMicrocircuit {
    pub fn new(config: CircuitConfig) -> Self {
        Self {
            config,
            state: ScAttractorState::default(),
        }
    }

    fn strictness_rank(index: u8) -> u8 {
        match index {
            IDX_INTEGRITY_U8 => 0,
            IDX_DLP_U8 => 1,
            IDX_RECOVERY_U8 => 2,
            IDX_REPLAY_U8 => 3,
            IDX_POLICY_PRESSURE_U8 => 4,
            _ => 5,
        }
    }

    fn target_from_index(index: u8) -> OrientTarget {
        match index {
            IDX_INTEGRITY_U8 => OrientTarget::Integrity,
            IDX_DLP_U8 => OrientTarget::Dlp,
            IDX_RECOVERY_U8 => OrientTarget::Recovery,
            IDX_REPLAY_U8 => OrientTarget::Replay,
            IDX_POLICY_PRESSURE_U8 => OrientTarget::PolicyPressure,
            _ => OrientTarget::Approval,
        }
    }

    fn dwm_from_target(target: OrientTarget) -> DwmMode {
        match target {
            OrientTarget::Integrity => DwmMode::Report,
            OrientTarget::Dlp => DwmMode::Stabilize,
            OrientTarget::Recovery => DwmMode::Report,
            OrientTarget::Replay => DwmMode::Simulate,
            OrientTarget::PolicyPressure => DwmMode::Simulate,
            OrientTarget::Approval => DwmMode::Report,
        }
    }

    fn apply_decay(scores: &mut [i32; 6]) {
        for score in scores.iter_mut() {
            if *score > 0 {
                *score = (*score - SCORE_DECAY).max(0);
            } else if *score < 0 {
                *score = (*score + SCORE_DECAY).min(0);
            }
        }
    }

    fn clamp_scores(scores: &mut [i32; 6]) {
        for score in scores.iter_mut() {
            *score = (*score).clamp(SCORE_MIN, SCORE_MAX);
        }
    }

    fn argmax_index(scores: &[i32; 6]) -> u8 {
        let mut best_idx = 0;
        let mut best_score = scores[0];
        let mut best_rank = Self::strictness_rank(0);
        for (idx, &score) in scores.iter().enumerate().skip(1) {
            let rank = Self::strictness_rank(idx as u8);
            if score > best_score || (score == best_score && rank < best_rank) {
                best_score = score;
                best_idx = idx;
                best_rank = rank;
            }
        }
        best_idx as u8
    }

    fn replay_mismatch_present(input: &ScInput) -> bool {
        if input.replay_mismatch_present {
            return true;
        }
        input
            .isv
            .dominant_reason_codes
            .codes
            .iter()
            .any(|code| code.contains("replay_mismatch"))
    }

    fn replay_salience_present(input: &ScInput) -> bool {
        input
            .salience_items
            .iter()
            .any(|item| item.source == SalienceSource::Replay)
    }

    fn dlp_critical(input: &ScInput) -> bool {
        if input.dlp_critical_present {
            return true;
        }
        input.isv.threat == LevelClass::High
            && input
                .isv
                .threat_vectors
                .as_ref()
                .map(|vectors| vectors.contains(&ThreatVector::Exfil))
                .unwrap_or(false)
    }

    fn encode_drives(input: &ScInput) -> [i32; 6] {
        let mut drives = [0i32; 6];

        if input.integrity == IntegrityState::Fail {
            drives[IDX_INTEGRITY] += 80;
        } else if input.integrity == IntegrityState::Degraded {
            drives[IDX_INTEGRITY] += 40;
        }

        if Self::replay_mismatch_present(input) {
            drives[IDX_INTEGRITY] += 30;
        }

        if input.isv.threat == LevelClass::High
            && input
                .isv
                .threat_vectors
                .as_ref()
                .map(|vectors| vectors.contains(&ThreatVector::Exfil))
                .unwrap_or(false)
        {
            drives[IDX_DLP] += 70;
        }

        if input.dlp_critical_present {
            drives[IDX_DLP] += 70;
        }

        if input.unlock_present {
            drives[IDX_RECOVERY] += 40;
        }

        if input.replay_planned_present || Self::replay_salience_present(input) {
            drives[IDX_REPLAY] += 40;
        }

        if input.isv.policy_pressure == LevelClass::High {
            drives[IDX_POLICY_PRESSURE] += 50;
        }

        drives[IDX_APPROVAL] += 10;

        drives
    }

    fn apply_lock(&mut self, tentative: u8) {
        let current = self.state.winner;
        let current_rank = Self::strictness_rank(current);
        let tentative_rank = Self::strictness_rank(tentative);

        if self.state.lock_steps > 0 && tentative_rank > current_rank {
            self.state.lock_steps = self.state.lock_steps.saturating_sub(1);
            return;
        }

        self.state.winner = tentative;

        if matches!(self.state.winner, IDX_INTEGRITY_U8 | IDX_DLP_U8) {
            self.state.lock_steps = 10;
        }
    }

    fn urgency_for_target(score: i32, target: OrientTarget) -> UrgencyClass {
        if matches!(target, OrientTarget::Integrity | OrientTarget::Dlp) || score >= 70 {
            UrgencyClass::High
        } else if score >= 40 {
            UrgencyClass::Med
        } else {
            UrgencyClass::Low
        }
    }

    fn insert_causes(input: &ScInput, reasons: &mut ReasonSet) {
        if matches!(
            input.integrity,
            IntegrityState::Fail | IntegrityState::Degraded
        ) || Self::replay_mismatch_present(input)
        {
            reasons.insert("RC.GV.ORIENT.CAUSE_INTEGRITY");
        }

        if Self::dlp_critical(input) {
            reasons.insert("RC.GV.ORIENT.CAUSE_DLP");
        }

        if input
            .isv
            .threat_vectors
            .as_ref()
            .map(|vectors| vectors.contains(&ThreatVector::Probing))
            .unwrap_or(false)
        {
            reasons.insert("RC.GV.ORIENT.CAUSE_PROBING");
        }
    }
}

impl MicrocircuitBackend<ScInput, ScOutput> for ScAttractorMicrocircuit {
    fn step(&mut self, input: &ScInput, _now_ms: u64) -> ScOutput {
        self.state.step_count = self.state.step_count.saturating_add(1);

        let drives = Self::encode_drives(input);

        Self::apply_decay(&mut self.state.scores);
        for (score, delta) in self.state.scores.iter_mut().zip(drives.iter()) {
            *score += delta;
        }
        Self::clamp_scores(&mut self.state.scores);

        let tentative = Self::argmax_index(&self.state.scores);
        self.apply_lock(tentative);

        let target = Self::target_from_index(self.state.winner);
        let score = self.state.scores[self.state.winner as usize];

        let mut reason_codes = ReasonSet::default();
        reason_codes.insert(match target {
            OrientTarget::Integrity => "RC.GV.ORIENT.TARGET_INTEGRITY",
            OrientTarget::Dlp => "RC.GV.ORIENT.TARGET_DLP",
            OrientTarget::Recovery => "RC.GV.ORIENT.TARGET_RECOVERY",
            OrientTarget::Approval => "RC.GV.ORIENT.TARGET_APPROVAL",
            OrientTarget::Replay => "RC.GV.ORIENT.TARGET_REPLAY",
            OrientTarget::PolicyPressure => "RC.GV.ORIENT.TARGET_POLICY_PRESSURE",
        });
        Self::insert_causes(input, &mut reason_codes);

        ScOutput {
            target,
            urgency: Self::urgency_for_target(score, target),
            recommended_dwm: Self::dwm_from_target(target),
            reason_codes,
        }
    }

    fn snapshot_digest(&self) -> [u8; 32] {
        let mut bytes = Vec::new();
        for score in self.state.scores {
            bytes.extend(score.to_le_bytes());
        }
        bytes.push(self.state.winner);
        bytes.push(self.state.lock_steps);
        bytes.extend(self.state.step_count.to_le_bytes());

        digest_meta("UCF:MC:SC", &bytes)
    }

    fn config_digest(&self) -> [u8; 32] {
        digest_config("UCF:MC:SC:CFG", &self.config)
    }
}

impl dbm_core::DbmModule for ScAttractorMicrocircuit {
    type Input = ScInput;
    type Output = ScOutput;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        self.step(input, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dbm_core::DbmModule;

    fn base_input() -> ScInput {
        ScInput {
            isv: dbm_core::IsvSnapshot::default(),
            ..Default::default()
        }
    }

    #[test]
    fn determinism_same_inputs() {
        let input = ScInput {
            isv: dbm_core::IsvSnapshot {
                policy_pressure: LevelClass::High,
                ..Default::default()
            },
            replay_planned_present: true,
            ..Default::default()
        };

        let mut a = ScAttractorMicrocircuit::new(CircuitConfig::default());
        let mut b = ScAttractorMicrocircuit::new(CircuitConfig::default());

        let out_a = a.step(&input, 0);
        let out_b = b.step(&input, 0);

        assert_eq!(out_a, out_b);
        assert_ne!(a.snapshot_digest(), [0u8; 32]);
    }

    #[test]
    fn integrity_fail_drives_integrity_high_report() {
        let mut circuit = ScAttractorMicrocircuit::new(CircuitConfig::default());
        let input = ScInput {
            integrity: IntegrityState::Fail,
            ..base_input()
        };

        let output = circuit.step(&input, 0);

        assert_eq!(output.target, OrientTarget::Integrity);
        assert_eq!(output.urgency, UrgencyClass::High);
        assert_eq!(output.recommended_dwm, DwmMode::Report);
    }

    #[test]
    fn exfil_drives_dlp_high_stabilize() {
        let mut circuit = ScAttractorMicrocircuit::new(CircuitConfig::default());
        let input = ScInput {
            isv: dbm_core::IsvSnapshot {
                threat: LevelClass::High,
                threat_vectors: Some(vec![ThreatVector::Exfil]),
                ..Default::default()
            },
            ..base_input()
        };

        let output = circuit.step(&input, 0);

        assert_eq!(output.target, OrientTarget::Dlp);
        assert_eq!(output.urgency, UrgencyClass::High);
        assert_eq!(output.recommended_dwm, DwmMode::Stabilize);
    }

    #[test]
    fn lock_prevents_relaxing_too_fast() {
        let mut circuit = ScAttractorMicrocircuit::new(CircuitConfig::default());
        let fail_input = ScInput {
            integrity: IntegrityState::Fail,
            ..base_input()
        };
        let calm = base_input();

        let first = circuit.step(&fail_input, 0);
        let second = circuit.step(&calm, 0);
        let third = circuit.step(&calm, 0);

        assert_eq!(first.target, OrientTarget::Integrity);
        assert_eq!(second.target, OrientTarget::Integrity);
        assert_eq!(third.target, OrientTarget::Integrity);
    }

    #[test]
    fn stricter_target_preempts_lock() {
        let mut circuit = ScAttractorMicrocircuit::new(CircuitConfig::default());
        let dlp_input = ScInput {
            isv: dbm_core::IsvSnapshot {
                threat: LevelClass::High,
                threat_vectors: Some(vec![ThreatVector::Exfil]),
                ..Default::default()
            },
            ..base_input()
        };
        let integrity_input = ScInput {
            integrity: IntegrityState::Fail,
            ..base_input()
        };

        let first = circuit.step(&dlp_input, 0);
        let second = circuit.step(&integrity_input, 0);

        assert_eq!(first.target, OrientTarget::Dlp);
        assert_eq!(second.target, OrientTarget::Integrity);
    }

    #[test]
    fn invariants_match_rules_backend() {
        let mut micro = ScAttractorMicrocircuit::new(CircuitConfig::default());
        let mut rules = microcircuit_sc_stub::ScRules::new();

        let integrity_input = ScInput {
            integrity: IntegrityState::Fail,
            ..base_input()
        };
        let exfil_input = ScInput {
            isv: dbm_core::IsvSnapshot {
                threat: LevelClass::High,
                threat_vectors: Some(vec![ThreatVector::Exfil]),
                ..Default::default()
            },
            ..base_input()
        };

        let micro_integrity = micro.step(&integrity_input, 0);
        let rules_integrity = rules.tick(&integrity_input);
        assert_eq!(micro_integrity.target, rules_integrity.target);
        assert_eq!(micro_integrity.urgency, UrgencyClass::High);
        assert_eq!(micro_integrity.recommended_dwm, DwmMode::Report);

        let mut micro_exfil_circuit = ScAttractorMicrocircuit::new(CircuitConfig::default());
        let micro_exfil = micro_exfil_circuit.step(&exfil_input, 0);
        let rules_exfil = rules.tick(&exfil_input);
        assert_eq!(micro_exfil.target, rules_exfil.target);
        assert_eq!(micro_exfil.recommended_dwm, DwmMode::Stabilize);
    }
}
