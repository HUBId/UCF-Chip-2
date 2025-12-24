#![forbid(unsafe_code)]

use dbm_core::{CooldownClass, DbmModule, IntegrityState, LevelClass, ReasonSet, ThreatVector};
use microcircuit_core::{digest_config, digest_meta, CircuitConfig, MicrocircuitBackend};
use microcircuit_pag_stub::{DefensePattern, PagInput, PagOutput};

const SCORE_MAX: u8 = 100;
const SCORE_DECAY: u8 = 5;
const LATCH_MAX: u8 = 20;

const DP1: u8 = 1;
const DP2: u8 = 2;
const DP3: u8 = 3;
const DP4: u8 = 4;

#[derive(Debug, Clone)]
struct PagAttractorState {
    score_dp1_freeze: u8,
    score_dp2_quarantine: u8,
    score_dp3_forensic: u8,
    score_dp4_contained: u8,
    winner: u8,
    latch_steps: u8,
    step_count: u64,
}

impl Default for PagAttractorState {
    fn default() -> Self {
        Self {
            score_dp1_freeze: 0,
            score_dp2_quarantine: 0,
            score_dp3_forensic: 0,
            score_dp4_contained: 0,
            winner: DP4,
            latch_steps: 0,
            step_count: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PagAttractorMicrocircuit {
    config: CircuitConfig,
    state: PagAttractorState,
}

#[derive(Debug, Clone, Copy, Default)]
struct PagDrives {
    freeze: u8,
    quarantine: u8,
    forensic: u8,
    contained: u8,
    strong_drive: bool,
    integrity_fail: bool,
    exfil_high: bool,
}

impl PagAttractorMicrocircuit {
    pub fn new(config: CircuitConfig) -> Self {
        Self {
            config,
            state: PagAttractorState::default(),
        }
    }

    fn clamp_drive(value: i32) -> u8 {
        value.clamp(0, SCORE_MAX as i32) as u8
    }

    fn encode_drives(input: &PagInput) -> PagDrives {
        let integrity_fail = matches!(input.integrity, IntegrityState::Fail);
        let exfil_high =
            input.vectors.contains(&ThreatVector::Exfil) && input.threat == LevelClass::High;

        let mut drives = PagDrives {
            integrity_fail,
            exfil_high,
            ..PagDrives::default()
        };

        let mut forensic = 0i32;
        let mut quarantine = 0i32;
        let mut freeze = 0i32;
        let mut contained = 0i32;

        if integrity_fail {
            forensic += 80;
        }

        if exfil_high {
            forensic += 70;
        }

        if input.threat == LevelClass::High && !integrity_fail {
            quarantine += 60;
        }

        if input.threat == LevelClass::Med {
            contained += 40;
        }

        if input.unlock_present && input.integrity == IntegrityState::Ok {
            contained += 20;
        }

        if input.stability == LevelClass::High {
            freeze += 30;
        }

        if input.serotonin_cooldown == CooldownClass::Longer {
            quarantine += 10;
            freeze += 10;
        }

        drives.forensic = Self::clamp_drive(forensic);
        drives.quarantine = Self::clamp_drive(quarantine);
        drives.freeze = Self::clamp_drive(freeze);
        drives.contained = Self::clamp_drive(contained);
        drives.strong_drive = integrity_fail || exfil_high || input.threat == LevelClass::High;

        drives
    }

    fn decay_score(score: u8) -> u8 {
        score.saturating_sub(SCORE_DECAY)
    }

    fn clamp_score(score: i32) -> u8 {
        score.clamp(0, SCORE_MAX as i32) as u8
    }

    fn severity(pattern: u8) -> u8 {
        match pattern {
            DP4 => 0,
            DP1 => 1,
            DP2 => 2,
            DP3 => 3,
            _ => 0,
        }
    }

    fn select_winner(scores: &[u8; 4]) -> u8 {
        let max_score = scores.iter().copied().max().unwrap_or(0);
        let order = [DP3, DP2, DP1, DP4];
        for candidate in order {
            let score = match candidate {
                DP1 => scores[0],
                DP2 => scores[1],
                DP3 => scores[2],
                DP4 => scores[3],
                _ => scores[3],
            };
            if score == max_score {
                return candidate;
            }
        }
        DP4
    }

    fn pattern_from_winner(winner: u8) -> DefensePattern {
        match winner {
            DP1 => DefensePattern::DP1_FREEZE,
            DP2 => DefensePattern::DP2_QUARANTINE,
            DP3 => DefensePattern::DP3_FORENSIC,
            DP4 => DefensePattern::DP4_CONTAINED_CONTINUE,
            _ => DefensePattern::DP4_CONTAINED_CONTINUE,
        }
    }

    fn build_reason_codes(input: &PagInput, winner: u8) -> ReasonSet {
        let mut reason_codes = ReasonSet::default();

        reason_codes.insert(match winner {
            DP1 => "RC.RX.ACTION.FREEZE",
            DP2 => "RC.RX.ACTION.QUARANTINE",
            DP3 => "RC.RX.ACTION.FORENSIC",
            DP4 => "RC.RX.ACTION.CONTAINED",
            _ => "RC.RX.ACTION.CONTAINED",
        });

        if matches!(input.integrity, IntegrityState::Fail)
            || input.vectors.contains(&ThreatVector::IntegrityCompromise)
        {
            reason_codes.insert("RC.TH.INTEGRITY_COMPROMISE");
        }
        if input.vectors.contains(&ThreatVector::Exfil) {
            reason_codes.insert("RC.TH.EXFIL.HIGH_CONFIDENCE");
        }
        if input.vectors.contains(&ThreatVector::Probing) {
            reason_codes.insert("RC.TH.POLICY_PROBING");
        }
        if input.vectors.contains(&ThreatVector::ToolSideEffects) {
            reason_codes.insert("RC.TH.TOOL_SIDE_EFFECTS");
        }

        reason_codes
    }
}

impl MicrocircuitBackend<PagInput, PagOutput> for PagAttractorMicrocircuit {
    fn step(&mut self, input: &PagInput, _now_ms: u64) -> PagOutput {
        self.state.step_count = self.state.step_count.saturating_add(1);

        let drives = Self::encode_drives(input);

        self.state.score_dp1_freeze = Self::decay_score(self.state.score_dp1_freeze);
        self.state.score_dp2_quarantine = Self::decay_score(self.state.score_dp2_quarantine);
        self.state.score_dp3_forensic = Self::decay_score(self.state.score_dp3_forensic);
        self.state.score_dp4_contained = Self::decay_score(self.state.score_dp4_contained);

        self.state.score_dp1_freeze =
            Self::clamp_score(self.state.score_dp1_freeze as i32 + drives.freeze as i32);
        self.state.score_dp2_quarantine =
            Self::clamp_score(self.state.score_dp2_quarantine as i32 + drives.quarantine as i32);
        self.state.score_dp3_forensic =
            Self::clamp_score(self.state.score_dp3_forensic as i32 + drives.forensic as i32);
        self.state.score_dp4_contained =
            Self::clamp_score(self.state.score_dp4_contained as i32 + drives.contained as i32);

        if drives.integrity_fail {
            self.state.score_dp3_forensic = SCORE_MAX;
        }
        if drives.exfil_high {
            self.state.score_dp3_forensic = self.state.score_dp3_forensic.max(80);
        }

        let scores = [
            self.state.score_dp1_freeze,
            self.state.score_dp2_quarantine,
            self.state.score_dp3_forensic,
            self.state.score_dp4_contained,
        ];
        let candidate = Self::select_winner(&scores);
        let previous = self.state.winner;
        let mut winner = candidate;

        if self.state.latch_steps > 0 && Self::severity(candidate) < Self::severity(previous) {
            winner = previous;
        }

        self.state.winner = winner;

        if drives.strong_drive && (winner == DP3 || winner == DP2) {
            self.state.latch_steps = LATCH_MAX;
        } else if self.state.latch_steps > 0 {
            self.state.latch_steps = self.state.latch_steps.saturating_sub(1);
        }

        let pattern = Self::pattern_from_winner(winner);
        let pattern_latched = (winner == DP2 || winner == DP3) && self.state.latch_steps > 0;
        let reason_codes = Self::build_reason_codes(input, winner);

        PagOutput {
            pattern,
            pattern_latched,
            reason_codes,
        }
    }

    fn snapshot_digest(&self) -> [u8; 32] {
        let mut bytes = vec![
            self.state.score_dp1_freeze,
            self.state.score_dp2_quarantine,
            self.state.score_dp3_forensic,
            self.state.score_dp4_contained,
            self.state.winner,
            self.state.latch_steps,
        ];
        bytes.extend(self.state.step_count.to_le_bytes());

        digest_meta("UCF:MC:PAG", &bytes)
    }

    fn config_digest(&self) -> [u8; 32] {
        digest_config("UCF:MC:PAG:CFG", &self.config)
    }
}

impl DbmModule for PagAttractorMicrocircuit {
    type Input = PagInput;
    type Output = PagOutput;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        self.step(input, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use biophys_core::ModulatorField;
    use microcircuit_pag_stub::PagRules;

    fn base_input() -> PagInput {
        PagInput {
            integrity: IntegrityState::Ok,
            threat: LevelClass::Low,
            vectors: Vec::new(),
            unlock_present: false,
            stability: LevelClass::Low,
            serotonin_cooldown: CooldownClass::Base,
            modulators: ModulatorField::default(),
        }
    }

    fn severity(pattern: DefensePattern) -> u8 {
        match pattern {
            DefensePattern::DP4_CONTAINED_CONTINUE => 0,
            DefensePattern::DP1_FREEZE => 1,
            DefensePattern::DP2_QUARANTINE => 2,
            DefensePattern::DP3_FORENSIC => 3,
        }
    }

    #[test]
    fn determinism_for_sequence() {
        let inputs = vec![
            PagInput {
                stability: LevelClass::High,
                ..base_input()
            },
            PagInput {
                threat: LevelClass::Med,
                ..base_input()
            },
            PagInput {
                threat: LevelClass::High,
                ..base_input()
            },
        ];

        let run = |inputs: &[PagInput]| {
            let mut circuit = PagAttractorMicrocircuit::new(CircuitConfig::default());
            inputs
                .iter()
                .map(|input| circuit.step(input, 0).pattern)
                .collect::<Vec<_>>()
        };

        assert_eq!(run(&inputs), run(&inputs));
    }

    #[test]
    fn integrity_fail_forces_forensic() {
        let mut circuit = PagAttractorMicrocircuit::new(CircuitConfig::default());
        let output = circuit.step(
            &PagInput {
                integrity: IntegrityState::Fail,
                ..base_input()
            },
            0,
        );

        assert_eq!(output.pattern, DefensePattern::DP3_FORENSIC);
    }

    #[test]
    fn exfil_high_forces_forensic() {
        let mut circuit = PagAttractorMicrocircuit::new(CircuitConfig::default());
        let output = circuit.step(
            &PagInput {
                threat: LevelClass::High,
                vectors: vec![ThreatVector::Exfil],
                ..base_input()
            },
            0,
        );

        assert_eq!(output.pattern, DefensePattern::DP3_FORENSIC);
    }

    #[test]
    fn high_threat_prefers_quarantine_or_forensic() {
        let mut circuit = PagAttractorMicrocircuit::new(CircuitConfig::default());
        let output = circuit.step(
            &PagInput {
                threat: LevelClass::High,
                ..base_input()
            },
            0,
        );

        assert!(
            matches!(
                output.pattern,
                DefensePattern::DP2_QUARANTINE | DefensePattern::DP3_FORENSIC
            ),
            "pattern was {:?}",
            output.pattern
        );
    }

    #[test]
    fn latch_prevents_immediate_relax() {
        let mut circuit = PagAttractorMicrocircuit::new(CircuitConfig::default());
        let forensic = PagInput {
            integrity: IntegrityState::Fail,
            ..base_input()
        };
        let calm = base_input();

        circuit.step(&forensic, 0);
        let first = circuit.step(&calm, 0);
        let second = circuit.step(&calm, 0);

        assert_eq!(first.pattern, DefensePattern::DP3_FORENSIC);
        assert_eq!(second.pattern, DefensePattern::DP3_FORENSIC);
    }

    #[test]
    fn invariants_vs_rules_backend() {
        let cases = vec![
            PagInput {
                integrity: IntegrityState::Fail,
                ..base_input()
            },
            PagInput {
                threat: LevelClass::High,
                vectors: vec![ThreatVector::Exfil],
                ..base_input()
            },
            PagInput {
                threat: LevelClass::High,
                ..base_input()
            },
        ];

        for input in cases {
            let mut rules = PagRules::new();
            let mut micro = PagAttractorMicrocircuit::new(CircuitConfig::default());

            let rules_output = rules.tick(&input);
            let micro_output = micro.step(&input, 0);

            assert!(
                severity(micro_output.pattern) >= severity(rules_output.pattern),
                "micro {:?} < rules {:?}",
                micro_output.pattern,
                rules_output.pattern
            );
        }
    }
}
