#![forbid(unsafe_code)]

use dbm_core::{DwmMode, IntegrityState, LevelClass, ReasonSet, SalienceItem, SalienceSource};
use microcircuit_core::{digest_meta, CircuitConfig, MicrocircuitBackend};
use microcircuit_sn_stub::{SnInput, SnOutput};

const SCORE_MIN: i32 = -100;
const SCORE_MAX: i32 = 100;
const SCORE_DECAY: i32 = 2;
const INHIBITION_STEP: i32 = 5;
const INHIBITION_MARGIN: i32 = 10;
const HYSTERESIS_THRESHOLD: u8 = 3;

const IDX_EXEC_PLAN: usize = 0;
const IDX_SIMULATE: usize = 1;
const IDX_STABILIZE: usize = 2;
const IDX_REPORT: usize = 3;

#[derive(Debug, Clone)]
struct SnAttractorState {
    scores: [i32; 4],
    winner: u8,
    hysteresis: u8,
    step_count: u64,
}

impl Default for SnAttractorState {
    fn default() -> Self {
        Self {
            scores: [0; 4],
            winner: IDX_EXEC_PLAN as u8,
            hysteresis: 0,
            step_count: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SnAttractorMicrocircuit {
    config: CircuitConfig,
    state: SnAttractorState,
}

impl SnAttractorMicrocircuit {
    pub fn new(config: CircuitConfig) -> Self {
        Self {
            config,
            state: SnAttractorState::default(),
        }
    }

    fn severity_index(index: u8) -> u8 {
        index
    }

    fn mode_from_index(index: u8) -> DwmMode {
        match index {
            0 => DwmMode::ExecPlan,
            1 => DwmMode::Simulate,
            2 => DwmMode::Stabilize,
            3 => DwmMode::Report,
            _ => DwmMode::ExecPlan,
        }
    }

    fn intensity_for_contribution(value: i32) -> LevelClass {
        if value >= 40 {
            LevelClass::High
        } else if value >= 30 {
            LevelClass::Med
        } else {
            LevelClass::Low
        }
    }

    fn apply_decay(scores: &mut [i32; 4]) {
        for score in scores.iter_mut() {
            if *score > 0 {
                *score = (*score - SCORE_DECAY).max(0);
            } else if *score < 0 {
                *score = (*score + SCORE_DECAY).min(0);
            }
        }
    }

    fn clamp_scores(scores: &mut [i32; 4]) {
        for score in scores.iter_mut() {
            *score = (*score).clamp(SCORE_MIN, SCORE_MAX);
        }
    }

    fn apply_inhibition(scores: &mut [i32; 4]) {
        if let Some(&max_score) = scores.iter().max() {
            for score in scores.iter_mut() {
                if max_score - *score > INHIBITION_MARGIN {
                    *score -= INHIBITION_STEP;
                }
            }
        }
    }

    fn argmax_index(scores: &[i32; 4]) -> u8 {
        let mut best_idx = 0;
        let mut best_score = scores[0];
        for (idx, &score) in scores.iter().enumerate().skip(1) {
            if score > best_score {
                best_score = score;
                best_idx = idx;
            }
        }
        best_idx as u8
    }

    fn encode_contributions(
        input: &SnInput,
    ) -> ([i32; 4], Vec<(SalienceSource, i32)>) {
        let mut contrib = [0i32; 4];
        let mut salience = Vec::new();

        if input.isv.integrity == IntegrityState::Fail {
            contrib[IDX_REPORT] += 50;
            salience.push((SalienceSource::Integrity, 50));
        }

        if input.isv.threat == LevelClass::High {
            contrib[IDX_STABILIZE] += 40;
            salience.push((SalienceSource::Threat, 40));
        }

        let policy_pressure_high = input.isv.policy_pressure == LevelClass::High;
        let arousal_high = input.isv.arousal == LevelClass::High;
        if policy_pressure_high || arousal_high {
            contrib[IDX_SIMULATE] += 30;
            if policy_pressure_high {
                salience.push((SalienceSource::PolicyPressure, 30));
            }
        }

        let everything_low = input.isv.integrity == IntegrityState::Ok
            && input.isv.threat == LevelClass::Low
            && input.isv.policy_pressure == LevelClass::Low
            && input.isv.arousal == LevelClass::Low;
        if everything_low {
            contrib[IDX_EXEC_PLAN] += 20;
        }

        if input.replay_hint {
            contrib[IDX_SIMULATE] += 20;
            salience.push((SalienceSource::Replay, 20));
        }

        if input.reward_block {
            contrib[IDX_STABILIZE] += 10;
        }

        (contrib, salience)
    }

    fn build_salience(input: &SnInput, salience: &[(SalienceSource, i32)]) -> Vec<SalienceItem> {
        let mut items: Vec<SalienceItem> = salience
            .iter()
            .map(|(source, value)| {
                SalienceItem::new(
                    *source,
                    Self::intensity_for_contribution(*value),
                    input.isv.dominant_reason_codes.codes.clone(),
                )
            })
            .collect();

        items.sort_by(|a, b| {
            let intensity_ord =
                Self::severity_level(a.intensity).cmp(&Self::severity_level(b.intensity));
            if intensity_ord != std::cmp::Ordering::Equal {
                return intensity_ord.reverse();
            }
            (a.source as u8).cmp(&(b.source as u8))
        });

        if items.len() > 8 {
            items.truncate(8);
        }

        items
    }

    fn severity_level(level: LevelClass) -> u8 {
        match level {
            LevelClass::High => 2,
            LevelClass::Med => 1,
            LevelClass::Low => 0,
        }
    }
}

impl MicrocircuitBackend<SnInput, SnOutput> for SnAttractorMicrocircuit {
    fn step(&mut self, input: &SnInput, _now_ms: u64) -> SnOutput {
        self.state.step_count = self.state.step_count.saturating_add(1);

        let (contrib, salience_contribs) = Self::encode_contributions(input);

        Self::apply_decay(&mut self.state.scores);
        for (score, delta) in self.state.scores.iter_mut().zip(contrib.iter()) {
            *score += delta;
        }
        Self::clamp_scores(&mut self.state.scores);
        Self::apply_inhibition(&mut self.state.scores);
        Self::clamp_scores(&mut self.state.scores);

        let tentative = Self::argmax_index(&self.state.scores);
        let current = self.state.winner;
        let mut changed = false;

        if tentative == current {
            self.state.hysteresis = 0;
        } else if Self::severity_index(tentative) > Self::severity_index(current) {
            self.state.winner = tentative;
            self.state.hysteresis = 0;
            changed = true;
        } else {
            let stability_low = input.isv.stability == LevelClass::Low;
            if stability_low {
                self.state.hysteresis = (self.state.hysteresis.saturating_add(1)).min(10);
                if self.state.hysteresis >= HYSTERESIS_THRESHOLD {
                    self.state.winner = tentative;
                    self.state.hysteresis = 0;
                    changed = true;
                }
            } else {
                self.state.hysteresis = 0;
            }
        }

        let dwm = Self::mode_from_index(self.state.winner);

        let mut reason_codes = ReasonSet::default();
        if changed {
            reason_codes.insert("RC.GV.DWM.SWITCHED");
        }
        reason_codes.insert(match dwm {
            DwmMode::Report => "RC.GV.DWM.REPORT",
            DwmMode::Stabilize => "RC.GV.DWM.STABILIZE",
            DwmMode::Simulate => "RC.GV.DWM.SIMULATE",
            DwmMode::ExecPlan => "RC.GV.DWM.EXEC_PLAN",
        });

        let salience_items = Self::build_salience(input, &salience_contribs);

        SnOutput {
            dwm,
            salience_items,
            reason_codes,
        }
    }

    fn snapshot_digest(&self) -> [u8; 32] {
        let mut bytes = Vec::new();
        for score in self.state.scores {
            bytes.extend(score.to_le_bytes());
        }
        bytes.push(self.state.winner);
        bytes.push(self.state.hysteresis);
        bytes.extend(self.state.step_count.to_le_bytes());
        bytes.extend(self.config.version.to_le_bytes());

        digest_meta("UCF:MC:SN", &bytes)
    }
}

impl dbm_core::DbmModule for SnAttractorMicrocircuit {
    type Input = SnInput;
    type Output = SnOutput;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        self.step(input, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_input() -> SnInput {
        SnInput {
            isv: dbm_core::IsvSnapshot::default(),
            ..Default::default()
        }
    }

    fn severity(mode: DwmMode) -> u8 {
        match mode {
            DwmMode::ExecPlan => 0,
            DwmMode::Simulate => 1,
            DwmMode::Stabilize => 2,
            DwmMode::Report => 3,
        }
    }

    #[test]
    fn determinism_sequence_and_digests() {
        let inputs = vec![
            SnInput {
                isv: dbm_core::IsvSnapshot {
                    policy_pressure: LevelClass::High,
                    ..dbm_core::IsvSnapshot::default()
                },
                replay_hint: true,
                ..Default::default()
            },
            SnInput {
                isv: dbm_core::IsvSnapshot {
                    threat: LevelClass::High,
                    ..dbm_core::IsvSnapshot::default()
                },
                reward_block: true,
                ..Default::default()
            },
            base_input(),
        ];

        let run_sequence = |inputs: &[SnInput]| -> Vec<(SnOutput, [u8; 32])> {
            let mut circuit = SnAttractorMicrocircuit::new(CircuitConfig::default());
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
    fn strictness_preempts_to_report() {
        let mut circuit = SnAttractorMicrocircuit::new(CircuitConfig::default());
        let input = SnInput {
            isv: dbm_core::IsvSnapshot {
                integrity: IntegrityState::Fail,
                ..dbm_core::IsvSnapshot::default()
            },
            ..Default::default()
        };

        let output = circuit.step(&input, 0);
        assert_eq!(output.dwm, DwmMode::Report);
    }

    #[test]
    fn hysteresis_prevents_immediate_relax() {
        let mut circuit = SnAttractorMicrocircuit::new(CircuitConfig::default());
        let report_input = SnInput {
            isv: dbm_core::IsvSnapshot {
                integrity: IntegrityState::Fail,
                ..dbm_core::IsvSnapshot::default()
            },
            ..Default::default()
        };

        let calm = base_input();

        circuit.step(&report_input, 0);
        let first = circuit.step(&calm, 0);
        let second = circuit.step(&calm, 0);

        let mut final_output = second.clone();
        for _ in 0..4 {
            final_output = circuit.step(&calm, 0);
        }

        assert_eq!(first.dwm, DwmMode::Report);
        assert_eq!(second.dwm, DwmMode::Report);
        assert_eq!(final_output.dwm, DwmMode::ExecPlan);
    }

    #[test]
    fn no_less_conservative_under_critical() {
        let mut circuit = SnAttractorMicrocircuit::new(CircuitConfig::default());
        let critical_cases = vec![
            SnInput {
                isv: dbm_core::IsvSnapshot {
                    integrity: IntegrityState::Fail,
                    ..dbm_core::IsvSnapshot::default()
                },
                ..Default::default()
            },
            SnInput {
                isv: dbm_core::IsvSnapshot {
                    threat: LevelClass::High,
                    ..dbm_core::IsvSnapshot::default()
                },
                ..Default::default()
            },
        ];

        for input in critical_cases {
            let output = circuit.step(&input, 0);
            let expected = if input.isv.integrity == IntegrityState::Fail {
                DwmMode::Report
            } else {
                DwmMode::Stabilize
            };
            assert!(severity(output.dwm) >= severity(expected));
        }
    }

    #[test]
    fn bounded_outputs() {
        let mut circuit = SnAttractorMicrocircuit::new(CircuitConfig::default());
        let input = SnInput {
            isv: dbm_core::IsvSnapshot {
                integrity: IntegrityState::Fail,
                threat: LevelClass::High,
                policy_pressure: LevelClass::High,
                ..dbm_core::IsvSnapshot::default()
            },
            replay_hint: true,
            reward_block: true,
            ..Default::default()
        };

        let output = circuit.step(&input, 0);

        assert!(output.salience_items.len() <= 8);
        assert!(output.reason_codes.codes.len() <= ReasonSet::DEFAULT_MAX_LEN);
    }

    #[test]
    fn microcircuit_invariants_golden() {
        let mut circuit = SnAttractorMicrocircuit::new(CircuitConfig::default());
        let inputs = vec![
            base_input(),
            SnInput {
                isv: dbm_core::IsvSnapshot {
                    policy_pressure: LevelClass::High,
                    ..dbm_core::IsvSnapshot::default()
                },
                ..Default::default()
            },
            SnInput {
                isv: dbm_core::IsvSnapshot {
                    threat: LevelClass::High,
                    ..dbm_core::IsvSnapshot::default()
                },
                ..Default::default()
            },
            base_input(),
        ];

        let outputs: Vec<SnOutput> = inputs.iter().map(|input| circuit.step(input, 0)).collect();

        let dwm_sequence: Vec<DwmMode> = outputs.iter().map(|output| output.dwm).collect();
        assert_eq!(
            dwm_sequence,
            vec![
                DwmMode::ExecPlan,
                DwmMode::Simulate,
                DwmMode::Stabilize,
                DwmMode::Stabilize
            ]
        );

        for output in outputs {
            assert!(output.salience_items.len() <= 8);
            assert!(output.reason_codes.codes.contains(&format!(
                "RC.GV.DWM.{}",
                match output.dwm {
                    DwmMode::Report => "REPORT",
                    DwmMode::Stabilize => "STABILIZE",
                    DwmMode::Simulate => "SIMULATE",
                    DwmMode::ExecPlan => "EXEC_PLAN",
                }
            )));
        }
    }
}
