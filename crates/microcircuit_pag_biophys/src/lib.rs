#![forbid(unsafe_code)]

use biophys_core::{
    LifParams, LifState, ModChannel, NeuronId, PopCode, StpParams, StpState, STP_SCALE,
};
use biophys_runtime::BiophysRuntime;
use dbm_core::{DbmModule, IntegrityState, LevelClass, ReasonSet, ThreatVector};
use microcircuit_core::{digest_meta, CircuitConfig, MicrocircuitBackend};
use microcircuit_pag_stub::{DefensePattern, PagInput, PagOutput};

const POOL_COUNT: usize = 4;
const POOL_SIZE: usize = 4;

const NEURON_COUNT: usize = 18;
const EXCITATORY_COUNT: usize = 16;
const INHIBITORY_START: usize = 16;

const DP1_POOL_START: usize = 0;
const DP2_POOL_START: usize = 4;
const DP3_POOL_START: usize = 8;
const DP4_POOL_START: usize = 12;

const DT_MS: u16 = 1;
const MAX_SPIKES_PER_STEP: usize = 18;
const MAX_EVENTS_PER_STEP: usize = 512;

const POOL_ACC_MAX: i32 = 100;
const POOL_ACC_DECAY: i32 = 6;
const POOL_ACC_SPIKE_GAIN: i32 = 10;

const LATCH_MAX: u8 = 20;

const CURRENT_BASELINE: i32 = 8;
const CURRENT_RECOVERY: i32 = 4;
const CURRENT_MODERATE: i32 = 14;
const CURRENT_STRONG: i32 = 22;
const CURRENT_VERY_STRONG: i32 = 30;
const CURRENT_OVERRIDE: i32 = 40;

const EXCITATORY_WEIGHT: i32 = 7;
const INHIBITORY_WEIGHT: i32 = -10;
const CROSS_INHIBIT_STRONG: i32 = -14;
const CROSS_INHIBIT_MODERATE: i32 = -9;

const EXCITATORY_STP: StpParams = StpParams {
    u: 250,
    tau_rec_steps: 5,
    tau_fac_steps: 3,
    mod_channel: None,
};

const NO_STP: StpParams = StpParams {
    u: STP_SCALE,
    tau_rec_steps: 0,
    tau_fac_steps: 0,
    mod_channel: None,
};

#[derive(Debug, Clone)]
struct PagBiophysState {
    pool_acc: [u8; POOL_COUNT],
    winner: u8,
    latch_steps: u8,
}

impl Default for PagBiophysState {
    fn default() -> Self {
        Self {
            pool_acc: [0; POOL_COUNT],
            winner: 3,
            latch_steps: 0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct PagDrives {
    dp1: i32,
    dp2: i32,
    dp3: i32,
    dp4: i32,
    integrity_fail: bool,
    exfil_active: bool,
    dp3_drive_low: bool,
}

#[derive(Debug, Clone)]
pub struct PagBiophysMicrocircuit {
    _config: CircuitConfig,
    runtime: BiophysRuntime,
    state: PagBiophysState,
}

impl PagBiophysMicrocircuit {
    pub fn new(config: CircuitConfig) -> Self {
        let params = vec![
            LifParams {
                tau_ms: 10,
                v_rest: 0,
                v_reset: 0,
                v_threshold: 20,
            };
            NEURON_COUNT
        ];
        let states = vec![
            LifState {
                v: 0,
                refractory_steps: 0,
            };
            NEURON_COUNT
        ];
        let (edges, stp_params) = build_edges();
        let runtime = BiophysRuntime::new_with_synapses(
            params,
            states,
            DT_MS,
            MAX_SPIKES_PER_STEP,
            edges,
            stp_params,
            MAX_EVENTS_PER_STEP,
        );
        Self {
            _config: config,
            runtime,
            state: PagBiophysState::default(),
        }
    }

    fn encode_drives(input: &PagInput) -> PagDrives {
        let integrity_fail = matches!(input.integrity, IntegrityState::Fail);
        let exfil_active = input.vectors.contains(&ThreatVector::Exfil);
        let tool_side_effects = input.vectors.contains(&ThreatVector::ToolSideEffects);

        let mut dp3: i32 = 0;
        if integrity_fail {
            dp3 += CURRENT_VERY_STRONG;
            dp3 = dp3.saturating_add(CURRENT_OVERRIDE);
        }
        if exfil_active {
            dp3 = dp3.saturating_add(CURRENT_STRONG);
        }

        let mut dp2: i32 = 0;
        if input.threat == LevelClass::High && !integrity_fail {
            dp2 = dp2.saturating_add(CURRENT_STRONG);
        }
        if tool_side_effects {
            dp2 = dp2.saturating_add(CURRENT_MODERATE);
        }

        let mut dp1: i32 = 0;
        if input.stability == LevelClass::High {
            dp1 = dp1.saturating_add(CURRENT_MODERATE);
        }

        let major_threat = integrity_fail || exfil_active || input.threat == LevelClass::High;
        let mut dp4: i32 = 0;
        if !major_threat {
            dp4 = dp4.saturating_add(CURRENT_BASELINE);
        }
        if input.unlock_present {
            dp4 = dp4.saturating_add(CURRENT_RECOVERY);
        }

        PagDrives {
            dp1,
            dp2,
            dp3,
            dp4,
            integrity_fail,
            exfil_active,
            dp3_drive_low: dp3 <= CURRENT_MODERATE,
        }
    }

    fn apply_pool_currents(currents: &mut [i32; NEURON_COUNT], start: usize, drive: i32) {
        for value in currents.iter_mut().skip(start).take(POOL_SIZE) {
            *value = value.saturating_add(drive);
        }
    }

    fn pool_for_neuron(idx: usize) -> Option<usize> {
        if idx < DP2_POOL_START {
            Some(0)
        } else if idx < DP3_POOL_START {
            Some(1)
        } else if idx < DP4_POOL_START {
            Some(2)
        } else if idx < EXCITATORY_COUNT {
            Some(3)
        } else {
            None
        }
    }

    fn update_pool_acc(current: u8, spikes: usize) -> u8 {
        let delta = spikes as i32 * POOL_ACC_SPIKE_GAIN - POOL_ACC_DECAY;
        let updated = (current as i32 + delta).clamp(0, POOL_ACC_MAX);
        updated as u8
    }

    fn select_winner(scores: &[u8; POOL_COUNT]) -> u8 {
        let max_score = scores.iter().copied().max().unwrap_or(0);
        let order = [2u8, 1u8, 0u8, 3u8];
        for candidate in order {
            let score = scores[candidate as usize];
            if score == max_score {
                return candidate;
            }
        }
        3
    }

    fn severity(winner: u8) -> u8 {
        match winner {
            3 => 0,
            0 => 1,
            1 => 2,
            2 => 3,
            _ => 0,
        }
    }

    fn pattern_from_winner(winner: u8) -> DefensePattern {
        match winner {
            0 => DefensePattern::DP1_FREEZE,
            1 => DefensePattern::DP2_QUARANTINE,
            2 => DefensePattern::DP3_FORENSIC,
            _ => DefensePattern::DP4_CONTAINED_CONTINUE,
        }
    }

    fn build_reason_codes(input: &PagInput, winner: u8) -> ReasonSet {
        let mut reason_codes = ReasonSet::default();

        reason_codes.insert(match winner {
            0 => "RC.RX.ACTION.FREEZE",
            1 => "RC.RX.ACTION.QUARANTINE",
            2 => "RC.RX.ACTION.FORENSIC",
            _ => "RC.RX.ACTION.CONTAINED",
        });

        if matches!(input.integrity, IntegrityState::Fail)
            || input.vectors.contains(&ThreatVector::IntegrityCompromise)
        {
            reason_codes.insert("RC.TH.INTEGRITY_COMPROMISE");
        }

        for vector in &input.vectors {
            match vector {
                ThreatVector::Exfil => reason_codes.insert("RC.TH.EXFIL.HIGH_CONFIDENCE"),
                ThreatVector::Probing => reason_codes.insert("RC.TH.POLICY_PROBING"),
                ThreatVector::IntegrityCompromise => {
                    reason_codes.insert("RC.TH.INTEGRITY_COMPROMISE")
                }
                ThreatVector::RuntimeEscape => reason_codes.insert("RC.TH.RUNTIME_ESCAPE"),
                ThreatVector::ToolSideEffects => reason_codes.insert("RC.TH.TOOL_SIDE_EFFECTS"),
            }
        }

        reason_codes
    }
}

impl MicrocircuitBackend<PagInput, PagOutput> for PagBiophysMicrocircuit {
    fn step(&mut self, input: &PagInput, _now_ms: u64) -> PagOutput {
        let drives = Self::encode_drives(input);

        let mut currents = [0i32; NEURON_COUNT];
        Self::apply_pool_currents(&mut currents, DP1_POOL_START, drives.dp1);
        Self::apply_pool_currents(&mut currents, DP2_POOL_START, drives.dp2);
        Self::apply_pool_currents(&mut currents, DP3_POOL_START, drives.dp3);
        Self::apply_pool_currents(&mut currents, DP4_POOL_START, drives.dp4);

        self.runtime.set_modulators(input.modulators);
        let PopCode { spikes } = self.runtime.step(&currents);
        let mut pool_spikes = [0usize; POOL_COUNT];
        for spike in &spikes {
            if let Some(pool_idx) = Self::pool_for_neuron(spike.0 as usize) {
                pool_spikes[pool_idx] = pool_spikes[pool_idx].saturating_add(1);
            }
        }

        for (idx, acc) in self.state.pool_acc.iter_mut().enumerate() {
            *acc = Self::update_pool_acc(*acc, pool_spikes[idx]);
        }

        if drives.integrity_fail {
            self.state.pool_acc[2] = POOL_ACC_MAX as u8;
        }
        if drives.exfil_active {
            self.state.pool_acc[2] = POOL_ACC_MAX as u8;
        }

        let candidate = Self::select_winner(&self.state.pool_acc);
        let previous = self.state.winner;
        let mut winner = candidate;

        if self.state.latch_steps > 0 && Self::severity(candidate) < Self::severity(previous) {
            winner = previous;
        }

        self.state.winner = winner;

        if winner == 2 || winner == 1 {
            self.state.latch_steps = LATCH_MAX;
        } else if self.state.latch_steps > 0 {
            let decay = if drives.dp3_drive_low && !drives.integrity_fail && !drives.exfil_active {
                2
            } else {
                1
            };
            self.state.latch_steps = self.state.latch_steps.saturating_sub(decay);
        }

        let pattern = Self::pattern_from_winner(winner);
        let pattern_latched = (winner == 1 || winner == 2) && self.state.latch_steps > 0;
        let reason_codes = Self::build_reason_codes(input, winner);

        PagOutput {
            pattern,
            pattern_latched,
            reason_codes,
        }
    }

    fn snapshot_digest(&self) -> [u8; 32] {
        digest_meta("UCF:BIO:PAG:SNAP", &self.runtime.snapshot_digest())
    }

    fn config_digest(&self) -> [u8; 32] {
        digest_meta("UCF:BIO:PAG:CFG", &self.runtime.config_digest())
    }
}

impl DbmModule for PagBiophysMicrocircuit {
    type Input = PagInput;
    type Output = PagOutput;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        self.step(input, 0)
    }
}

fn build_edges() -> (Vec<biophys_core::SynapseEdge>, Vec<StpParams>) {
    let mut edges = Vec::new();
    let mut params = Vec::new();

    let pool_starts = [
        DP1_POOL_START,
        DP2_POOL_START,
        DP3_POOL_START,
        DP4_POOL_START,
    ];

    for &start in &pool_starts {
        for pre in start..start + POOL_SIZE {
            for post in start..start + POOL_SIZE {
                if pre == post {
                    continue;
                }
                let delay_steps = 1 + ((pre + post) % 2) as u16;
                edges.push(biophys_core::SynapseEdge {
                    pre: NeuronId(pre as u32),
                    post: NeuronId(post as u32),
                    weight_base: EXCITATORY_WEIGHT,
                    weight_effective: EXCITATORY_WEIGHT,
                    delay_steps,
                    mod_channel: ModChannel::None,
                    stp: StpState {
                        x: STP_SCALE,
                        u: EXCITATORY_STP.u,
                    },
                });
                params.push(EXCITATORY_STP);
            }
        }
    }

    for pre in INHIBITORY_START..NEURON_COUNT {
        for post in 0..EXCITATORY_COUNT {
            edges.push(biophys_core::SynapseEdge {
                pre: NeuronId(pre as u32),
                post: NeuronId(post as u32),
                weight_base: INHIBITORY_WEIGHT,
                weight_effective: INHIBITORY_WEIGHT,
                delay_steps: 1,
                mod_channel: ModChannel::None,
                stp: StpState {
                    x: STP_SCALE,
                    u: NO_STP.u,
                },
            });
            params.push(NO_STP);
        }
    }

    for pre in DP3_POOL_START..DP3_POOL_START + POOL_SIZE {
        for post in DP4_POOL_START..DP4_POOL_START + POOL_SIZE {
            edges.push(biophys_core::SynapseEdge {
                pre: NeuronId(pre as u32),
                post: NeuronId(post as u32),
                weight_base: CROSS_INHIBIT_STRONG,
                weight_effective: CROSS_INHIBIT_STRONG,
                delay_steps: 1,
                mod_channel: ModChannel::None,
                stp: StpState {
                    x: STP_SCALE,
                    u: NO_STP.u,
                },
            });
            params.push(NO_STP);
        }
    }

    for pre in DP2_POOL_START..DP2_POOL_START + POOL_SIZE {
        for post in DP4_POOL_START..DP4_POOL_START + POOL_SIZE {
            edges.push(biophys_core::SynapseEdge {
                pre: NeuronId(pre as u32),
                post: NeuronId(post as u32),
                weight_base: CROSS_INHIBIT_MODERATE,
                weight_effective: CROSS_INHIBIT_MODERATE,
                delay_steps: 1,
                mod_channel: ModChannel::None,
                stp: StpState {
                    x: STP_SCALE,
                    u: NO_STP.u,
                },
            });
            params.push(NO_STP);
        }
    }

    (edges, params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use biophys_core::ModulatorField;
    use dbm_core::CooldownClass;
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
            let mut circuit = PagBiophysMicrocircuit::new(CircuitConfig::default());
            inputs
                .iter()
                .map(|input| circuit.step(input, 0).pattern)
                .collect::<Vec<_>>()
        };

        assert_eq!(run(&inputs), run(&inputs));
    }

    #[test]
    fn integrity_fail_forces_forensic() {
        let mut circuit = PagBiophysMicrocircuit::new(CircuitConfig::default());
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
    fn exfil_forces_forensic() {
        let mut circuit = PagBiophysMicrocircuit::new(CircuitConfig::default());
        let output = circuit.step(
            &PagInput {
                vectors: vec![ThreatVector::Exfil],
                threat: LevelClass::Med,
                ..base_input()
            },
            0,
        );

        assert_eq!(output.pattern, DefensePattern::DP3_FORENSIC);
    }

    #[test]
    fn high_threat_prefers_quarantine_or_forensic() {
        let mut circuit = PagBiophysMicrocircuit::new(CircuitConfig::default());
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
        let mut circuit = PagBiophysMicrocircuit::new(CircuitConfig::default());
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
                vectors: vec![ThreatVector::Exfil],
                threat: LevelClass::High,
                ..base_input()
            },
            PagInput {
                threat: LevelClass::High,
                ..base_input()
            },
        ];

        for input in cases {
            let mut rules = PagRules::new();
            let mut micro = PagBiophysMicrocircuit::new(CircuitConfig::default());

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
