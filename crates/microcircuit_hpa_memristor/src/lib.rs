#![forbid(unsafe_code)]

use dbm_core::{IntegrityState, LevelClass, ReasonSet};
use memristor_backend::{CellKey, EmulatedMemristorBackend, MemristorBackend, DEFAULT_MAX_VALUE};
use microcircuit_core::{digest_meta, CircuitConfig, MicrocircuitBackend};

const STABLE_DECAY_INTERVAL: u16 = 5;
const OFFSET_MAX: u16 = 10;
const CALIBRATION_INTERVAL: u32 = 24;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HpaInput {
    pub integrity_state: IntegrityState,
    pub replay_mismatch_present: bool,
    pub dlp_critical_present: bool,
    pub receipt_invalid_present: bool,
    pub deny_storm_present: bool,
    pub timeouts_burst_present: bool,
    pub unlock_present: bool,
    pub stable_medium_window: bool,
    pub calibrate_now: bool,
}

impl Default for HpaInput {
    fn default() -> Self {
        Self {
            integrity_state: IntegrityState::Ok,
            replay_mismatch_present: false,
            dlp_critical_present: false,
            receipt_invalid_present: false,
            deny_storm_present: false,
            timeouts_burst_present: false,
            unlock_present: false,
            stable_medium_window: false,
            calibrate_now: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HpaOutput {
    pub allostatic_load_class: LevelClass,
    pub baseline_caution_offset: u16,
    pub baseline_novelty_dampening_offset: u16,
    pub baseline_approval_strictness_offset: u16,
    pub baseline_export_strictness_offset: u16,
    pub baseline_chain_conservatism_offset: u16,
    pub baseline_cooldown_multiplier_class: u16,
    pub reason_codes: ReasonSet,
    pub mem_snapshot_digest: Option<[u8; 32]>,
}

impl Default for HpaOutput {
    fn default() -> Self {
        Self {
            allostatic_load_class: LevelClass::Low,
            baseline_caution_offset: 0,
            baseline_novelty_dampening_offset: 0,
            baseline_approval_strictness_offset: 0,
            baseline_export_strictness_offset: 0,
            baseline_chain_conservatism_offset: 0,
            baseline_cooldown_multiplier_class: 0,
            reason_codes: ReasonSet::default(),
            mem_snapshot_digest: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HpaCircuit<B: MemristorBackend> {
    config: CircuitConfig,
    backend: B,
    step_count: u64,
    stable_window_counter: u16,
    calibration_counter: u32,
}

impl<B: MemristorBackend> HpaCircuit<B> {
    pub fn new(config: CircuitConfig, backend: B) -> Self {
        Self {
            config,
            backend,
            step_count: 0,
            stable_window_counter: 0,
            calibration_counter: 0,
        }
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn read_cell(&self, key: CellKey) -> u16 {
        self.backend.read_cell(key)
    }

    pub fn current_output(&self) -> HpaOutput {
        self.derive_output(ReasonSet::default(), Some(self.backend.snapshot_digest()))
    }

    fn bump_cell(&mut self, key: CellKey, amount: u16) {
        let value = self.backend.read_cell(key);
        let updated = value.saturating_add(amount);
        self.backend.write_cell(key, updated);
    }

    fn set_cell(&mut self, key: CellKey, value: u16) {
        self.backend.write_cell(key, value);
    }

    fn decay_cell(&mut self, key: CellKey, amount: u16) {
        let value = self.backend.read_cell(key);
        let updated = value.saturating_sub(amount);
        self.backend.write_cell(key, updated);
    }

    fn maybe_calibrate(&mut self, reasons: &mut ReasonSet) {
        if self.calibration_counter >= CALIBRATION_INTERVAL {
            self.calibration_counter = 0;
            let report = self.backend.calibrate();
            if report.applied {
                reasons.insert("RC.GV.HPA.CALIBRATED");
            }
        }
    }

    fn derive_output(
        &self,
        reasons: ReasonSet,
        mem_snapshot_digest: Option<[u8; 32]>,
    ) -> HpaOutput {
        let allostatic_load = self.backend.read_cell(CellKey::AL);
        let integrity_sens = self.backend.read_cell(CellKey::INTEGRITY_SENS);
        let exfil_sens = self.backend.read_cell(CellKey::EXFIL_SENS);
        let probing_sens = self.backend.read_cell(CellKey::PROBING_SENS);
        let reliability_sens = self.backend.read_cell(CellKey::RELIABILITY_SENS);

        let allostatic_load_class = match allostatic_load {
            v if v >= 70 => LevelClass::High,
            v if v >= 30 => LevelClass::Med,
            _ => LevelClass::Low,
        };

        let caution_offset = clamp_offset(
            div_ceil(allostatic_load, 10) + div_ceil(integrity_sens, 20) + div_ceil(exfil_sens, 20),
        );
        let novelty_offset =
            clamp_offset(div_ceil(allostatic_load, 10) + div_ceil(probing_sens, 20));
        let approval_offset = clamp_offset(
            div_ceil(allostatic_load, 10)
                + div_ceil(integrity_sens, 20)
                + div_ceil(probing_sens, 20),
        );
        let export_offset = clamp_offset(div_ceil(exfil_sens, 10) + div_ceil(allostatic_load, 20));
        let chain_offset = clamp_offset(
            div_ceil(reliability_sens, 10)
                + div_ceil(probing_sens, 20)
                + div_ceil(allostatic_load, 20),
        );

        let cooldown_multiplier_class = if allostatic_load >= 80 {
            3
        } else if allostatic_load >= 50 {
            2
        } else if allostatic_load >= 20 {
            1
        } else {
            0
        };

        HpaOutput {
            allostatic_load_class,
            baseline_caution_offset: caution_offset,
            baseline_novelty_dampening_offset: novelty_offset,
            baseline_approval_strictness_offset: approval_offset,
            baseline_export_strictness_offset: export_offset,
            baseline_chain_conservatism_offset: chain_offset,
            baseline_cooldown_multiplier_class: cooldown_multiplier_class,
            reason_codes: reasons,
            mem_snapshot_digest,
        }
    }
}

impl HpaCircuit<EmulatedMemristorBackend> {
    pub fn new_emulated(config: CircuitConfig, seed: u64) -> Self {
        let backend = EmulatedMemristorBackend::new(seed, DEFAULT_MAX_VALUE);
        Self::new(config, backend)
    }
}

impl<B: MemristorBackend> MicrocircuitBackend<HpaInput, HpaOutput> for HpaCircuit<B> {
    fn step(&mut self, input: &HpaInput, _now_ms: u64) -> HpaOutput {
        self.step_count = self.step_count.saturating_add(1);
        let mut reasons = ReasonSet::default();
        let mut stable_window = input.stable_medium_window;

        if matches!(input.integrity_state, IntegrityState::Fail) {
            self.bump_cell(CellKey::AL, 5);
            self.bump_cell(CellKey::INTEGRITY_SENS, 5);
            self.set_cell(CellKey::RECOVERY_CONF, 0);
            stable_window = false;
            reasons.insert("hpa_integrity_fail");
        }

        if input.dlp_critical_present {
            self.bump_cell(CellKey::AL, 2);
            self.bump_cell(CellKey::EXFIL_SENS, 3);
            stable_window = false;
            reasons.insert("hpa_dlp_critical");
        }

        if input.receipt_invalid_present {
            self.bump_cell(CellKey::AL, 2);
            self.bump_cell(CellKey::INTEGRITY_SENS, 3);
            stable_window = false;
            reasons.insert("hpa_receipt_invalid");
        }

        if input.replay_mismatch_present {
            self.bump_cell(CellKey::AL, 2);
            self.bump_cell(CellKey::INTEGRITY_SENS, 3);
            stable_window = false;
            reasons.insert("hpa_replay_mismatch");
        }

        if input.deny_storm_present {
            self.bump_cell(CellKey::AL, 1);
            self.bump_cell(CellKey::PROBING_SENS, 2);
            stable_window = false;
            reasons.insert("hpa_deny_storm");
        }

        if input.timeouts_burst_present {
            self.bump_cell(CellKey::AL, 1);
            self.bump_cell(CellKey::RELIABILITY_SENS, 1);
            stable_window = false;
            reasons.insert("hpa_timeout_burst");
        }

        if input.unlock_present && stable_window {
            self.bump_cell(CellKey::RECOVERY_CONF, 3);
            self.decay_cell(CellKey::AL, 1);
            reasons.insert("hpa_recovery_unlock");
        }

        if stable_window {
            self.stable_window_counter = self.stable_window_counter.saturating_add(1);
            if self.stable_window_counter >= STABLE_DECAY_INTERVAL {
                self.decay_cell(CellKey::AL, 1);
                self.stable_window_counter = 0;
                reasons.insert("hpa_stable_decay");
            }
        } else {
            self.stable_window_counter = 0;
        }

        if input.calibrate_now {
            self.calibration_counter = CALIBRATION_INTERVAL;
        }

        self.calibration_counter = self.calibration_counter.saturating_add(1);
        self.maybe_calibrate(&mut reasons);

        self.derive_output(reasons, Some(self.backend.snapshot_digest()))
    }

    fn snapshot_digest(&self) -> [u8; 32] {
        let mut bytes = Vec::new();
        bytes.extend(self.backend.snapshot_digest());
        bytes.extend(self.step_count.to_le_bytes());
        digest_meta("UCF:MC:HPA", &bytes)
    }

    fn config_digest(&self) -> [u8; 32] {
        let mut bytes = Vec::new();
        bytes.extend(self.config.version.to_le_bytes());
        bytes.extend(self.config.seed.to_le_bytes());
        bytes.extend(self.config.max_neurons.to_le_bytes());
        bytes.extend(self.backend.config_digest());
        digest_meta("UCF:MC:HPA:CFG", &bytes)
    }
}

fn div_ceil(value: u16, divisor: u16) -> u16 {
    if value == 0 {
        0
    } else {
        value.div_ceil(divisor)
    }
}

fn clamp_offset(value: u16) -> u16 {
    value.min(OFFSET_MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_circuit() -> HpaCircuit<EmulatedMemristorBackend> {
        HpaCircuit::new_emulated(CircuitConfig::default(), 42)
    }

    #[test]
    fn determinism_matches_outputs_and_cells() {
        let mut circuit_a = build_circuit();
        let mut circuit_b = build_circuit();

        let input = HpaInput {
            integrity_state: IntegrityState::Fail,
            replay_mismatch_present: true,
            dlp_critical_present: false,
            receipt_invalid_present: true,
            deny_storm_present: false,
            timeouts_burst_present: true,
            unlock_present: false,
            stable_medium_window: false,
            calibrate_now: false,
        };

        let output_a = circuit_a.step(&input, 0);
        let output_b = circuit_b.step(&input, 0);

        assert_eq!(output_a, output_b);
        for key in CellKey::ALL.iter().copied() {
            assert_eq!(circuit_a.read_cell(key), circuit_b.read_cell(key));
        }
    }

    #[test]
    fn cells_remain_bounded() {
        let mut circuit = build_circuit();
        let input = HpaInput {
            integrity_state: IntegrityState::Fail,
            replay_mismatch_present: true,
            dlp_critical_present: true,
            receipt_invalid_present: true,
            deny_storm_present: true,
            timeouts_burst_present: true,
            unlock_present: false,
            stable_medium_window: false,
            calibrate_now: false,
        };

        for _ in 0..200 {
            circuit.step(&input, 0);
        }

        for key in CellKey::ALL.iter().copied() {
            assert!(circuit.read_cell(key) <= circuit.backend.max_value());
        }
    }

    #[test]
    fn calibration_triggers_after_interval() {
        let mut circuit = build_circuit();
        circuit.backend.write_cell(CellKey::AL, 20);
        let input = HpaInput {
            stable_medium_window: true,
            ..HpaInput::default()
        };

        let previous_digest = circuit.backend.snapshot_digest();
        let mut calibrated_digest = None;
        for _ in 0..CALIBRATION_INTERVAL {
            let output = circuit.step(&input, 0);
            if output
                .reason_codes
                .codes
                .iter()
                .any(|code| code == "RC.GV.HPA.CALIBRATED")
            {
                calibrated_digest = output.mem_snapshot_digest;
                break;
            }
        }

        let current_digest = circuit.backend.snapshot_digest();
        assert!(calibrated_digest.is_some());
        assert_ne!(previous_digest, current_digest);
    }

    #[test]
    fn config_digest_ignores_state_changes() {
        let mut circuit = build_circuit();
        let baseline_digest = circuit.config_digest();
        let input = HpaInput {
            integrity_state: IntegrityState::Fail,
            replay_mismatch_present: true,
            dlp_critical_present: true,
            receipt_invalid_present: true,
            deny_storm_present: true,
            timeouts_burst_present: true,
            unlock_present: false,
            stable_medium_window: false,
            calibrate_now: true,
        };

        for _ in 0..10 {
            circuit.step(&input, 0);
        }

        assert_eq!(baseline_digest, circuit.config_digest());
    }

    #[test]
    fn micro_is_no_less_conservative_than_rules_for_critical_inputs() {
        #[derive(Default)]
        struct HpaState {
            allostatic_load: u16,
            exfil_sensitivity: u16,
            integrity_sensitivity: u16,
            probing_sensitivity: u16,
            reliability_sensitivity: u16,
            recovery_confidence: u16,
            stable_window_counter: u16,
        }

        fn rules_tick(state: &mut HpaState, input: &HpaInput) -> HpaOutput {
            let mut reasons = ReasonSet::default();
            let mut stable_window = input.stable_medium_window;

            if matches!(input.integrity_state, IntegrityState::Fail) {
                state.allostatic_load = state.allostatic_load.saturating_add(5);
                state.integrity_sensitivity = state.integrity_sensitivity.saturating_add(5);
                state.recovery_confidence = 0;
                state.stable_window_counter = 0;
                stable_window = false;
                reasons.insert("hpa_integrity_fail");
            }

            if input.dlp_critical_present {
                state.allostatic_load = state.allostatic_load.saturating_add(2);
                state.exfil_sensitivity = state.exfil_sensitivity.saturating_add(3);
                state.stable_window_counter = 0;
                stable_window = false;
                reasons.insert("hpa_dlp_critical");
            }

            if input.receipt_invalid_present {
                state.allostatic_load = state.allostatic_load.saturating_add(2);
                state.integrity_sensitivity = state.integrity_sensitivity.saturating_add(3);
                state.stable_window_counter = 0;
                stable_window = false;
                reasons.insert("hpa_receipt_invalid");
            }

            if input.replay_mismatch_present {
                state.allostatic_load = state.allostatic_load.saturating_add(2);
                state.integrity_sensitivity = state.integrity_sensitivity.saturating_add(3);
                state.stable_window_counter = 0;
                stable_window = false;
                reasons.insert("hpa_replay_mismatch");
            }

            if input.deny_storm_present {
                state.allostatic_load = state.allostatic_load.saturating_add(1);
                state.probing_sensitivity = state.probing_sensitivity.saturating_add(2);
                state.stable_window_counter = 0;
                stable_window = false;
                reasons.insert("hpa_deny_storm");
            }

            if input.timeouts_burst_present {
                state.allostatic_load = state.allostatic_load.saturating_add(1);
                state.reliability_sensitivity = state.reliability_sensitivity.saturating_add(1);
                state.stable_window_counter = 0;
                stable_window = false;
                reasons.insert("hpa_timeout_burst");
            }

            if input.unlock_present && stable_window {
                state.recovery_confidence = state.recovery_confidence.saturating_add(3);
                state.allostatic_load = state.allostatic_load.saturating_sub(1);
                reasons.insert("hpa_recovery_unlock");
            }

            if stable_window {
                state.stable_window_counter = state.stable_window_counter.saturating_add(1);
                if state.stable_window_counter >= STABLE_DECAY_INTERVAL {
                    state.allostatic_load = state.allostatic_load.saturating_sub(1);
                    state.stable_window_counter = 0;
                    reasons.insert("hpa_stable_decay");
                }
            } else {
                state.stable_window_counter = 0;
            }

            let allostatic_load_class = match state.allostatic_load {
                v if v >= 70 => LevelClass::High,
                v if v >= 30 => LevelClass::Med,
                _ => LevelClass::Low,
            };

            let caution_offset = clamp_offset(
                div_ceil(state.allostatic_load, 10)
                    + div_ceil(state.integrity_sensitivity, 20)
                    + div_ceil(state.exfil_sensitivity, 20),
            );
            let novelty_offset = clamp_offset(
                div_ceil(state.allostatic_load, 10) + div_ceil(state.probing_sensitivity, 20),
            );
            let approval_offset = clamp_offset(
                div_ceil(state.allostatic_load, 10)
                    + div_ceil(state.integrity_sensitivity, 20)
                    + div_ceil(state.probing_sensitivity, 20),
            );
            let export_offset = clamp_offset(
                div_ceil(state.exfil_sensitivity, 10) + div_ceil(state.allostatic_load, 20),
            );
            let chain_offset = clamp_offset(
                div_ceil(state.reliability_sensitivity, 10)
                    + div_ceil(state.probing_sensitivity, 20)
                    + div_ceil(state.allostatic_load, 20),
            );

            let cooldown_multiplier_class = if state.allostatic_load >= 80 {
                3
            } else if state.allostatic_load >= 50 {
                2
            } else if state.allostatic_load >= 20 {
                1
            } else {
                0
            };

            HpaOutput {
                allostatic_load_class,
                baseline_caution_offset: caution_offset,
                baseline_novelty_dampening_offset: novelty_offset,
                baseline_approval_strictness_offset: approval_offset,
                baseline_export_strictness_offset: export_offset,
                baseline_chain_conservatism_offset: chain_offset,
                baseline_cooldown_multiplier_class: cooldown_multiplier_class,
                reason_codes: reasons,
                mem_snapshot_digest: None,
            }
        }

        fn severity(level: LevelClass) -> u8 {
            match level {
                LevelClass::Low => 0,
                LevelClass::Med => 1,
                LevelClass::High => 2,
            }
        }

        let critical_inputs = [
            HpaInput {
                integrity_state: IntegrityState::Fail,
                ..HpaInput::default()
            },
            HpaInput {
                dlp_critical_present: true,
                receipt_invalid_present: true,
                replay_mismatch_present: true,
                deny_storm_present: true,
                timeouts_burst_present: true,
                ..HpaInput::default()
            },
        ];

        for input in critical_inputs {
            let mut rules_state = HpaState::default();
            let rules_out = rules_tick(&mut rules_state, &input);
            let mut circuit = build_circuit();
            let micro_out = circuit.step(&input, 0);

            assert!(
                severity(micro_out.allostatic_load_class)
                    >= severity(rules_out.allostatic_load_class)
            );
            assert!(micro_out.baseline_caution_offset >= rules_out.baseline_caution_offset);
            assert!(
                micro_out.baseline_novelty_dampening_offset
                    >= rules_out.baseline_novelty_dampening_offset
            );
            assert!(
                micro_out.baseline_approval_strictness_offset
                    >= rules_out.baseline_approval_strictness_offset
            );
            assert!(
                micro_out.baseline_export_strictness_offset
                    >= rules_out.baseline_export_strictness_offset
            );
            assert!(
                micro_out.baseline_chain_conservatism_offset
                    >= rules_out.baseline_chain_conservatism_offset
            );
            assert!(
                micro_out.baseline_cooldown_multiplier_class
                    >= rules_out.baseline_cooldown_multiplier_class
            );
        }
    }
}
