#![forbid(unsafe_code)]

use blake3::Hasher;
use profiles::{
    apply_cbv_modifiers, apply_classification, apply_pev_modifiers, classify_signal_frame,
    decide_with_fallback, ControlDecision, FlappingPenaltyMode, OverlaySet, ProfileState,
    RegulationConfig,
};
use prost::Message;
use pvgs_client::{PvgsReader, PvgsWriter};
use rsv::RsvState;
use std::collections::VecDeque;
use std::path::PathBuf;
use ucf::v1::{
    ActiveProfile, ControlFrame, IntegrityStateClass, LevelClass, Overlays, ReasonCode,
    ToolClassMask,
};

const CONTROL_FRAME_DOMAIN: &str = "UCF:HASH:CONTROL_FRAME";

#[derive(Debug)]
pub enum EngineError {
    QueueSaturated,
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::QueueSaturated => {
                write!(f, "signal queue saturated; dropped oldest frame")
            }
        }
    }
}

impl std::error::Error for EngineError {}

#[derive(Debug, Default)]
struct AntiFlappingState {
    last_profile_change_ms: Option<u64>,
    last_overlay_change_ms: Option<u64>,
    switch_count_medium_window: u32,
    medium_window_started_ms: Option<u64>,
    current_profile: Option<ProfileState>,
    current_overlays: Option<OverlaySet>,
}

pub struct RegulationEngine {
    pub rsv: RsvState,
    config: RegulationConfig,
    pvgs_reader: Option<Box<dyn PvgsReader + Send + Sync>>,
    pvgs_writer: Option<Box<dyn PvgsWriter + Send>>,
    session_id: Option<String>,
    signal_queue: VecDeque<ucf::v1::SignalFrame>,
    anti_flapping_state: AntiFlappingState,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RegulationSnapshot {
    pub profile: ProfileState,
    pub overlays: OverlaySet,
    pub deescalation_lock: bool,
    pub control_frame_digest: Option<[u8; 32]>,
    pub rsv_summary: RsvSummary,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RsvSummary {
    pub integrity: IntegrityStateClass,
    pub threat: LevelClass,
    pub policy_pressure: LevelClass,
    pub arousal: LevelClass,
    pub stability: LevelClass,
}

impl Default for RegulationEngine {
    fn default() -> Self {
        let config_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("config");
        match RegulationConfig::load_from_dir(config_dir) {
            Ok(config) => RegulationEngine {
                rsv: RsvState::default(),
                config,
                pvgs_reader: None,
                pvgs_writer: None,
                session_id: None,
                signal_queue: VecDeque::new(),
                anti_flapping_state: AntiFlappingState::default(),
            },
            Err(_) => RegulationEngine {
                rsv: RsvState::default(),
                config: RegulationConfig::fallback(),
                pvgs_reader: None,
                pvgs_writer: None,
                session_id: None,
                signal_queue: VecDeque::new(),
                anti_flapping_state: AntiFlappingState::default(),
            },
        }
    }
}

impl RegulationEngine {
    pub fn new(config: RegulationConfig) -> Self {
        RegulationEngine {
            rsv: RsvState::default(),
            config,
            pvgs_reader: None,
            pvgs_writer: None,
            session_id: None,
            signal_queue: VecDeque::new(),
            anti_flapping_state: AntiFlappingState::default(),
        }
    }

    pub fn set_pvgs_reader(&mut self, reader: impl PvgsReader + 'static) {
        self.pvgs_reader = Some(Box::new(reader));
    }

    pub fn set_pvgs_writer(&mut self, writer: impl PvgsWriter + 'static) {
        self.pvgs_writer = Some(Box::new(writer));
    }

    pub fn set_session_id(&mut self, session_id: impl Into<String>) {
        self.session_id = Some(session_id.into());
    }

    pub fn snapshot(&mut self) -> RegulationSnapshot {
        let now_ms = self.rsv.last_seen_frame_ts_ms.unwrap_or(0);
        let decision = decide_with_fallback(&self.rsv, now_ms, &self.config);
        let decision = self.apply_forensic_override(decision);
        let decision = self.apply_anti_flapping(decision, now_ms);
        let (control_frame, _) = self.build_control_frame(decision.clone());
        let digest = control_frame
            .control_frame_digest
            .as_ref()
            .and_then(|bytes| bytes.as_slice().try_into().ok());

        RegulationSnapshot {
            profile: decision.profile,
            overlays: decision.overlays,
            deescalation_lock: decision.deescalation_lock,
            control_frame_digest: digest,
            rsv_summary: self.rsv_summary(),
        }
    }

    pub fn enqueue_signal_frame(&mut self, frame: ucf::v1::SignalFrame) -> Result<(), EngineError> {
        if self.signal_queue.len() >= self.config.engine_limits.max_queue_len {
            let _ = self.signal_queue.pop_front();
            eprintln!(
                "dropping oldest signal frame to honor max_queue_len={}",
                self.config.engine_limits.max_queue_len
            );
            self.signal_queue.push_back(frame);
            return Err(EngineError::QueueSaturated);
        }

        self.signal_queue.push_back(frame);
        Ok(())
    }

    pub fn queued_frames(&self) -> usize {
        self.signal_queue.len()
    }

    pub fn on_signal_frame(&mut self, frame: ucf::v1::SignalFrame, now_ms: u64) -> ControlFrame {
        let decision = self.decide_from_frame(frame, now_ms);
        self.apply_and_render(decision, now_ms)
    }

    pub fn on_tick(&mut self, now_ms: u64) -> ControlFrame {
        let decision = self.decide_on_missing(now_ms);
        self.apply_and_render(decision, now_ms)
    }

    pub fn tick(&mut self, now_ms: u64) -> ControlFrame {
        let mut control_frame: Option<ControlFrame> = None;
        let max_to_process = self
            .config
            .engine_limits
            .max_frames_per_tick
            .min(self.signal_queue.len());

        for _ in 0..max_to_process {
            if let Some(frame) = self.signal_queue.pop_front() {
                let decision = self.decide_from_frame(frame, now_ms);
                control_frame = Some(self.apply_and_render(decision, now_ms));
            }
        }

        if let Some(control_frame) = control_frame {
            return control_frame;
        }

        let decision = self.decide_on_missing(now_ms);
        self.apply_and_render(decision, now_ms)
    }

    pub fn reset_forensic(&mut self) {
        self.rsv.reset_forensic();
        self.anti_flapping_state.current_profile = None;
        self.anti_flapping_state.current_overlays = None;
    }

    fn decide_from_frame(
        &mut self,
        mut frame: ucf::v1::SignalFrame,
        now_ms: u64,
    ) -> ControlDecision {
        if frame.timestamp_ms.is_none() {
            frame.timestamp_ms = Some(now_ms);
        }

        let timestamp_ms = frame.timestamp_ms.unwrap_or(now_ms);
        let classified = classify_signal_frame(&frame, &self.config.thresholds);
        apply_classification(&mut self.rsv, &classified, timestamp_ms);
        self.rsv.update_unlock_state(&frame);

        decide_with_fallback(&self.rsv, now_ms, &self.config)
    }

    fn decide_on_missing(&mut self, now_ms: u64) -> ControlDecision {
        self.rsv.missing_frame_counter = self.rsv.missing_frame_counter.saturating_add(1);
        decide_with_fallback(&self.rsv, now_ms, &self.config)
    }

    fn apply_and_render(&mut self, decision: ControlDecision, now_ms: u64) -> ControlFrame {
        let decision = self.apply_forensic_override(decision);
        let decision = self.apply_anti_flapping(decision, now_ms);
        self.render_control_frame(decision)
    }

    fn apply_forensic_override(&mut self, mut decision: ControlDecision) -> ControlDecision {
        let integrity_fail = self.rsv.integrity == IntegrityStateClass::Fail;
        if integrity_fail {
            self.rsv.forensic_latched = true;
        }

        if self.rsv.forensic_latched {
            let unlock_ready = self.rsv.unlock_ready;
            decision.profile = if unlock_ready {
                ProfileState::M1Restricted
            } else {
                ProfileState::M3Forensic
            };
            decision.overlays = OverlaySet::all_enabled();
            decision.deescalation_lock = true;
            decision.cooldown_class = LevelClass::High;

            if unlock_ready {
                decision
                    .profile_reason_codes
                    .retain(|code| *code != ReasonCode::ReIntegrityFail);

                if !decision
                    .profile_reason_codes
                    .contains(&ReasonCode::ReIntegrityDegraded)
                {
                    decision
                        .profile_reason_codes
                        .push(ReasonCode::ReIntegrityDegraded);
                }

                if !decision
                    .profile_reason_codes
                    .contains(&ReasonCode::RcGvRecoveryUnlockGranted)
                {
                    decision
                        .profile_reason_codes
                        .push(ReasonCode::RcGvRecoveryUnlockGranted);
                }

                if !decision
                    .profile_reason_codes
                    .contains(&ReasonCode::RcRgProfileM1Restricted)
                {
                    decision
                        .profile_reason_codes
                        .push(ReasonCode::RcRgProfileM1Restricted);
                }
            } else if !decision
                .profile_reason_codes
                .contains(&ReasonCode::ReIntegrityFail)
            {
                decision
                    .profile_reason_codes
                    .push(ReasonCode::ReIntegrityFail);
            }

            if !decision
                .profile_reason_codes
                .contains(&ReasonCode::RcRxActionForensic)
            {
                decision
                    .profile_reason_codes
                    .push(ReasonCode::RcRxActionForensic);
            }
        }

        decision
    }

    fn apply_anti_flapping(
        &mut self,
        mut decision: ControlDecision,
        now_ms: u64,
    ) -> ControlDecision {
        self.reset_switch_window_if_needed(now_ms);

        let mut profile_changed = false;
        let mut overlay_changed = false;

        let current_profile = self
            .anti_flapping_state
            .current_profile
            .unwrap_or(decision.profile);

        if self.anti_flapping_state.current_profile.is_none() {
            self.anti_flapping_state.current_profile = Some(decision.profile);
            self.anti_flapping_state.last_profile_change_ms = Some(now_ms);
        } else if decision.profile != current_profile {
            let unlock_deescalation = self.rsv.forensic_latched && self.rsv.unlock_ready;
            let tightening = Self::is_tightening(decision.profile, current_profile);
            let within_cooldown = self.within_profile_cooldown(now_ms);

            if unlock_deescalation {
                profile_changed = true;
                self.anti_flapping_state.current_profile = Some(decision.profile);
                self.anti_flapping_state.last_profile_change_ms = Some(now_ms);
            } else if !tightening && within_cooldown {
                decision.profile = current_profile;
            } else {
                profile_changed = true;
                self.anti_flapping_state.current_profile = Some(decision.profile);
                self.anti_flapping_state.last_profile_change_ms = Some(now_ms);
            }
        }

        let current_overlays = self
            .anti_flapping_state
            .current_overlays
            .clone()
            .unwrap_or_else(|| decision.overlays.clone());

        if self.anti_flapping_state.current_overlays.is_none() {
            self.anti_flapping_state.current_overlays = Some(decision.overlays.clone());
            self.anti_flapping_state.last_overlay_change_ms = Some(now_ms);
        } else if decision.overlays != current_overlays {
            let mut overlays = current_overlays.clone();
            let overlay_cooldown_passed = self.overlay_cooldown_passed(now_ms);

            if decision.overlays.simulate_first && !current_overlays.simulate_first {
                overlay_changed = true;
                overlays.simulate_first = true;
            } else if !decision.overlays.simulate_first
                && current_overlays.simulate_first
                && overlay_cooldown_passed
            {
                overlay_changed = true;
                overlays.simulate_first = false;
            }

            if decision.overlays.export_lock && !current_overlays.export_lock {
                overlay_changed = true;
                overlays.export_lock = true;
            } else if !decision.overlays.export_lock
                && current_overlays.export_lock
                && overlay_cooldown_passed
            {
                overlay_changed = true;
                overlays.export_lock = false;
            }

            if decision.overlays.novelty_lock && !current_overlays.novelty_lock {
                overlay_changed = true;
                overlays.novelty_lock = true;
            } else if !decision.overlays.novelty_lock
                && current_overlays.novelty_lock
                && overlay_cooldown_passed
            {
                overlay_changed = true;
                overlays.novelty_lock = false;
            }

            if decision.overlays.chain_tightening && !current_overlays.chain_tightening {
                overlay_changed = true;
                overlays.chain_tightening = true;
            } else if !decision.overlays.chain_tightening
                && current_overlays.chain_tightening
                && overlay_cooldown_passed
            {
                overlay_changed = true;
                overlays.chain_tightening = false;
            }

            if overlay_changed {
                self.anti_flapping_state.current_overlays = Some(overlays.clone());
                self.anti_flapping_state.last_overlay_change_ms = Some(now_ms);
                decision.overlays = overlays;
            } else {
                decision.overlays = current_overlays;
            }
        }

        if profile_changed || overlay_changed {
            self.anti_flapping_state.switch_count_medium_window = self
                .anti_flapping_state
                .switch_count_medium_window
                .saturating_add((profile_changed as u32) + (overlay_changed as u32));
        }

        if self.anti_flapping_state.switch_count_medium_window
            > self.config.anti_flapping.max_switches_per_medium_window
        {
            if matches!(
                self.config.anti_flapping.flapping_penalty_mode,
                FlappingPenaltyMode::ForceM1Lock
            ) {
                decision.profile = ProfileState::M1Restricted;
                decision.overlays = OverlaySet::all_enabled();
                decision.deescalation_lock = true;
            }

            self.anti_flapping_state.current_profile = Some(decision.profile);
            self.anti_flapping_state.current_overlays = Some(decision.overlays.clone());
            self.anti_flapping_state.last_profile_change_ms = Some(now_ms);
            self.anti_flapping_state.last_overlay_change_ms = Some(now_ms);
        }

        decision
    }

    fn reset_switch_window_if_needed(&mut self, now_ms: u64) {
        let start = self
            .anti_flapping_state
            .medium_window_started_ms
            .unwrap_or(now_ms);

        if now_ms.saturating_sub(start) >= self.config.windowing.medium.max_age_ms {
            self.anti_flapping_state.switch_count_medium_window = 0;
            self.anti_flapping_state.medium_window_started_ms = Some(now_ms);
        } else if self.anti_flapping_state.medium_window_started_ms.is_none() {
            self.anti_flapping_state.medium_window_started_ms = Some(start);
        }
    }

    fn within_profile_cooldown(&self, now_ms: u64) -> bool {
        self.anti_flapping_state
            .last_profile_change_ms
            .map(|ts| {
                now_ms.saturating_sub(ts) < self.config.anti_flapping.min_ms_between_profile_changes
            })
            .unwrap_or(false)
    }

    fn overlay_cooldown_passed(&self, now_ms: u64) -> bool {
        self.anti_flapping_state
            .last_overlay_change_ms
            .map(|ts| {
                now_ms.saturating_sub(ts)
                    >= self.config.anti_flapping.min_ms_between_overlay_changes
            })
            .unwrap_or(true)
    }

    fn is_tightening(next: ProfileState, current: ProfileState) -> bool {
        Self::profile_rank(next) >= Self::profile_rank(current)
    }

    fn profile_rank(profile: ProfileState) -> u8 {
        match profile {
            ProfileState::M0Research => 0,
            ProfileState::M1Restricted => 1,
            ProfileState::M2Quarantine => 2,
            ProfileState::M3Forensic => 3,
        }
    }

    fn render_control_frame(&mut self, decision: ControlDecision) -> ControlFrame {
        let (control_frame, digest_bytes) = self.build_control_frame(decision);

        if let Some(writer) = self.pvgs_writer.as_mut() {
            let _ = writer.commit_control_frame_evidence(
                self.session_id.as_deref().unwrap_or("default-session"),
                digest_bytes,
            );
        }

        control_frame
    }

    fn build_control_frame(&self, mut decision: ControlDecision) -> (ControlFrame, [u8; 32]) {
        let cbv_digest = self
            .pvgs_reader
            .as_ref()
            .and_then(|reader| reader.get_latest_cbv_digest());
        let cbv = self
            .pvgs_reader
            .as_ref()
            .and_then(|reader| reader.get_latest_cbv());
        let pev_digest = self
            .pvgs_reader
            .as_ref()
            .and_then(|reader| reader.get_latest_pev_digest());
        let pev = self
            .pvgs_reader
            .as_ref()
            .and_then(|reader| reader.get_latest_pev());

        if cbv_digest.is_some()
            && self.config.character_baselines.cbv_influence_enabled
            && self.config.character_baselines.novelty_lock_on_cbv
        {
            decision.overlays.novelty_lock = true;
        }

        if cbv_digest.is_some() && self.config.character_baselines.cbv_influence_enabled {
            decision.approval_mode = self.config.character_baselines.strict_approval_mode.clone();
        }

        let decision = apply_cbv_modifiers(decision, cbv);
        let decision = apply_pev_modifiers(decision, pev);

        let overlays = Overlays {
            simulate_first: decision.overlays.simulate_first,
            export_lock: decision.overlays.export_lock,
            novelty_lock: decision.overlays.novelty_lock,
            chain_tightening: decision.overlays.chain_tightening,
        };

        let profile = ActiveProfile {
            profile: decision.profile.as_str().to_string(),
        };

        let toolclass_mask_config = self.toolclass_mask_for(&decision);

        let mut profile_reason_codes: Vec<i32> = decision
            .profile_reason_codes
            .iter()
            .map(|code| *code as i32)
            .collect();
        profile_reason_codes.sort();

        let mut control_frame = ControlFrame {
            active_profile: Some(profile),
            overlays: Some(overlays),
            toolclass_mask: Some(toolclass_mask_config),
            profile_reason_codes,
            control_frame_digest: None,
            character_epoch_digest: cbv_digest.map(|digest| digest.to_vec()),
            policy_ecology_digest: pev_digest.map(|digest| digest.to_vec()),
            approval_mode: Some(decision.approval_mode.clone()),
            deescalation_lock: decision.deescalation_lock.then_some(true),
            cooldown_class: Some(decision.cooldown_class as i32),
        };

        let mut buf = Vec::new();
        control_frame.encode(&mut buf).unwrap();

        let mut hasher = Hasher::new_derive_key(CONTROL_FRAME_DOMAIN);
        hasher.update(&buf);
        let digest = hasher.finalize();
        let digest_bytes: [u8; 32] = *digest.as_bytes();

        control_frame.control_frame_digest = Some(digest_bytes.to_vec());

        (control_frame, digest_bytes)
    }

    fn rsv_summary(&self) -> RsvSummary {
        RsvSummary {
            integrity: self.rsv.integrity,
            threat: self.rsv.threat,
            policy_pressure: self.rsv.policy_pressure,
            arousal: self.rsv.arousal,
            stability: self.rsv.stability,
        }
    }

    fn toolclass_mask_for(&self, decision: &ControlDecision) -> ToolClassMask {
        if self.rsv.forensic_latched {
            return ToolClassMask {
                read: true,
                write: false,
                execute: false,
                transform: true,
                export: false,
            };
        }

        let mut mask_cfg = self
            .config
            .profiles
            .get(decision.profile)
            .toolclass_mask
            .clone();

        if decision.overlays.simulate_first {
            if let Some(overlay_mask) = &self.config.overlays.simulate_first.toolclass_mask {
                mask_cfg = mask_cfg.merge_overlay(overlay_mask);
            }
        }

        if decision.overlays.export_lock {
            if let Some(overlay_mask) = &self.config.overlays.export_lock.toolclass_mask {
                mask_cfg = mask_cfg.merge_overlay(overlay_mask);
            }
        }

        if decision.overlays.novelty_lock {
            if let Some(overlay_mask) = &self.config.overlays.novelty_lock.toolclass_mask {
                mask_cfg = mask_cfg.merge_overlay(overlay_mask);
            }
        }

        mask_cfg.to_tool_class_mask()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use profiles::{OverlayEnableRule, OverlaySet, ProfileState, RegulationConfig, RuleCondition};
    use pvgs_client::{MockPvgsReader, PvgsError};
    use std::sync::{Arc, Mutex};
    use ucf::v1::{
        CharacterBaselineVector, ExecStats, IntegrityStateClass, LevelClass, PolicyStats,
        ReasonCode, ReceiptStats, SignalFrame, WindowKind,
    };

    type EvidenceLog = Arc<Mutex<Vec<(String, [u8; 32])>>>;

    #[derive(Clone, Default)]
    struct RecordingWriter {
        calls: EvidenceLog,
    }

    impl PvgsWriter for RecordingWriter {
        fn commit_control_frame_evidence(
            &mut self,
            session_id: &str,
            control_frame_digest: [u8; 32],
        ) -> Result<(), PvgsError> {
            self.calls
                .lock()
                .unwrap()
                .push((session_id.to_string(), control_frame_digest));
            Ok(())
        }
    }

    fn base_frame() -> SignalFrame {
        SignalFrame {
            window_kind: WindowKind::Short as i32,
            window_index: Some(1),
            timestamp_ms: Some(1),
            policy_stats: Some(PolicyStats {
                deny_count: 0,
                allow_count: 10,
                top_reason_codes: Vec::new(),
            }),
            exec_stats: Some(ExecStats {
                timeout_count: 0,
                top_reason_codes: Vec::new(),
            }),
            integrity_state: IntegrityStateClass::Ok as i32,
            top_reason_codes: Vec::new(),
            signal_frame_digest: None,
            receipt_stats: Some(ReceiptStats {
                receipt_missing_count: 0,
                receipt_invalid_count: 0,
            }),
            reason_codes: Vec::new(),
        }
    }

    fn medium_unlock_frame(
        ts: u64,
        integrity: IntegrityStateClass,
        mut reason_codes: Vec<i32>,
    ) -> SignalFrame {
        reason_codes.push(ReasonCode::RcGvRecoveryUnlockGranted as i32);

        SignalFrame {
            window_kind: WindowKind::Medium as i32,
            window_index: Some(ts),
            timestamp_ms: Some(ts),
            integrity_state: integrity as i32,
            reason_codes,
            ..base_frame()
        }
    }

    fn replay_mismatch_frame(ts: u64) -> SignalFrame {
        let mut frame = base_frame();
        frame.timestamp_ms = Some(ts);
        frame.top_reason_codes = vec![ReasonCode::RcReReplayMismatch as i32];
        frame
    }

    fn cbv_digest(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    fn pev_digest(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    fn pev_vector() -> ucf::v1::PolicyEcologyVector {
        ucf::v1::PolicyEcologyVector {
            conservatism_bias: 0,
            novelty_penalty_bias: 0,
            manipulation_aversion_bias: 0,
            reversibility_bias: 0,
        }
    }

    #[test]
    fn digest_is_deterministic() {
        let mut engine = RegulationEngine::default();
        let frame = base_frame();
        let control_a = engine.on_signal_frame(frame.clone(), 1);
        let control_b = engine.on_signal_frame(frame, 1);
        assert_eq!(
            control_a.control_frame_digest,
            control_b.control_frame_digest
        );
    }

    #[test]
    fn cbv_digest_embeds_and_changes_control_frame_hash() {
        let frame = base_frame();

        let mut engine_with_cbv_a = RegulationEngine::default();
        engine_with_cbv_a.set_pvgs_reader(MockPvgsReader::with_cbv(cbv_digest(1)));
        let control_a = engine_with_cbv_a.on_signal_frame(frame.clone(), 1);

        let mut engine_with_cbv_b = RegulationEngine::default();
        engine_with_cbv_b.set_pvgs_reader(MockPvgsReader::with_cbv(cbv_digest(2)));
        let control_b = engine_with_cbv_b.on_signal_frame(frame, 1);

        assert_eq!(control_a.character_epoch_digest, Some(vec![1u8; 32]));
        assert_eq!(control_b.character_epoch_digest, Some(vec![2u8; 32]));
        assert_ne!(
            control_a.control_frame_digest,
            control_b.control_frame_digest
        );
    }

    #[test]
    fn pev_digest_embeds_and_changes_control_frame_hash() {
        let frame = base_frame();

        let mut reader_a = MockPvgsReader::with_pev_digest(pev_digest(8));
        reader_a.pev = Some(pev_vector());
        let mut engine_with_pev_a = RegulationEngine::default();
        engine_with_pev_a.set_pvgs_reader(reader_a);
        let control_a = engine_with_pev_a.on_signal_frame(frame.clone(), 1);

        let mut reader_b = MockPvgsReader::with_pev_digest(pev_digest(9));
        reader_b.pev = Some(pev_vector());
        let mut engine_with_pev_b = RegulationEngine::default();
        engine_with_pev_b.set_pvgs_reader(reader_b);
        let control_b = engine_with_pev_b.on_signal_frame(frame, 1);

        assert_eq!(control_a.policy_ecology_digest, Some(vec![8u8; 32]));
        assert_eq!(control_b.policy_ecology_digest, Some(vec![9u8; 32]));
        assert_ne!(
            control_a.control_frame_digest,
            control_b.control_frame_digest
        );
    }

    #[test]
    fn cbv_absence_is_stable() {
        let frame = base_frame();

        let mut engine_a = RegulationEngine::default();
        let control_a = engine_a.on_signal_frame(frame.clone(), 1);

        let mut engine_b = RegulationEngine::default();
        let control_b = engine_b.on_signal_frame(frame, 1);

        assert!(control_a.character_epoch_digest.is_none());
        assert_eq!(
            control_a.control_frame_digest,
            control_b.control_frame_digest
        );
    }

    #[test]
    fn pev_absence_is_stable() {
        let frame = base_frame();

        let mut engine_a = RegulationEngine::default();
        let control_a = engine_a.on_signal_frame(frame.clone(), 1);

        let mut engine_b = RegulationEngine::default();
        let control_b = engine_b.on_signal_frame(frame, 1);

        assert!(control_a.policy_ecology_digest.is_none());
        assert_eq!(
            control_a.control_frame_digest,
            control_b.control_frame_digest
        );
    }

    #[test]
    fn cbv_presence_triggers_strict_mode_when_enabled() {
        let mut config = RegulationConfig::fallback();
        config.character_baselines.cbv_influence_enabled = true;
        config.character_baselines.strict_approval_mode = "STRICT".to_string();
        config.character_baselines.novelty_lock_on_cbv = true;

        let mut engine = RegulationEngine::new(config);
        engine.set_pvgs_reader(MockPvgsReader::with_cbv(cbv_digest(3)));

        let control = engine.on_signal_frame(base_frame(), 1);
        assert_eq!(control.approval_mode.as_deref(), Some("STRICT"));
        assert!(control.overlays.unwrap().novelty_lock);
    }

    #[test]
    fn pev_conservatism_bias_enforces_strict_and_lock() {
        let reader = MockPvgsReader::with_pev_vector(ucf::v1::PolicyEcologyVector {
            conservatism_bias: 1,
            novelty_penalty_bias: 0,
            manipulation_aversion_bias: 0,
            reversibility_bias: 0,
        });

        let mut engine = RegulationEngine::default();
        engine.set_pvgs_reader(reader);

        let control = engine.on_signal_frame(base_frame(), 1);
        assert_eq!(control.approval_mode.as_deref(), Some("STRICT"));
        assert_eq!(control.deescalation_lock, Some(true));
    }

    #[test]
    fn pev_novelty_bias_enables_lock() {
        let reader = MockPvgsReader::with_pev_vector(ucf::v1::PolicyEcologyVector {
            conservatism_bias: 0,
            novelty_penalty_bias: 1,
            manipulation_aversion_bias: 0,
            reversibility_bias: 0,
        });

        let mut engine = RegulationEngine::default();
        engine.set_pvgs_reader(reader);

        let control = engine.on_signal_frame(base_frame(), 1);
        assert!(control.overlays.unwrap().novelty_lock);
    }

    #[test]
    fn pev_manipulation_bias_enables_export_lock() {
        let reader = MockPvgsReader::with_pev_vector(ucf::v1::PolicyEcologyVector {
            conservatism_bias: 0,
            novelty_penalty_bias: 0,
            manipulation_aversion_bias: 1,
            reversibility_bias: 0,
        });

        let mut engine = RegulationEngine::default();
        engine.set_pvgs_reader(reader);

        let control = engine.on_signal_frame(base_frame(), 1);
        assert!(control.overlays.unwrap().export_lock);
    }

    #[test]
    fn pev_reversibility_bias_prefers_simulation() {
        let reader = MockPvgsReader::with_pev_vector(ucf::v1::PolicyEcologyVector {
            conservatism_bias: 0,
            novelty_penalty_bias: 0,
            manipulation_aversion_bias: 0,
            reversibility_bias: 1,
        });

        let mut engine = RegulationEngine::default();
        engine.set_pvgs_reader(reader);

        let control = engine.on_signal_frame(base_frame(), 1);
        let overlays = control.overlays.unwrap();
        assert!(overlays.simulate_first);
        assert!(overlays.chain_tightening);
    }

    #[test]
    fn cbv_digest_preserves_determinism_with_same_inputs() {
        let mut config = RegulationConfig::fallback();
        config.character_baselines.cbv_influence_enabled = true;

        let mut reader = MockPvgsReader::with_cbv(cbv_digest(4));
        reader.cbv = Some(CharacterBaselineVector {
            baseline_caution_offset: 2,
            baseline_novelty_dampening_offset: 2,
            baseline_approval_strictness_offset: 1,
            baseline_export_strictness_offset: 1,
            baseline_chain_conservatism_offset: 2,
            baseline_cooldown_multiplier_class: 2,
        });

        let mut engine_a = RegulationEngine::new(config.clone());
        engine_a.set_pvgs_reader(reader.clone());
        let control_a = engine_a.on_signal_frame(base_frame(), 1);

        let mut engine_b = RegulationEngine::new(config);
        engine_b.set_pvgs_reader(reader);
        let control_b = engine_b.on_signal_frame(base_frame(), 1);

        assert_eq!(
            control_a.control_frame_digest,
            control_b.control_frame_digest
        );
    }

    #[test]
    fn pev_digest_preserves_determinism_with_same_inputs() {
        let mut reader = MockPvgsReader::with_pev_digest(pev_digest(7));
        reader.pev = Some(ucf::v1::PolicyEcologyVector {
            conservatism_bias: 1,
            novelty_penalty_bias: 1,
            manipulation_aversion_bias: 1,
            reversibility_bias: 1,
        });

        let mut engine_a = RegulationEngine::default();
        engine_a.set_pvgs_reader(reader.clone());
        let control_a = engine_a.on_signal_frame(base_frame(), 1);

        let mut engine_b = RegulationEngine::default();
        engine_b.set_pvgs_reader(reader);
        let control_b = engine_b.on_signal_frame(base_frame(), 1);

        assert_eq!(control_a.policy_ecology_digest, Some(vec![7u8; 32]));
        assert_eq!(
            control_a.control_frame_digest,
            control_b.control_frame_digest
        );
    }

    #[test]
    fn evidence_writer_receives_control_frame_digest() {
        let mut engine = RegulationEngine::default();
        engine.set_session_id("session-123");
        engine.set_pvgs_reader(MockPvgsReader::with_cbv(cbv_digest(5)));

        let calls = Arc::new(Mutex::new(Vec::new()));
        let writer = RecordingWriter {
            calls: Arc::clone(&calls),
        };
        engine.set_pvgs_writer(writer);

        let control = engine.on_signal_frame(base_frame(), 1);

        let recorded = calls.lock().unwrap();
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].0, "session-123");
        assert_eq!(
            recorded[0].1.to_vec(),
            control.control_frame_digest.unwrap()
        );
    }

    #[test]
    fn config_driven_profile_switches() {
        let mut engine = RegulationEngine::default();
        let mut frame = base_frame();
        frame.receipt_stats = Some(ReceiptStats {
            receipt_missing_count: 0,
            receipt_invalid_count: 2,
        });

        let control = engine.on_signal_frame(frame, 1);
        assert_eq!(control.active_profile.unwrap().profile, "M1_RESTRICTED");
        let overlays = control.overlays.unwrap();
        assert!(overlays.simulate_first && overlays.export_lock && overlays.novelty_lock);
        assert!(control
            .profile_reason_codes
            .contains(&(ReasonCode::RcGeExecDispatchBlocked as i32)));
    }

    #[test]
    fn integrity_fail_triggers_forensic_profile() {
        let mut engine = RegulationEngine::default();
        let mut frame = base_frame();
        frame.integrity_state = IntegrityStateClass::Fail as i32;
        let control = engine.on_signal_frame(frame, 1);

        let mask = control.toolclass_mask.unwrap();
        assert!(!mask.export && !mask.write && !mask.execute);
        assert!(mask.read && mask.transform);
        assert_eq!(control.active_profile.unwrap().profile, "M3_FORENSIC");
        let overlays = control.overlays.unwrap();
        assert!(overlays.simulate_first);
        assert!(overlays.export_lock);
        assert!(overlays.novelty_lock);
        assert!(overlays.chain_tightening);
        assert_eq!(
            control.profile_reason_codes,
            vec![
                ReasonCode::ReIntegrityFail as i32,
                ReasonCode::RcRxActionForensic as i32
            ]
        );
        assert_eq!(control.deescalation_lock, Some(true));
        assert_eq!(control.cooldown_class, Some(LevelClass::High as i32));
    }

    #[test]
    fn integrity_fail_without_unlock_remains_forensic() {
        let mut engine = RegulationEngine::default();
        let mut frame = base_frame();
        frame.window_kind = WindowKind::Medium as i32;
        frame.integrity_state = IntegrityStateClass::Fail as i32;
        frame.reason_codes = vec![ReasonCode::ReIntegrityFail as i32];

        let control_first = engine.on_signal_frame(frame.clone(), 1);
        assert_eq!(control_first.active_profile.unwrap().profile, "M3_FORENSIC");

        let control_second = engine.on_signal_frame(frame, 2);
        assert_eq!(
            control_second.active_profile.unwrap().profile,
            "M3_FORENSIC"
        );
    }

    #[test]
    fn single_unlock_window_keeps_forensic_profile() {
        let mut engine = RegulationEngine::default();
        let frame = medium_unlock_frame(
            1,
            IntegrityStateClass::Fail,
            vec![ReasonCode::ReIntegrityFail as i32],
        );

        let control = engine.on_signal_frame(frame, 1);
        assert_eq!(control.active_profile.unwrap().profile, "M3_FORENSIC");
    }

    #[test]
    fn stable_unlock_windows_allow_restricted_recovery() {
        let mut engine = RegulationEngine::default();
        let first = medium_unlock_frame(
            1,
            IntegrityStateClass::Fail,
            vec![ReasonCode::ReIntegrityFail as i32],
        );
        let _ = engine.on_signal_frame(first, 1);

        let second = medium_unlock_frame(
            2,
            IntegrityStateClass::Degraded,
            vec![ReasonCode::ReIntegrityDegraded as i32],
        );
        let control = engine.on_signal_frame(second, 2);

        assert_eq!(control.active_profile.unwrap().profile, "M1_RESTRICTED");
        let overlays = control.overlays.unwrap();
        assert!(overlays.simulate_first);
        assert!(overlays.export_lock);
        assert!(overlays.novelty_lock);
        assert!(overlays.chain_tightening);

        let mask = control.toolclass_mask.unwrap();
        assert!(mask.read && mask.transform);
        assert!(!mask.export && !mask.write && !mask.execute);

        assert!(control
            .profile_reason_codes
            .contains(&(ReasonCode::RcGvRecoveryUnlockGranted as i32)));
        assert!(control
            .profile_reason_codes
            .contains(&(ReasonCode::RcRgProfileM1Restricted as i32)));
        assert!(control
            .profile_reason_codes
            .contains(&(ReasonCode::ReIntegrityDegraded as i32)));
        assert!(control
            .profile_reason_codes
            .contains(&(ReasonCode::RcRxActionForensic as i32)));
        assert!(!control
            .profile_reason_codes
            .contains(&(ReasonCode::ReIntegrityFail as i32)));
        assert_eq!(control.deescalation_lock, Some(true));
    }

    #[test]
    fn integrity_fail_latches_until_reset() {
        let mut engine = RegulationEngine::default();
        let mut frame = base_frame();
        frame.integrity_state = IntegrityStateClass::Fail as i32;
        let _ = engine.on_signal_frame(frame, 1);

        let mut ok_frame = base_frame();
        ok_frame.integrity_state = IntegrityStateClass::Ok as i32;
        let control = engine.on_signal_frame(ok_frame, 2);
        assert_eq!(control.active_profile.unwrap().profile, "M3_FORENSIC");

        engine.reset_forensic();

        let control_after_reset = engine.on_signal_frame(base_frame(), 3);
        assert_ne!(
            control_after_reset.active_profile.unwrap().profile,
            "M3_FORENSIC"
        );
    }

    #[test]
    fn forensic_override_is_deterministic() {
        let mut frame = base_frame();
        frame.integrity_state = IntegrityStateClass::Fail as i32;

        let mut engine_a = RegulationEngine::default();
        let mut engine_b = RegulationEngine::default();

        let control_a = engine_a.on_signal_frame(frame.clone(), 1);
        let control_b = engine_b.on_signal_frame(frame, 1);

        assert_eq!(
            control_a.control_frame_digest,
            control_b.control_frame_digest
        );
    }

    #[test]
    fn replay_mismatch_tightens_profiles_and_overlays() {
        let mut engine = RegulationEngine::default();
        let mut frame = base_frame();
        frame.top_reason_codes = vec![ReasonCode::RcReReplayMismatch as i32];

        let control = engine.on_signal_frame(frame, 1);
        let overlays = control.overlays.unwrap();

        assert_eq!(control.active_profile.unwrap().profile, "M1_RESTRICTED");
        assert!(overlays.simulate_first);
        assert!(overlays.export_lock);
        assert!(overlays.novelty_lock);
        assert!(overlays.chain_tightening);
        assert!(control
            .profile_reason_codes
            .contains(&(ReasonCode::RcReReplayMismatch as i32)));
        assert_eq!(control.deescalation_lock, Some(true));
    }

    #[test]
    fn degraded_integrity_with_replay_mismatch_escalates_profile() {
        let mut engine = RegulationEngine::default();
        let mut frame = base_frame();
        frame.integrity_state = IntegrityStateClass::Degraded as i32;
        frame.top_reason_codes = vec![ReasonCode::RcReReplayMismatch as i32];

        let control = engine.on_signal_frame(frame, 1);

        assert_eq!(control.active_profile.unwrap().profile, "M2_QUARANTINE");
        assert!(control
            .profile_reason_codes
            .contains(&(ReasonCode::RcReReplayMismatch as i32)));
        assert!(control
            .profile_reason_codes
            .contains(&(ReasonCode::ReIntegrityDegraded as i32)));
    }

    #[test]
    fn missing_frame_triggers_restriction() {
        let mut engine = RegulationEngine::default();
        let frame = base_frame();
        let _ = engine.on_signal_frame(frame, 0);
        let control = engine.on_tick(60_000);
        assert_eq!(control.active_profile.unwrap().profile, "M1_RESTRICTED");
        assert!(control.overlays.unwrap().export_lock);
        assert!(control
            .profile_reason_codes
            .contains(&(ReasonCode::ReIntegrityDegraded as i32)));
    }

    #[test]
    fn bounded_tick_processing_respects_limit() {
        let mut engine = RegulationEngine::default();
        engine.config.engine_limits.max_frames_per_tick = 3;
        engine.config.engine_limits.max_queue_len = 64;

        for i in 0..20u64 {
            let mut frame = base_frame();
            frame.timestamp_ms = Some(i);
            engine.enqueue_signal_frame(frame).unwrap();
        }

        let _ = engine.tick(100);
        assert_eq!(engine.queued_frames(), 17);
    }

    #[test]
    fn tighten_is_allowed_during_cooldown() {
        let mut engine = RegulationEngine::default();
        engine.config.anti_flapping.min_ms_between_profile_changes = 10_000;

        let _ = engine.on_signal_frame(base_frame(), 0);
        let control = engine.on_signal_frame(replay_mismatch_frame(1), 1);
        assert_eq!(control.active_profile.unwrap().profile, "M1_RESTRICTED");

        let control = engine.on_signal_frame(base_frame(), 2);
        assert_eq!(control.active_profile.unwrap().profile, "M1_RESTRICTED");
    }

    #[test]
    fn overlays_only_relax_after_cooldown() {
        let mut engine = RegulationEngine::default();
        engine.config.update_tables.overlay_enable.clear();
        engine.config.anti_flapping.min_ms_between_overlay_changes = 10_000;

        let control = engine.on_signal_frame(base_frame(), 0);
        assert!(!control.overlays.unwrap().simulate_first);

        engine
            .config
            .update_tables
            .overlay_enable
            .push(OverlayEnableRule {
                name: "enable_all".to_string(),
                conditions: RuleCondition {
                    any: Some(true),
                    ..Default::default()
                },
                overlays: OverlaySet {
                    simulate_first: true,
                    export_lock: true,
                    novelty_lock: true,
                    chain_tightening: true,
                },
            });

        let control = engine.on_signal_frame(base_frame(), 1);
        let overlays = control.overlays.unwrap();
        assert!(overlays.simulate_first && overlays.export_lock && overlays.novelty_lock);

        engine.config.update_tables.overlay_enable.clear();
        let control = engine.on_signal_frame(base_frame(), 2);
        assert!(control.overlays.unwrap().simulate_first);
    }

    #[test]
    fn flapping_penalty_forces_restriction() {
        let mut engine = RegulationEngine::default();
        engine.config.anti_flapping.max_switches_per_medium_window = 1;
        engine.config.anti_flapping.min_ms_between_profile_changes = 0;
        engine.config.anti_flapping.min_ms_between_overlay_changes = 0;
        engine.config.update_tables.overlay_enable.clear();

        let _ = engine.on_signal_frame(base_frame(), 0);
        let control = engine.on_signal_frame(replay_mismatch_frame(1), 1);
        assert_eq!(control.active_profile.unwrap().profile, "M1_RESTRICTED");

        let control = engine.on_signal_frame(base_frame(), 2);
        let overlays = control.overlays.unwrap();
        assert_eq!(control.active_profile.unwrap().profile, "M1_RESTRICTED");
        assert!(overlays.simulate_first && overlays.export_lock && overlays.novelty_lock);
        assert_eq!(control.deescalation_lock, Some(true));
    }

    #[test]
    fn deterministic_control_frames_from_same_sequence() {
        let mut engine_a = RegulationEngine::default();
        let mut engine_b = RegulationEngine::default();

        for ts in 0..5u64 {
            let mut frame = base_frame();
            frame.timestamp_ms = Some(ts);
            engine_a.enqueue_signal_frame(frame.clone()).unwrap();
            engine_b.enqueue_signal_frame(frame).unwrap();
        }

        let mut digests_a = Vec::new();
        let mut digests_b = Vec::new();

        for now in [10u64, 20u64] {
            digests_a.push(engine_a.tick(now).control_frame_digest.unwrap());
            digests_b.push(engine_b.tick(now).control_frame_digest.unwrap());
        }

        assert_eq!(digests_a, digests_b);
    }

    #[test]
    fn overlay_masks_are_applied() {
        let engine = RegulationEngine::default();
        let decision = ControlDecision {
            profile: ProfileState::M0Research,
            overlays: OverlaySet {
                simulate_first: true,
                export_lock: false,
                novelty_lock: false,
                chain_tightening: false,
            },
            deescalation_lock: false,
            missing_frame_override: false,
            profile_reason_codes: vec![ReasonCode::ReIntegrityDegraded],
            approval_mode: "monitor".to_string(),
            cooldown_class: LevelClass::Low,
        };

        let mask = engine.toolclass_mask_for(&decision);
        assert!(!mask.export);
        assert!(mask.read);
    }

    #[test]
    fn fallback_config_is_conservative() {
        let mut engine = RegulationEngine::new(RegulationConfig::fallback());
        let control = engine.on_signal_frame(base_frame(), 1);
        let mask = control.toolclass_mask.unwrap();
        assert_eq!(control.active_profile.unwrap().profile, "M1_RESTRICTED");
        assert!(!mask.export && !mask.write && !mask.execute);
    }

    #[test]
    fn reason_codes_are_sorted_before_digest() {
        let mut engine = RegulationEngine::default();
        let decision = ControlDecision {
            profile: ProfileState::M1Restricted,
            overlays: OverlaySet {
                simulate_first: true,
                export_lock: true,
                novelty_lock: true,
                chain_tightening: false,
            },
            deescalation_lock: true,
            missing_frame_override: false,
            profile_reason_codes: vec![
                ReasonCode::RcThIntegrityCompromise,
                ReasonCode::RcGeExecDispatchBlocked,
            ],
            approval_mode: "restricted".to_string(),
            cooldown_class: LevelClass::High,
        };

        let control_frame = engine.render_control_frame(decision);
        assert_eq!(
            control_frame.profile_reason_codes,
            vec![
                ReasonCode::RcGeExecDispatchBlocked as i32,
                ReasonCode::RcThIntegrityCompromise as i32,
            ]
        );
    }
}
