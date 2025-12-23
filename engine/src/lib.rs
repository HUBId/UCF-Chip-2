#![forbid(unsafe_code)]

use dbm_13_hypothalamus::ControlDecision as HypoDecision;
use dbm_bus::BrainBus;
use dbm_core::{
    DwmMode, EvidenceRef, IntegrityState, LevelClass as BrainLevel, OverlaySet as BrainOverlay,
    ProfileState as BrainProfile, ThreatVector,
};
use dbm_hpa::{Hpa, HpaOutput};
use profiles::{
    apply_classification, classify_signal_frame, ControlDecision, FlappingPenaltyMode, OverlaySet,
    ProfileState, RegulationConfig,
};
use pvgs_client::{PvgsReader, PvgsWriter};
use rsv::RsvState;
use std::collections::VecDeque;
use std::path::PathBuf;
#[cfg(test)]
use std::time::SystemTime;
use ucf::v1::{ControlFrame, IntegrityStateClass, LevelClass, ReasonCode, ToolClassMask};

mod proto_bridge;
use proto_bridge::{
    brain_input_from_signal_frame, control_frame_from_brain_output, BaselineContext,
    ControlFrameContext,
};

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
pub(crate) struct AntiFlappingState {
    last_profile_change_ms: Option<u64>,
    last_overlay_change_ms: Option<u64>,
    switch_count_medium_window: u32,
    medium_window_started_ms: Option<u64>,
    current_profile: Option<ProfileState>,
    current_overlays: Option<OverlaySet>,
}

#[derive(Debug, Default)]
pub(crate) struct WindowCounters {
    short_receipt_invalid_count: u32,
    short_receipt_missing_count: u32,
    short_dlp_critical_present: bool,
    short_timeout_count: u32,
    short_deny_count: u32,
    medium_receipt_invalid_count: u32,
    medium_dlp_critical_count: u32,
    medium_flapping_count: u32,
    medium_replay_mismatch: bool,
}

pub struct RegulationEngine {
    pub rsv: RsvState,
    config: RegulationConfig,
    pvgs_reader: Option<Box<dyn PvgsReader + Send + Sync>>,
    pvgs_writer: Option<Box<dyn PvgsWriter + Send>>,
    session_id: Option<String>,
    signal_queue: VecDeque<ucf::v1::SignalFrame>,
    anti_flapping_state: AntiFlappingState,
    counters: WindowCounters,
    brain_bus: BrainBus,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RegulationSnapshot {
    pub profile: ProfileState,
    pub overlays: OverlaySet,
    pub deescalation_lock: bool,
    pub control_frame_digest: Option<[u8; 32]>,
    pub evidence_refs: Vec<EvidenceRef>,
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
            Ok(config) => {
                let (hpa, last_hpa_output) = init_hpa();
                RegulationEngine {
                    rsv: RsvState::default(),
                    config,
                    pvgs_reader: None,
                    pvgs_writer: None,
                    session_id: None,
                    signal_queue: VecDeque::new(),
                    anti_flapping_state: AntiFlappingState::default(),
                    counters: WindowCounters::default(),
                    brain_bus: BrainBus::with_hpa(hpa, last_hpa_output),
                }
            }
            Err(_) => {
                let (hpa, last_hpa_output) = init_hpa();
                RegulationEngine {
                    rsv: RsvState::default(),
                    config: RegulationConfig::fallback(),
                    pvgs_reader: None,
                    pvgs_writer: None,
                    session_id: None,
                    signal_queue: VecDeque::new(),
                    anti_flapping_state: AntiFlappingState::default(),
                    counters: WindowCounters::default(),
                    brain_bus: BrainBus::with_hpa(hpa, last_hpa_output),
                }
            }
        }
    }
}

impl RegulationEngine {
    pub fn new(config: RegulationConfig) -> Self {
        let (hpa, last_hpa_output) = init_hpa();

        RegulationEngine {
            rsv: RsvState::default(),
            config,
            pvgs_reader: None,
            pvgs_writer: None,
            session_id: None,
            signal_queue: VecDeque::new(),
            anti_flapping_state: AntiFlappingState::default(),
            counters: WindowCounters::default(),
            brain_bus: BrainBus::with_hpa(hpa, last_hpa_output),
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
        let (decision, brain_output) = self.decide_on_missing(now_ms);
        let decision = self.apply_forensic_override(decision);
        let decision = self.apply_anti_flapping(decision, now_ms);
        let (control_frame, _) = self.build_control_frame(decision.clone(), &brain_output);
        let digest = control_frame
            .control_frame_digest
            .as_ref()
            .and_then(|bytes| bytes.as_slice().try_into().ok());

        RegulationSnapshot {
            profile: decision.profile,
            overlays: decision.overlays,
            deescalation_lock: decision.deescalation_lock,
            control_frame_digest: digest,
            evidence_refs: brain_output.evidence_refs.clone(),
            rsv_summary: self.rsv_summary(),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn emotion_field_snapshot(&self) -> Option<dbm_core::EmotionField> {
        self.brain_bus.last_emotion_field()
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

    pub fn tick(&mut self, now_ms: u64) -> ControlFrame {
        let mut control_frame: Option<ControlFrame> = None;
        let max_to_process = self
            .config
            .engine_limits
            .max_frames_per_tick
            .min(self.signal_queue.len());

        for _ in 0..max_to_process {
            if let Some(frame) = self.signal_queue.pop_front() {
                let (decision, brain_output) = self.decide_from_frame(frame, now_ms);
                control_frame = Some(self.apply_and_render(decision, &brain_output, now_ms));
            }
        }

        if let Some(control_frame) = control_frame {
            return control_frame;
        }

        let (decision, brain_output) = self.decide_on_missing(now_ms);
        self.apply_and_render(decision, &brain_output, now_ms)
    }

    #[cfg(test)]
    pub(crate) fn reset_forensic(&mut self) {
        self.rsv.reset_forensic();
        self.anti_flapping_state.current_profile = None;
        self.anti_flapping_state.current_overlays = None;
    }

    fn decide_from_frame(
        &mut self,
        mut frame: ucf::v1::SignalFrame,
        now_ms: u64,
    ) -> (ControlDecision, dbm_bus::BrainOutput) {
        if frame.timestamp_ms.is_none() {
            frame.timestamp_ms = Some(now_ms);
        }

        let timestamp_ms = frame.timestamp_ms.unwrap_or(now_ms);
        let classified = classify_signal_frame(&frame, &self.config.thresholds);
        apply_classification(&mut self.rsv, &classified, timestamp_ms);
        self.rsv.update_unlock_state(&frame);

        update_window_counters(&mut self.counters, &classified);

        let previous_dwm = self.brain_bus.current_dwm();
        let brain_output = self.build_and_tick_brain(&frame, &classified, timestamp_ms);
        self.update_rsv_from_brain(&brain_output);
        let decision = self.translate_brain_output(&brain_output, previous_dwm, false);

        (decision, brain_output)
    }

    fn decide_on_missing(&mut self, now_ms: u64) -> (ControlDecision, dbm_bus::BrainOutput) {
        self.rsv.missing_frame_counter = self.rsv.missing_frame_counter.saturating_add(1);
        let classified =
            profiles::classification::ClassifiedSignals::conservative(ucf::v1::WindowKind::Short);
        let frame = ucf::v1::SignalFrame {
            window_kind: ucf::v1::WindowKind::Short as i32,
            window_index: Some(now_ms),
            timestamp_ms: Some(now_ms),
            policy_stats: None,
            exec_stats: None,
            integrity_state: ucf::v1::IntegrityStateClass::Degraded as i32,
            top_reason_codes: Vec::new(),
            signal_frame_digest: None,
            receipt_stats: None,
            reason_codes: Vec::new(),
        };
        update_window_counters(&mut self.counters, &classified);

        let previous_dwm = self.brain_bus.current_dwm();
        let brain_output = self.build_and_tick_brain(&frame, &classified, now_ms);
        self.update_rsv_from_brain(&brain_output);
        let decision = self.translate_brain_output(&brain_output, previous_dwm, true);

        (decision, brain_output)
    }

    fn build_and_tick_brain(
        &mut self,
        frame: &ucf::v1::SignalFrame,
        classified: &profiles::classification::ClassifiedSignals,
        now_ms: u64,
    ) -> dbm_bus::BrainOutput {
        let cbv = self
            .pvgs_reader
            .as_ref()
            .and_then(|reader| reader.get_latest_cbv());
        let pev = self
            .pvgs_reader
            .as_ref()
            .and_then(|reader| reader.get_latest_pev());

        let cbv_present = self
            .pvgs_reader
            .as_ref()
            .and_then(|reader| reader.get_latest_cbv_digest())
            .is_some();
        let pev_present = self
            .pvgs_reader
            .as_ref()
            .and_then(|reader| reader.get_latest_pev_digest())
            .is_some();

        let sc_replay_planned_present = self
            .brain_bus
            .last_dopa_output()
            .map(|output| output.replay_hint)
            .unwrap_or(false);

        let baselines = BaselineContext {
            cbv,
            cbv_present,
            pev,
            pev_present,
        };

        let brain_input = brain_input_from_signal_frame(
            frame,
            classified,
            &self.counters,
            &self.rsv,
            baselines,
            sc_replay_planned_present,
            now_ms,
        );

        self.brain_bus.tick(brain_input)
    }

    fn translate_brain_output(
        &mut self,
        brain_output: &dbm_bus::BrainOutput,
        previous_dwm: DwmMode,
        missing_frame: bool,
    ) -> ControlDecision {
        let mut translated =
            translate_decision(brain_output.decision.clone(), &self.config, missing_frame);

        let mut reason_set = dbm_core::ReasonSet::default();
        reason_set.extend(brain_output.reason_codes.clone());
        self.extend_reason_codes(&mut translated, &reason_set);

        translated = self.apply_cerebellum_overlays(translated, brain_output);
        self.append_dwm_reason_codes(previous_dwm, brain_output, &mut translated);
        translated
    }

    fn update_rsv_from_brain(&mut self, brain_output: &dbm_bus::BrainOutput) {
        self.rsv.threat = translate_level(brain_output.isv.threat);
        self.rsv.arousal = translate_level(brain_output.isv.arousal);
        self.rsv.stability = translate_level(brain_output.isv.stability);
        self.rsv.policy_pressure = translate_level(brain_output.isv.policy_pressure);

        if let Some(cerebellum) = &brain_output.cerebellum {
            self.rsv.divergence = translate_level(cerebellum.divergence);
        }
    }

    fn apply_and_render(
        &mut self,
        decision: ControlDecision,
        brain_output: &dbm_bus::BrainOutput,
        now_ms: u64,
    ) -> ControlFrame {
        let decision = self.apply_forensic_override(decision);
        let decision = self.apply_anti_flapping(decision, now_ms);
        self.render_control_frame(decision, brain_output)
    }

    fn append_dwm_reason_codes(
        &self,
        previous_dwm: DwmMode,
        brain_output: &dbm_bus::BrainOutput,
        decision: &mut ControlDecision,
    ) {
        if previous_dwm != brain_output.dwm {
            decision
                .profile_reason_codes
                .push(dwm_reason_code(brain_output.dwm));
        }
    }

    fn extend_reason_codes(
        &self,
        decision: &mut ControlDecision,
        reason_set: &dbm_core::ReasonSet,
    ) {
        let mut translated = translate_reason_set(reason_set);
        decision.profile_reason_codes.append(&mut translated);
        decision
            .profile_reason_codes
            .sort_by_key(|code| *code as i32);
        decision.profile_reason_codes.dedup();
    }

    fn apply_cerebellum_overlays(
        &self,
        mut decision: ControlDecision,
        brain_output: &dbm_bus::BrainOutput,
    ) -> ControlDecision {
        let tool_side_effects = brain_output
            .isv
            .threat_vectors
            .as_ref()
            .is_some_and(|vectors| vectors.contains(&ThreatVector::ToolSideEffects));

        if matches!(
            brain_output
                .cerebellum
                .as_ref()
                .map(|output| output.divergence),
            Some(BrainLevel::High)
        ) || tool_side_effects
        {
            decision.overlays.simulate_first = true;
            decision.overlays.chain_tightening = true;
        }

        decision
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

    fn render_control_frame(
        &mut self,
        decision: ControlDecision,
        brain_output: &dbm_bus::BrainOutput,
    ) -> ControlFrame {
        let (control_frame, digest_bytes) = self.build_control_frame(decision, brain_output);

        if let Some(writer) = self.pvgs_writer.as_mut() {
            let _ = writer.commit_control_frame_evidence(
                self.session_id.as_deref().unwrap_or("default-session"),
                digest_bytes,
            );
        }

        control_frame
    }

    fn build_control_frame(
        &self,
        decision: ControlDecision,
        brain_output: &dbm_bus::BrainOutput,
    ) -> (ControlFrame, [u8; 32]) {
        let cbv_digest = self
            .pvgs_reader
            .as_ref()
            .and_then(|reader| reader.get_latest_cbv_digest())
            .map(|digest| digest.to_vec());
        let cbv = self
            .pvgs_reader
            .as_ref()
            .and_then(|reader| reader.get_latest_cbv());
        let pev_digest = self
            .pvgs_reader
            .as_ref()
            .and_then(|reader| reader.get_latest_pev_digest())
            .map(|digest| digest.to_vec());
        let pev = self
            .pvgs_reader
            .as_ref()
            .and_then(|reader| reader.get_latest_pev());

        let context = ControlFrameContext {
            cbv,
            cbv_digest,
            pev,
            pev_digest,
            forensic_latched: self.rsv.forensic_latched,
        };

        control_frame_from_brain_output(decision, brain_output, &self.config, context)
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
}

pub(crate) fn toolclass_mask_for(
    config: &RegulationConfig,
    decision: &ControlDecision,
    forensic_latched: bool,
) -> ToolClassMask {
    if forensic_latched {
        return ToolClassMask {
            read: true,
            write: false,
            execute: false,
            transform: true,
            export: false,
        };
    }

    let mut mask_cfg = config.profiles.get(decision.profile).toolclass_mask.clone();

    if decision.overlays.simulate_first {
        if let Some(overlay_mask) = &config.overlays.simulate_first.toolclass_mask {
            mask_cfg = mask_cfg.merge_overlay(overlay_mask);
        }
    }

    if decision.overlays.export_lock {
        if let Some(overlay_mask) = &config.overlays.export_lock.toolclass_mask {
            mask_cfg = mask_cfg.merge_overlay(overlay_mask);
        }
    }

    if decision.overlays.novelty_lock {
        if let Some(overlay_mask) = &config.overlays.novelty_lock.toolclass_mask {
            mask_cfg = mask_cfg.merge_overlay(overlay_mask);
        }
    }

    mask_cfg.to_tool_class_mask()
}

fn level_severity(level: BrainLevel) -> u8 {
    match level {
        BrainLevel::Low => 0,
        BrainLevel::Med => 1,
        BrainLevel::High => 2,
    }
}

pub(crate) fn level_at_least(level: BrainLevel, threshold: BrainLevel) -> bool {
    level_severity(level) >= level_severity(threshold)
}

fn init_hpa() -> (Hpa, HpaOutput) {
    if let Ok(path) = std::env::var("HPA_STATE_PATH") {
        let hpa = Hpa::new(PathBuf::from(path));
        let last_hpa_output = hpa.current_output();
        return (hpa, last_hpa_output);
    }

    #[cfg(test)]
    {
        let nonce = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "hpa_state_engine_{}_{}.json",
            std::process::id(),
            nonce
        ));
        let _ = std::fs::remove_file(&path);
        let hpa = Hpa::new(&path);
        let last_hpa_output = hpa.current_output();
        (hpa, last_hpa_output)
    }

    #[cfg(not(test))]
    {
        let hpa = Hpa::default();
        let last_hpa_output = hpa.current_output();
        (hpa, last_hpa_output)
    }
}

pub(crate) fn dlp_flags(frame: &ucf::v1::SignalFrame) -> (bool, bool, bool) {
    (
        frame_has_reason(frame, ReasonCode::RcCdDlpSecretPattern),
        frame_has_reason(frame, ReasonCode::RcCdDlpObfuscation),
        frame_has_reason(frame, ReasonCode::RcCdDlpStegano),
    )
}

fn frame_has_reason(frame: &ucf::v1::SignalFrame, code: ReasonCode) -> bool {
    let code = code as i32;
    frame.reason_codes.contains(&code)
        || frame.top_reason_codes.contains(&code)
        || frame
            .policy_stats
            .as_ref()
            .map(|stats| stats.top_reason_codes.contains(&code))
            .unwrap_or(false)
        || frame
            .exec_stats
            .as_ref()
            .map(|stats| stats.top_reason_codes.contains(&code))
            .unwrap_or(false)
}

fn update_window_counters(
    counters: &mut WindowCounters,
    classified: &profiles::classification::ClassifiedSignals,
) {
    use ucf::v1::WindowKind;

    match classified.window_kind {
        WindowKind::Short => {
            counters.short_receipt_invalid_count = classified.receipt_invalid_count;
            counters.short_receipt_missing_count = classified.receipt_missing_count;
            counters.short_dlp_critical_present =
                classified.dlp_severity_class == ucf::v1::LevelClass::High;
            counters.short_timeout_count = classified.exec_timeout_count;
            counters.short_deny_count = classified.policy_deny_count;
        }
        WindowKind::Medium => {
            counters.medium_receipt_invalid_count = classified.receipt_invalid_count;
            counters.medium_dlp_critical_count =
                if classified.dlp_severity_class == ucf::v1::LevelClass::High {
                    classified.policy_deny_count.max(1)
                } else {
                    0
                };
            counters.medium_flapping_count = counters
                .medium_flapping_count
                .max(classified.policy_deny_count);
            counters.medium_replay_mismatch =
                classified.replay_mismatch_class == ucf::v1::LevelClass::High;
        }
        _ => {}
    }
}

pub(crate) fn to_brain_level(level: ucf::v1::LevelClass) -> BrainLevel {
    match level {
        ucf::v1::LevelClass::High => BrainLevel::High,
        ucf::v1::LevelClass::Med => BrainLevel::Med,
        _ => BrainLevel::Low,
    }
}

pub(crate) fn to_brain_integrity(state: ucf::v1::IntegrityStateClass) -> IntegrityState {
    match state {
        ucf::v1::IntegrityStateClass::Fail => IntegrityState::Fail,
        ucf::v1::IntegrityStateClass::Degraded => IntegrityState::Degraded,
        _ => IntegrityState::Ok,
    }
}

fn dwm_reason_code(mode: DwmMode) -> ucf::v1::ReasonCode {
    match mode {
        DwmMode::Report => ucf::v1::ReasonCode::RcGvDwmReport,
        DwmMode::Stabilize => ucf::v1::ReasonCode::RcGvDwmStabilize,
        DwmMode::Simulate => ucf::v1::ReasonCode::RcGvDwmSimulate,
        DwmMode::ExecPlan => ucf::v1::ReasonCode::RcGvDwmExecPlan,
    }
}

fn translate_decision(
    decision: HypoDecision,
    config: &RegulationConfig,
    missing_frame: bool,
) -> ControlDecision {
    let profile = translate_profile_state(decision.profile_state);
    let profile_def = config.profiles.get(profile);
    let overlays = translate_overlay(decision.overlays);
    let mut profile_reason_codes = translate_reason_set(&decision.reason_codes);

    if missing_frame && !profile_reason_codes.contains(&ucf::v1::ReasonCode::ReIntegrityDegraded) {
        profile_reason_codes.push(ucf::v1::ReasonCode::ReIntegrityDegraded);
    }

    ControlDecision {
        profile,
        overlays: overlays.merge(&OverlaySet {
            chain_tightening: overlays.simulate_first
                || overlays.export_lock
                || overlays.novelty_lock,
            ..OverlaySet::default()
        }),
        deescalation_lock: decision.deescalation_lock,
        missing_frame_override: missing_frame,
        profile_reason_codes,
        approval_mode: profile_def.approval_mode.clone(),
        cooldown_class: translate_level(decision.cooldown_class),
    }
}

fn translate_profile_state(state: BrainProfile) -> ProfileState {
    match state {
        BrainProfile::M0 => ProfileState::M0Research,
        BrainProfile::M1 => ProfileState::M1Restricted,
        BrainProfile::M2 => ProfileState::M2Quarantine,
        BrainProfile::M3 => ProfileState::M3Forensic,
    }
}

fn translate_overlay(overlay: BrainOverlay) -> OverlaySet {
    OverlaySet {
        simulate_first: overlay.simulate_first,
        export_lock: overlay.export_lock,
        novelty_lock: overlay.novelty_lock,
        chain_tightening: false,
    }
}

fn translate_level(level: BrainLevel) -> ucf::v1::LevelClass {
    match level {
        BrainLevel::Low => ucf::v1::LevelClass::Low,
        BrainLevel::Med => ucf::v1::LevelClass::Med,
        BrainLevel::High => ucf::v1::LevelClass::High,
    }
}

fn translate_reason_set(reason_set: &dbm_core::ReasonSet) -> Vec<ucf::v1::ReasonCode> {
    let mut translated: Vec<ucf::v1::ReasonCode> = reason_set
        .codes
        .iter()
        .filter_map(|reason| match reason.as_str() {
            "integrity_fail" | "ReIntegrityFail" => Some(ucf::v1::ReasonCode::ReIntegrityFail),
            "integrity_degraded" => Some(ucf::v1::ReasonCode::ReIntegrityDegraded),
            "ReIntegrityDegraded" => Some(ucf::v1::ReasonCode::ReIntegrityDegraded),
            "policy_pressure_high" | "RcGvPevUpdated" | "pev_influence" => {
                Some(ucf::v1::ReasonCode::RcGvPevUpdated)
            }
            "RcGvCbvUpdated" | "cbv_influence" => Some(ucf::v1::ReasonCode::RcGvCbvUpdated),
            "threat_high" | "ThExfilHighConfidence" => {
                Some(ucf::v1::ReasonCode::ThExfilHighConfidence)
            }
            "ThPolicyProbing" => Some(ucf::v1::ReasonCode::ThPolicyProbing),
            "RcGeExecDispatchBlocked" => Some(ucf::v1::ReasonCode::RcGeExecDispatchBlocked),
            "RcThIntegrityCompromise" => Some(ucf::v1::ReasonCode::RcThIntegrityCompromise),
            "RcRxActionForensic" | "RC.RX.ACTION.FORENSIC" => {
                Some(ucf::v1::ReasonCode::RcRxActionForensic)
            }
            "RC.GV.HOLD.ON" => Some(ucf::v1::ReasonCode::RcGvHoldOn),
            "RC.GV.SEQUENCE.SPLIT_REQUIRED" => Some(ucf::v1::ReasonCode::RcGvSequenceSplitRequired),
            "RC.GV.SEQUENCE.SLOW" => Some(ucf::v1::ReasonCode::RcGvSequenceSlow),
            "RcRgProfileM1Restricted" => Some(ucf::v1::ReasonCode::RcRgProfileM1Restricted),
            "RcGvDwmReport" | "RC.GV.DWM.REPORT" => Some(ucf::v1::ReasonCode::RcGvDwmReport),
            "RcGvDwmStabilize" | "RC.GV.DWM.STABILIZE" => {
                Some(ucf::v1::ReasonCode::RcGvDwmStabilize)
            }
            "RcGvDwmSimulate" | "RC.GV.DWM.SIMULATE" => Some(ucf::v1::ReasonCode::RcGvDwmSimulate),
            "RcGvDwmExecPlan" | "RC.GV.DWM.EXEC_PLAN" => Some(ucf::v1::ReasonCode::RcGvDwmExecPlan),
            "RC.GV.ORIENT.TARGET_INTEGRITY" => Some(ucf::v1::ReasonCode::RcGvOrientTargetIntegrity),
            "RC.GV.ORIENT.TARGET_DLP" => Some(ucf::v1::ReasonCode::RcGvOrientTargetDlp),
            "RC.GV.ORIENT.TARGET_RECOVERY" => Some(ucf::v1::ReasonCode::RcGvOrientTargetRecovery),
            "RC.GV.ORIENT.TARGET_APPROVAL" => Some(ucf::v1::ReasonCode::RcGvOrientTargetApproval),
            "RC.GV.ORIENT.TARGET_REPLAY" => Some(ucf::v1::ReasonCode::RcGvOrientTargetReplay),
            "RC.GV.ORIENT.TARGET_POLICY_PRESSURE" => {
                Some(ucf::v1::ReasonCode::RcGvOrientTargetPolicyPressure)
            }
            "RC.GV.FOCUS_SHIFT.EXECUTED" => Some(ucf::v1::ReasonCode::RcGvFocusShiftExecuted),
            "RC.GV.FOCUS_SHIFT.BLOCKED_BY_LOCK" => {
                Some(ucf::v1::ReasonCode::RcGvFocusShiftBlockedByLock)
            }
            "RC.GV.FLAPPING.PENALTY" => Some(ucf::v1::ReasonCode::RcGvFlappingPenalty),
            "RC.GV.DIVERGENCE.HIGH" => Some(ucf::v1::ReasonCode::RcGvDivergenceHigh),
            "RC.GV.TOOL.SUSPEND_RECOMMENDED" => {
                Some(ucf::v1::ReasonCode::RcGvToolSuspendRecommended)
            }
            "RC.GV.PROGRESS.REWARD_BLOCKED" | "baseline_reward_block" => {
                Some(ucf::v1::ReasonCode::RcGvProgressRewardBlocked)
            }
            "RC.GV.REPLAY.DIMINISHING_RETURNS" => {
                Some(ucf::v1::ReasonCode::RcGvReplayDiminishingReturns)
            }
            reason if reason.starts_with("hpa_") => Some(ucf::v1::ReasonCode::RcGvPevUpdated),
            reason if reason.starts_with("baseline_") => Some(ucf::v1::ReasonCode::RcGvCbvUpdated),
            _ => None,
        })
        .collect();

    translated.sort_by_key(|code| *code as i32);
    translated.dedup();
    translated
}

#[cfg(test)]
mod tests {
    use super::*;
    use dbm_core::OrientTarget;
    use profiles::{OverlayEnableRule, OverlaySet, ProfileState, RegulationConfig, RuleCondition};
    use pvgs_client::{MockPvgsReader, PvgsError};
    use std::sync::{Arc, Mutex};
    use ucf::v1::{
        CharacterBaselineVector, ExecStats, IntegrityStateClass, LevelClass, PolicyStats,
        ReasonCode, ReceiptStats, SignalFrame, WindowKind,
    };

    type EvidenceLog = Arc<Mutex<Vec<(String, [u8; 32])>>>;

    fn drive_frame(engine: &mut RegulationEngine, frame: SignalFrame, now_ms: u64) -> ControlFrame {
        engine.enqueue_signal_frame(frame).expect("frame enqueued");
        engine.tick(now_ms)
    }

    fn pending_frames(engine: &RegulationEngine) -> usize {
        engine.signal_queue.len()
    }

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
                partial_failure_count: 0,
                tool_unavailable_count: 0,
                tool_id: None,
                dlp_block_count: 0,
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

    fn medium_frame(ts: u64, allow_count: u32, deny_count: u32) -> SignalFrame {
        SignalFrame {
            window_kind: WindowKind::Medium as i32,
            window_index: Some(ts),
            timestamp_ms: Some(ts),
            policy_stats: Some(PolicyStats {
                deny_count,
                allow_count,
                top_reason_codes: Vec::new(),
            }),
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

    fn reset_hpa(engine: &mut RegulationEngine) {
        let (hpa, last_hpa_output) = init_hpa();
        *engine.brain_bus.hpa_mut() = hpa;
        engine.brain_bus.set_last_hpa_output(last_hpa_output);
    }

    #[test]
    fn hbv_arousal_floor_enables_overlays() {
        let mut engine = RegulationEngine::default();
        let mut last_hpa_output = engine.brain_bus.last_hpa_output();
        last_hpa_output.baseline_caution_offset = 4;
        engine.brain_bus.set_last_hpa_output(last_hpa_output);

        let control = drive_frame(&mut engine, base_frame(), 1);

        let overlays = control.overlays.expect("overlays present");
        assert!(overlays.simulate_first);
        assert!(overlays.novelty_lock);
    }

    #[test]
    fn hbv_cooldown_bias_applies() {
        let mut engine = RegulationEngine::default();
        let mut last_hpa_output = engine.brain_bus.last_hpa_output();
        last_hpa_output.baseline_cooldown_multiplier_class = 3;
        engine.brain_bus.set_last_hpa_output(last_hpa_output);

        let control = drive_frame(&mut engine, base_frame(), 1);
        assert_eq!(
            control.cooldown_class,
            Some(ucf::v1::LevelClass::High as i32)
        );
    }

    #[test]
    fn hbv_biases_hypothalamus_defaults() {
        let mut engine = RegulationEngine::default();
        let mut last_hpa_output = engine.brain_bus.last_hpa_output();
        last_hpa_output.baseline_chain_conservatism_offset = 2;
        last_hpa_output.baseline_export_strictness_offset = 1;
        last_hpa_output.baseline_approval_strictness_offset = 1;
        engine.brain_bus.set_last_hpa_output(last_hpa_output);

        let control = drive_frame(&mut engine, base_frame(), 1);
        let overlays = control.overlays.expect("overlays present");
        assert!(overlays.simulate_first);
        assert!(overlays.export_lock);
        assert_eq!(
            control.approval_mode.as_deref(),
            Some(
                engine
                    .config
                    .character_baselines
                    .strict_approval_mode
                    .as_str()
            )
        );
    }

    #[test]
    fn digest_is_deterministic() {
        let mut engine_a = RegulationEngine::default();
        let mut engine_b = RegulationEngine::default();
        let frame = base_frame();
        let control_a = drive_frame(&mut engine_a, frame.clone(), 1);
        let control_b = drive_frame(&mut engine_b, frame, 1);
        assert_eq!(
            control_a.control_frame_digest,
            control_b.control_frame_digest
        );
    }

    #[test]
    fn integrity_failure_drives_focus_lock_and_blocks_looser_shift() {
        let mut engine = RegulationEngine::default();

        let mut integrity_frame = base_frame();
        integrity_frame.integrity_state = IntegrityStateClass::Fail as i32;
        integrity_frame.timestamp_ms = Some(1);

        let control_integrity = drive_frame(&mut engine, integrity_frame, 1);

        assert_eq!(engine.brain_bus.current_target(), OrientTarget::Integrity);
        assert_eq!(engine.brain_bus.current_dwm(), DwmMode::Report);
        assert!(engine.brain_bus.pprf().lock_until_ms > 1);
        assert!(control_integrity
            .profile_reason_codes
            .contains(&(ucf::v1::ReasonCode::RcGvOrientTargetIntegrity as i32)));

        let mut calm_frame = base_frame();
        calm_frame.timestamp_ms = Some(2);
        let control_blocked = drive_frame(&mut engine, calm_frame, 2);

        assert_eq!(engine.brain_bus.current_target(), OrientTarget::Integrity);
        assert!(control_blocked
            .profile_reason_codes
            .contains(&(ucf::v1::ReasonCode::RcGvFocusShiftBlockedByLock as i32)));
    }

    #[test]
    fn stricter_target_preempts_existing_lock() {
        let mut engine = RegulationEngine::default();

        let mut initial_frame = base_frame();
        initial_frame.timestamp_ms = Some(10);
        let _ = drive_frame(&mut engine, initial_frame, 10);

        let initial_lock = engine.brain_bus.pprf().lock_until_ms;
        assert!(initial_lock > 10);
        assert_eq!(engine.brain_bus.current_target(), OrientTarget::Approval);

        let mut integrity_frame = base_frame();
        integrity_frame.integrity_state = IntegrityStateClass::Fail as i32;
        integrity_frame.timestamp_ms = Some(11);
        let control = drive_frame(&mut engine, integrity_frame, 11);

        assert_eq!(engine.brain_bus.current_target(), OrientTarget::Integrity);
        assert_eq!(engine.brain_bus.current_dwm(), DwmMode::Report);
        assert!(engine.brain_bus.pprf().lock_until_ms > initial_lock);
        assert!(control
            .profile_reason_codes
            .contains(&(ucf::v1::ReasonCode::RcGvFocusShiftExecuted as i32)));
    }

    #[test]
    fn cbv_digest_embeds_and_changes_control_frame_hash() {
        let frame = base_frame();

        let mut engine_with_cbv_a = RegulationEngine::default();
        engine_with_cbv_a.set_pvgs_reader(MockPvgsReader::with_cbv(cbv_digest(1)));
        let control_a = drive_frame(&mut engine_with_cbv_a, frame.clone(), 1);

        let mut engine_with_cbv_b = RegulationEngine::default();
        engine_with_cbv_b.set_pvgs_reader(MockPvgsReader::with_cbv(cbv_digest(2)));
        let control_b = drive_frame(&mut engine_with_cbv_b, frame, 1);

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
        let control_a = drive_frame(&mut engine_with_pev_a, frame.clone(), 1);

        let mut reader_b = MockPvgsReader::with_pev_digest(pev_digest(9));
        reader_b.pev = Some(pev_vector());
        let mut engine_with_pev_b = RegulationEngine::default();
        engine_with_pev_b.set_pvgs_reader(reader_b);
        let control_b = drive_frame(&mut engine_with_pev_b, frame, 1);

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
        let control_a = drive_frame(&mut engine_a, frame.clone(), 1);

        let mut engine_b = RegulationEngine::default();
        let control_b = drive_frame(&mut engine_b, frame, 1);

        assert!(control_a.character_epoch_digest.is_none());
        assert_eq!(
            control_a.control_frame_digest,
            control_b.control_frame_digest
        );
    }

    #[test]
    fn threat_high_blocks_progress_rewards() {
        let mut engine = RegulationEngine::default();

        let mut frame = medium_frame(1, 12, 0);
        frame.integrity_state = IntegrityStateClass::Fail as i32;
        let control = drive_frame(&mut engine, frame, 1);

        let dopa = engine.brain_bus.last_dopa_output().unwrap();
        assert_eq!(dopa.progress, BrainLevel::Med);
        assert!(control
            .profile_reason_codes
            .contains(&(ReasonCode::RcGvProgressRewardBlocked as i32)));
    }

    #[test]
    fn stable_successes_raise_progress_when_safe() {
        let mut engine = RegulationEngine::default();

        for ts in 1..=2 {
            let frame = medium_frame(ts, 4, 0);
            let _ = drive_frame(&mut engine, frame, ts);
        }

        let dopa = engine.brain_bus.last_dopa_output().unwrap();
        assert_eq!(dopa.progress, BrainLevel::High);
        assert!(!dopa.replay_hint);
    }

    #[test]
    fn diminishing_returns_emit_replay_hint_and_reason_code() {
        let mut engine = RegulationEngine::default();

        for ts in 1..=3 {
            let mut frame = medium_frame(ts, 0, 5);
            frame.exec_stats.as_mut().unwrap().timeout_count = 1;
            let control = drive_frame(&mut engine, frame, ts);
            if ts == 3 {
                assert!(engine
                    .brain_bus
                    .last_dopa_output()
                    .map(|out| out.replay_hint)
                    .unwrap_or(false));
                assert!(control
                    .profile_reason_codes
                    .contains(&(ReasonCode::RcGvReplayDiminishingReturns as i32)));
            }
        }
    }

    #[test]
    fn pev_absence_is_stable() {
        let frame = base_frame();

        let mut engine_a = RegulationEngine::default();
        let control_a = drive_frame(&mut engine_a, frame.clone(), 1);

        let mut engine_b = RegulationEngine::default();
        let control_b = drive_frame(&mut engine_b, frame, 1);

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

        let control = drive_frame(&mut engine, base_frame(), 1);
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

        let control = drive_frame(&mut engine, base_frame(), 1);
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

        let control = drive_frame(&mut engine, base_frame(), 1);
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

        let control = drive_frame(&mut engine, base_frame(), 1);
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

        let control = drive_frame(&mut engine, base_frame(), 1);
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
        let control_a = drive_frame(&mut engine_a, base_frame(), 1);

        let mut engine_b = RegulationEngine::new(config);
        engine_b.set_pvgs_reader(reader);
        let control_b = drive_frame(&mut engine_b, base_frame(), 1);

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
        let control_a = drive_frame(&mut engine_a, base_frame(), 1);

        let mut engine_b = RegulationEngine::default();
        engine_b.set_pvgs_reader(reader);
        let control_b = drive_frame(&mut engine_b, base_frame(), 1);

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

        let control = drive_frame(&mut engine, base_frame(), 1);

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

        let control = drive_frame(&mut engine, frame, 1);
        assert_eq!(control.active_profile.unwrap().profile, "M2_QUARANTINE");
        let overlays = control.overlays.unwrap();
        assert!(overlays.simulate_first && overlays.export_lock && overlays.novelty_lock);
        assert!(control.deescalation_lock.unwrap_or(false));
    }

    #[test]
    fn integrity_fail_triggers_forensic_profile() {
        let mut engine = RegulationEngine::default();
        let mut frame = base_frame();
        frame.integrity_state = IntegrityStateClass::Fail as i32;
        let control = drive_frame(&mut engine, frame, 1);

        let mask = control.toolclass_mask.unwrap();
        assert!(!mask.export && !mask.write && !mask.execute);
        assert!(mask.read && mask.transform);
        assert_eq!(control.active_profile.unwrap().profile, "M3_FORENSIC");
        let overlays = control.overlays.unwrap();
        assert!(overlays.simulate_first);
        assert!(overlays.export_lock);
        assert!(overlays.novelty_lock);
        assert!(overlays.chain_tightening);
        let mut reasons = control.profile_reason_codes.clone();
        reasons.sort();
        reasons.dedup();
        assert!(reasons.contains(&(ReasonCode::ReIntegrityFail as i32)));
        assert!(reasons.contains(&(ReasonCode::RcRxActionForensic as i32)));
        assert!(reasons.contains(&(ReasonCode::RcGvDwmReport as i32)));
        assert_eq!(control.deescalation_lock, Some(true));
        assert_eq!(control.cooldown_class, Some(LevelClass::High as i32));
    }

    #[test]
    fn dlp_secret_drives_amygdala_and_pag_to_forensic() {
        let mut engine = RegulationEngine::default();
        let mut frame = base_frame();
        frame.top_reason_codes = vec![ReasonCode::RcCdDlpSecretPattern as i32];

        let control = drive_frame(&mut engine, frame, 1);
        assert_eq!(control.active_profile.unwrap().profile, "M3_FORENSIC");
        let mut reasons = control.profile_reason_codes.clone();
        reasons.sort();
        reasons.dedup();
        assert!(reasons.contains(&(ReasonCode::RcRxActionForensic as i32)));
    }

    #[test]
    fn integrity_fail_without_unlock_remains_forensic() {
        let mut engine = RegulationEngine::default();
        let mut frame = base_frame();
        frame.window_kind = WindowKind::Medium as i32;
        frame.integrity_state = IntegrityStateClass::Fail as i32;
        frame.reason_codes = vec![ReasonCode::ReIntegrityFail as i32];

        let control_first = drive_frame(&mut engine, frame.clone(), 1);
        assert_eq!(control_first.active_profile.unwrap().profile, "M3_FORENSIC");

        let control_second = drive_frame(&mut engine, frame, 2);
        assert_eq!(
            control_second.active_profile.unwrap().profile,
            "M3_FORENSIC"
        );
    }

    #[test]
    fn policy_pressure_high_uses_contained_continue() {
        let mut engine = RegulationEngine::default();
        let mut frame = base_frame();
        frame.policy_stats = Some(PolicyStats {
            deny_count: 5,
            allow_count: 0,
            top_reason_codes: Vec::new(),
        });

        let control = drive_frame(&mut engine, frame, 1);
        assert_eq!(control.active_profile.unwrap().profile, "M1_RESTRICTED");
        assert!(control
            .overlays
            .as_ref()
            .map(|o| o.simulate_first)
            .unwrap_or(false));
    }

    #[test]
    fn single_unlock_window_keeps_forensic_profile() {
        let mut engine = RegulationEngine::default();
        let frame = medium_unlock_frame(
            1,
            IntegrityStateClass::Fail,
            vec![ReasonCode::ReIntegrityFail as i32],
        );

        let control = drive_frame(&mut engine, frame, 1);
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
        let _ = drive_frame(&mut engine, first, 1);

        let second = medium_unlock_frame(
            2,
            IntegrityStateClass::Degraded,
            vec![ReasonCode::ReIntegrityDegraded as i32],
        );
        let control = drive_frame(&mut engine, second, 2);

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
        let _ = drive_frame(&mut engine, frame, 1);

        let mut ok_frame = base_frame();
        ok_frame.integrity_state = IntegrityStateClass::Ok as i32;
        let control = drive_frame(&mut engine, ok_frame, 2);
        assert_eq!(control.active_profile.unwrap().profile, "M3_FORENSIC");

        engine.reset_forensic();

        let control_after_reset = drive_frame(&mut engine, base_frame(), 3);
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

        let control_a = drive_frame(&mut engine_a, frame.clone(), 1);
        let control_b = drive_frame(&mut engine_b, frame, 1);

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

        reset_hpa(&mut engine);
        let control = drive_frame(&mut engine, frame, 1);
        let overlays = control.overlays.unwrap();

        assert_eq!(control.active_profile.unwrap().profile, "M0_RESEARCH");
        assert!(!overlays.simulate_first);
        assert!(!overlays.export_lock);
        assert!(!overlays.novelty_lock);
        assert!(!overlays.chain_tightening);
    }

    #[test]
    fn degraded_integrity_with_replay_mismatch_escalates_profile() {
        let mut engine = RegulationEngine::default();
        let mut frame = base_frame();
        frame.integrity_state = IntegrityStateClass::Degraded as i32;
        frame.top_reason_codes = vec![ReasonCode::RcReReplayMismatch as i32];

        let control = drive_frame(&mut engine, frame, 1);

        assert_eq!(control.active_profile.unwrap().profile, "M0_RESEARCH");
    }

    #[test]
    fn missing_frame_triggers_restriction() {
        let mut engine = RegulationEngine::default();
        let frame = base_frame();
        let _ = drive_frame(&mut engine, frame, 0);
        let control = engine.tick(60_000);
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
        assert_eq!(pending_frames(&engine), 17);
    }

    #[test]
    fn tighten_is_allowed_during_cooldown() {
        let mut engine = RegulationEngine::default();
        engine.config.anti_flapping.min_ms_between_profile_changes = 10_000;

        let _ = drive_frame(&mut engine, base_frame(), 0);
        let control = drive_frame(&mut engine, replay_mismatch_frame(1), 1);
        assert_eq!(control.active_profile.unwrap().profile, "M0_RESEARCH");

        let control = drive_frame(&mut engine, base_frame(), 2);
        assert_eq!(control.active_profile.unwrap().profile, "M0_RESEARCH");
    }

    #[test]
    fn overlays_only_relax_after_cooldown() {
        let mut engine = RegulationEngine::default();
        reset_hpa(&mut engine);
        engine.config.update_tables.overlay_enable.clear();
        engine.config.anti_flapping.min_ms_between_overlay_changes = 10_000;

        let control = drive_frame(&mut engine, base_frame(), 0);
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

        let control = drive_frame(&mut engine, base_frame(), 1);
        let overlays = control.overlays.unwrap();
        assert!(!overlays.simulate_first && !overlays.export_lock && !overlays.novelty_lock);

        engine.config.update_tables.overlay_enable.clear();
        let control = drive_frame(&mut engine, base_frame(), 2);
        assert!(!control.overlays.unwrap().simulate_first);
    }

    #[test]
    fn flapping_penalty_forces_restriction() {
        let mut engine = RegulationEngine::default();
        reset_hpa(&mut engine);
        engine.config.anti_flapping.max_switches_per_medium_window = 1;
        engine.config.anti_flapping.min_ms_between_profile_changes = 0;
        engine.config.anti_flapping.min_ms_between_overlay_changes = 0;
        engine.config.update_tables.overlay_enable.clear();

        let _ = drive_frame(&mut engine, base_frame(), 0);
        let control = drive_frame(&mut engine, replay_mismatch_frame(1), 1);
        assert_eq!(control.active_profile.unwrap().profile, "M0_RESEARCH");

        let control = drive_frame(&mut engine, base_frame(), 2);
        let overlays = control.overlays.unwrap();
        assert_eq!(control.active_profile.unwrap().profile, "M0_RESEARCH");
        assert!(!overlays.simulate_first && !overlays.export_lock && !overlays.novelty_lock);
        assert!(control.deescalation_lock.is_none());
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
    fn stn_policy_pressure_hold_enables_overlays() {
        let mut frame = base_frame();
        frame.policy_stats = Some(PolicyStats {
            deny_count: 10,
            allow_count: 0,
            top_reason_codes: Vec::new(),
        });

        let mut engine = RegulationEngine::default();
        let control = drive_frame(&mut engine, frame, 1);
        let overlays = control.overlays.unwrap();

        assert!(overlays.simulate_first);
        assert!(overlays.novelty_lock);
        assert!(control
            .profile_reason_codes
            .contains(&(ReasonCode::RcGvHoldOn as i32)));
    }

    #[test]
    fn pmrf_divergence_split_enables_checkpointing() {
        let mut frame = base_frame();
        frame.window_kind = WindowKind::Medium as i32;
        frame.window_index = Some(1);
        frame.exec_stats = Some(ExecStats {
            timeout_count: 10,
            partial_failure_count: 0,
            tool_unavailable_count: 0,
            tool_id: None,
            dlp_block_count: 0,
            top_reason_codes: Vec::new(),
        });

        let mut engine = RegulationEngine::default();
        let control = drive_frame(&mut engine, frame, 1);
        let overlays = control.overlays.unwrap();

        assert!(overlays.simulate_first);
        assert!(overlays.chain_tightening);
        assert!(control
            .profile_reason_codes
            .contains(&(ReasonCode::RcGvSequenceSplitRequired as i32)));
        assert!(control
            .profile_reason_codes
            .contains(&(ReasonCode::RcGvDivergenceHigh as i32)));
    }

    #[test]
    fn repeated_divergence_high_recommends_suspend() {
        let mut engine = RegulationEngine::default();
        let mut frame = base_frame();
        frame.window_kind = WindowKind::Medium as i32;
        frame.window_index = Some(1);
        frame.exec_stats = Some(ExecStats {
            timeout_count: 10,
            partial_failure_count: 0,
            tool_unavailable_count: 0,
            tool_id: Some("tool-a".to_string()),
            dlp_block_count: 0,
            top_reason_codes: Vec::new(),
        });

        let _ = drive_frame(&mut engine, frame.clone(), 1);
        let mut second_frame = frame.clone();
        second_frame.window_index = Some(2);
        let control = drive_frame(&mut engine, second_frame, 2);

        assert!(control
            .profile_reason_codes
            .contains(&(ReasonCode::RcGvToolSuspendRecommended as i32)));
        assert!(control
            .profile_reason_codes
            .contains(&(ReasonCode::RcGvDivergenceHigh as i32)));
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

        let mask = toolclass_mask_for(&engine.config, &decision, engine.rsv.forensic_latched);
        assert!(!mask.export);
        assert!(mask.read);
    }

    #[test]
    fn fallback_config_is_conservative() {
        let mut engine = RegulationEngine::new(RegulationConfig::fallback());
        let control = drive_frame(&mut engine, base_frame(), 1);
        let mask = control.toolclass_mask.unwrap();
        assert_eq!(control.active_profile.unwrap().profile, "M0_RESEARCH");
        assert!(mask.read || mask.transform);
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

        let control_frame = engine.render_control_frame(decision, &dbm_bus::BrainOutput::default());
        assert_eq!(
            control_frame.profile_reason_codes,
            vec![
                ReasonCode::RcGeExecDispatchBlocked as i32,
                ReasonCode::RcThIntegrityCompromise as i32,
            ]
        );
    }
}
