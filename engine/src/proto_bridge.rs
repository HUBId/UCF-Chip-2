use blake3::Hasher;
use dbm_12_insula::InsulaInput;
use dbm_18_cerebellum::{CerInput, ToolFailureCounts};
use dbm_6_dopamin_nacc::DopaInput;
use dbm_7_lc::LcInput;
use dbm_8_serotonin::SerInput;
use dbm_9_amygdala::AmyInput;
use dbm_bus::BrainInput;
use dbm_core::{CooldownClass as BrainCooldown, LevelClass as BrainLevel, ReasonSet, ToolKey};
use dbm_hpa::HpaInput;
use dbm_pag::PagInput;
use dbm_pmrf::PmrfInput;
use dbm_stn::StnInput;
use profiles::classification::ClassifiedSignals;
use profiles::{apply_cbv_modifiers, apply_pev_modifiers, ControlDecision, RegulationConfig};
use prost::Message;
use ucf::v1::{
    ActiveProfile, CharacterBaselineVector, ControlFrame, ExecStats, IntegrityStateClass, Overlays,
    PolicyEcologyVector, PolicyStats, ReasonCode, ReceiptStats,
};

use crate::{
    level_at_least, to_brain_integrity, to_brain_level, toolclass_mask_for, WindowCounters,
};

const CONTROL_FRAME_DOMAIN: &str = "UCF:HASH:CONTROL_FRAME";
const CONTROL_FRAME_REASON_MAX_LEN: usize = 16;

#[derive(Debug, Clone)]
pub struct BaselineContext {
    pub cbv: Option<CharacterBaselineVector>,
    pub cbv_present: bool,
    pub pev: Option<PolicyEcologyVector>,
    pub pev_present: bool,
}

#[derive(Debug, Clone)]
pub struct ControlFrameContext {
    pub cbv: Option<CharacterBaselineVector>,
    pub cbv_digest: Option<Vec<u8>>,
    pub pev: Option<PolicyEcologyVector>,
    pub pev_digest: Option<Vec<u8>>,
    pub forensic_latched: bool,
}

pub fn brain_input_from_signal_frame(
    frame: &ucf::v1::SignalFrame,
    classified: &ClassifiedSignals,
    counters: &WindowCounters,
    rsv: &rsv::RsvState,
    baselines: BaselineContext,
    sc_replay_planned_present: bool,
    now_ms: u64,
) -> BrainInput {
    let normalized_frame = normalize_signal_frame(frame);
    let BaselineContext {
        cbv,
        cbv_present,
        pev,
        pev_present,
    } = baselines;

    let dopamin_input = build_dopa_input(
        &normalized_frame,
        classified,
        BrainLevel::Low,
        counters,
        to_brain_level(rsv.budget_stress),
    );

    BrainInput {
        now_ms,
        window_kind: classified.window_kind,
        hpa: build_hpa_input(classified, counters, rsv.unlock_ready),
        cbv,
        pev,
        lc: LcInput {
            integrity: to_brain_integrity(classified.integrity_state),
            receipt_invalid_count_short: counters.short_receipt_invalid_count,
            receipt_missing_count_short: counters.short_receipt_missing_count,
            dlp_critical_present_short: counters.short_dlp_critical_present,
            timeout_count_short: counters.short_timeout_count,
            deny_count_short: counters.short_deny_count,
            arousal_floor: BrainLevel::Low,
        },
        serotonin: SerInput {
            integrity: to_brain_integrity(classified.integrity_state),
            replay_mismatch_present: counters.medium_replay_mismatch,
            receipt_invalid_count_medium: counters.medium_receipt_invalid_count,
            dlp_critical_count_medium: counters.medium_dlp_critical_count,
            flapping_count_medium: counters.medium_flapping_count,
            unlock_present: rsv.unlock_ready,
            stability_floor: BrainLevel::Low,
        },
        amygdala: build_amygdala_input(
            &normalized_frame,
            classified,
            counters,
            to_brain_level(rsv.divergence),
        ),
        pag: PagInput {
            integrity: to_brain_integrity(classified.integrity_state),
            threat: BrainLevel::Low,
            vectors: Vec::new(),
            unlock_present: rsv.unlock_ready,
            stability: BrainLevel::Low,
            serotonin_cooldown: BrainCooldown::Base,
        },
        cerebellum: build_cerebellum_input(&normalized_frame, classified),
        stn: StnInput {
            policy_pressure: to_brain_level(classified.policy_pressure_class),
            arousal: BrainLevel::Low,
            threat: BrainLevel::Low,
            receipt_invalid_present: classified.receipt_invalid_count > 0,
            dlp_critical_present: counters.short_dlp_critical_present,
            integrity: to_brain_integrity(classified.integrity_state),
            tool_side_effects_present: false,
            cerebellum_divergence: BrainLevel::Low,
        },
        pmrf: PmrfInput {
            divergence: to_brain_level(rsv.divergence),
            policy_pressure: to_brain_level(classified.policy_pressure_class),
            stability: BrainLevel::Low,
            hold_active: false,
            budget_stress: to_brain_level(rsv.budget_stress),
        },
        dopamin: dopamin_input,
        insula: build_insula_input(
            &normalized_frame,
            classified,
            cbv_present,
            pev_present,
            BrainLevel::Low,
        ),
        sc_unlock_present: rsv.unlock_ready,
        sc_replay_planned_present,
        pprf_cooldown_class: BrainCooldown::Base,
    }
}

pub fn control_frame_from_brain_output(
    mut decision: ControlDecision,
    brain_output: &dbm_bus::BrainOutput,
    config: &RegulationConfig,
    context: ControlFrameContext,
) -> (ControlFrame, [u8; 32]) {
    let ControlFrameContext {
        cbv,
        cbv_digest,
        pev,
        pev_digest,
        forensic_latched,
    } = context;
    let last_baseline_vector = &brain_output.baseline;

    if level_at_least(last_baseline_vector.export_strictness, BrainLevel::Med) {
        decision.overlays.export_lock = true;
    }

    if level_at_least(last_baseline_vector.chain_conservatism, BrainLevel::High) {
        decision.overlays.simulate_first = true;
    }

    if level_at_least(last_baseline_vector.approval_strictness, BrainLevel::Med) {
        decision.approval_mode = config.character_baselines.strict_approval_mode.clone();
    }

    if level_at_least(last_baseline_vector.novelty_dampening, BrainLevel::Med) {
        decision.overlays.novelty_lock = true;
    }

    if cbv_digest.is_some()
        && config.character_baselines.cbv_influence_enabled
        && config.character_baselines.novelty_lock_on_cbv
    {
        decision.overlays.novelty_lock = true;
    }

    if cbv_digest.is_some() && config.character_baselines.cbv_influence_enabled {
        decision.approval_mode = config.character_baselines.strict_approval_mode.clone();
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

    let toolclass_mask_config = toolclass_mask_for(config, &decision, forensic_latched);

    let mut profile_reason_codes: Vec<i32> = decision
        .profile_reason_codes
        .iter()
        .map(|code| *code as i32)
        .collect();

    // Preserve all reason codes for callers while keeping a deterministic order.
    sanitize_reason_codes(&mut profile_reason_codes, usize::MAX);

    let mut control_frame = ControlFrame {
        active_profile: Some(profile),
        overlays: Some(overlays),
        toolclass_mask: Some(toolclass_mask_config),
        profile_reason_codes,
        control_frame_digest: None,
        character_epoch_digest: cbv_digest,
        policy_ecology_digest: pev_digest,
        approval_mode: Some(decision.approval_mode.clone()),
        deescalation_lock: decision.deescalation_lock.then_some(true),
        cooldown_class: Some(decision.cooldown_class as i32),
    };

    let digest_bytes = control_frame_digest(&control_frame);
    control_frame.control_frame_digest = Some(digest_bytes.to_vec());

    (control_frame, digest_bytes)
}

fn normalize_signal_frame(frame: &ucf::v1::SignalFrame) -> ucf::v1::SignalFrame {
    let mut normalized = frame.clone();
    normalized.reason_codes =
        normalized_reason_code_list(&frame.reason_codes, ReasonSet::DEFAULT_MAX_LEN);
    normalized.top_reason_codes =
        normalized_reason_code_list(&frame.top_reason_codes, ReasonSet::DEFAULT_MAX_LEN);

    normalize_policy_stats(
        normalized
            .policy_stats
            .get_or_insert_with(default_policy_stats),
    );
    normalize_exec_stats(normalized.exec_stats.get_or_insert_with(default_exec_stats));
    normalize_receipt_stats(
        normalized
            .receipt_stats
            .get_or_insert_with(default_receipt_stats),
    );

    normalized
}

fn normalize_policy_stats(stats: &mut PolicyStats) {
    stats.top_reason_codes =
        normalized_reason_code_list(&stats.top_reason_codes, ReasonSet::DEFAULT_MAX_LEN);
}

fn normalize_exec_stats(stats: &mut ExecStats) {
    stats.top_reason_codes =
        normalized_reason_code_list(&stats.top_reason_codes, ReasonSet::DEFAULT_MAX_LEN);
}

fn normalize_receipt_stats(stats: &mut ReceiptStats) {
    let _ = stats;
}

fn default_policy_stats() -> PolicyStats {
    PolicyStats {
        deny_count: 0,
        allow_count: 0,
        top_reason_codes: Vec::new(),
    }
}

fn default_exec_stats() -> ExecStats {
    ExecStats {
        timeout_count: 0,
        partial_failure_count: 0,
        tool_unavailable_count: 0,
        tool_id: None,
        dlp_block_count: 0,
        top_reason_codes: Vec::new(),
    }
}

fn default_receipt_stats() -> ReceiptStats {
    ReceiptStats {
        receipt_missing_count: 0,
        receipt_invalid_count: 0,
    }
}

fn normalized_reason_code_list(codes: &[i32], max_len: usize) -> Vec<i32> {
    let mut sanitized: Vec<i32> = codes
        .iter()
        .filter_map(|code| ReasonCode::try_from(*code).ok())
        .map(|code| code as i32)
        .collect();
    sanitize_reason_codes(&mut sanitized, max_len);
    sanitized
}

fn sanitize_reason_codes(codes: &mut Vec<i32>, max_len: usize) {
    codes.sort();
    codes.dedup();
    codes.truncate(max_len);
}

fn control_frame_digest(control_frame: &ControlFrame) -> [u8; 32] {
    let mut normalized = control_frame.clone();
    normalized.control_frame_digest = None;
    normalized.profile_reason_codes = normalized_reason_code_list(
        &normalized.profile_reason_codes,
        CONTROL_FRAME_REASON_MAX_LEN,
    );

    let mut buf = Vec::new();
    normalized.encode(&mut buf).unwrap();

    let mut hasher = Hasher::new_derive_key(CONTROL_FRAME_DOMAIN);
    hasher.update(&buf);
    *hasher.finalize().as_bytes()
}

fn normalized_reason_strings(codes: &[i32]) -> Vec<String> {
    normalized_reason_code_list(codes, ReasonSet::DEFAULT_MAX_LEN)
        .into_iter()
        .filter_map(|code| ReasonCode::try_from(code).ok())
        .map(|code| format!("{:?}", code))
        .collect()
}

fn build_insula_input(
    frame: &ucf::v1::SignalFrame,
    classified: &ClassifiedSignals,
    cbv_present: bool,
    pev_present: bool,
    progress: BrainLevel,
) -> InsulaInput {
    InsulaInput {
        policy_pressure: to_brain_level(classified.policy_pressure_class),
        receipt_failures: to_brain_level(classified.receipt_failures_class),
        receipt_invalid_present: classified.receipt_invalid_count > 0,
        exec_reliability: to_brain_level(classified.exec_reliability_class),
        integrity: to_brain_integrity(classified.integrity_state),
        timeout_burst: classified.exec_timeout_count > 0,
        cbv_present,
        pev_present,
        hbv_present: false,
        progress,
        dominant_reason_codes: normalized_reason_strings(&frame.reason_codes),
        arousal: BrainLevel::Low,
        stability: BrainLevel::Low,
        threat: BrainLevel::Low,
        threat_vectors: Vec::new(),
        pag_pattern: None,
        stn_hold_active: false,
    }
}

fn build_cerebellum_input(
    frame: &ucf::v1::SignalFrame,
    classified: &ClassifiedSignals,
) -> Option<CerInput> {
    if classified.window_kind != ucf::v1::WindowKind::Medium {
        return None;
    }

    let exec_stats = frame.exec_stats.as_ref();
    let mut tool_failures = Vec::new();
    if let Some(stats) = exec_stats {
        if let Some(tool_id) = &stats.tool_id {
            tool_failures.push((
                ToolKey::new(tool_id.clone(), String::new()),
                ToolFailureCounts {
                    timeouts: stats.timeout_count,
                    partial_failures: stats.partial_failure_count,
                    unavailable: stats.tool_unavailable_count,
                },
            ));
        }
    }
    Some(CerInput {
        timeout_count_medium: exec_stats.map(|stats| stats.timeout_count).unwrap_or(0),
        partial_failure_count_medium: exec_stats
            .map(|stats| stats.partial_failure_count)
            .unwrap_or(0),
        tool_unavailable_count_medium: exec_stats
            .map(|stats| stats.tool_unavailable_count)
            .unwrap_or(0),
        receipt_invalid_present: classified.receipt_invalid_count > 0,
        integrity: to_brain_integrity(classified.integrity_state),
        tool_id: exec_stats.and_then(|stats| stats.tool_id.clone()),
        dlp_block_count_medium: exec_stats.map(|stats| stats.dlp_block_count).unwrap_or(0),
        tool_failures,
    })
}

fn build_hpa_input(
    classified: &ClassifiedSignals,
    counters: &WindowCounters,
    unlock_ready: bool,
) -> HpaInput {
    use ucf::v1::{IntegrityStateClass, LevelClass, WindowKind};

    let stable_medium_window = classified.window_kind == WindowKind::Medium
        && classified.integrity_state == IntegrityStateClass::Ok
        && classified.receipt_invalid_count == 0
        && classified.replay_mismatch_class != LevelClass::High
        && classified.dlp_severity_class != LevelClass::High;

    HpaInput {
        integrity_state: to_brain_integrity(classified.integrity_state),
        replay_mismatch_present: counters.medium_replay_mismatch,
        dlp_critical_present: counters.medium_dlp_critical_count > 0,
        receipt_invalid_present: counters.medium_receipt_invalid_count > 0,
        deny_storm_present: classified.policy_deny_count >= 5,
        timeouts_burst_present: classified.exec_timeout_count >= 2,
        unlock_present: unlock_ready,
        stable_medium_window,
        calibrate_now: false,
    }
}

fn build_dopa_input(
    frame: &ucf::v1::SignalFrame,
    classified: &ClassifiedSignals,
    threat: BrainLevel,
    counters: &WindowCounters,
    budget_stress: BrainLevel,
) -> Option<DopaInput> {
    use ucf::v1::WindowKind;

    if classified.window_kind != WindowKind::Medium {
        return None;
    }

    let exec_stats = frame.exec_stats.as_ref();
    let exec_failure_count_medium = exec_stats
        .map(|stats| {
            stats
                .timeout_count
                .saturating_add(stats.partial_failure_count)
                .saturating_add(stats.tool_unavailable_count)
                .saturating_add(stats.dlp_block_count)
        })
        .unwrap_or(0);
    let exec_success_count_medium = frame
        .policy_stats
        .as_ref()
        .map(|stats| stats.allow_count)
        .unwrap_or(classified.policy_allow_count);

    Some(DopaInput {
        integrity: to_brain_integrity(classified.integrity_state),
        threat,
        policy_pressure: to_brain_level(classified.policy_pressure_class),
        receipt_invalid_present: classified.receipt_invalid_count > 0,
        dlp_critical_present: counters.medium_dlp_critical_count > 0,
        replay_mismatch_present: counters.medium_replay_mismatch,
        exec_success_count_medium,
        exec_failure_count_medium,
        deny_count_medium: classified.policy_deny_count,
        budget_stress,
        macro_finalized_count_long: 0,
    })
}

fn build_amygdala_input(
    frame: &ucf::v1::SignalFrame,
    classified: &ClassifiedSignals,
    counters: &WindowCounters,
    divergence: BrainLevel,
) -> AmyInput {
    let (dlp_secret_present, dlp_obfuscation_present, dlp_stegano_present) =
        crate::dlp_flags(frame);

    AmyInput {
        integrity: to_brain_integrity(classified.integrity_state),
        replay_mismatch_present: counters.medium_replay_mismatch,
        dlp_secret_present,
        dlp_obfuscation_present,
        dlp_stegano_present,
        dlp_critical_count_med: counters.medium_dlp_critical_count,
        receipt_invalid_medium: counters.medium_receipt_invalid_count,
        policy_pressure: to_brain_level(classified.policy_pressure_class),
        deny_storm_present: classified.policy_deny_count >= 5,
        sealed: Some(classified.integrity_state == IntegrityStateClass::Fail),
        tool_anomaly_present: false,
        tool_anomalies: Vec::new(),
        divergence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn control_frame_fixture(profile_reason_codes: Vec<i32>) -> ControlFrame {
        ControlFrame {
            active_profile: Some(ActiveProfile {
                profile: "M0".to_string(),
            }),
            overlays: Some(Overlays {
                simulate_first: false,
                export_lock: false,
                novelty_lock: false,
                chain_tightening: false,
            }),
            toolclass_mask: Some(ucf::v1::ToolClassMask {
                read: true,
                write: true,
                execute: true,
                transform: true,
                export: true,
            }),
            profile_reason_codes,
            control_frame_digest: None,
            character_epoch_digest: None,
            policy_ecology_digest: None,
            approval_mode: Some("strict".to_string()),
            deescalation_lock: Some(false),
            cooldown_class: Some(ucf::v1::LevelClass::Low as i32),
        }
    }

    #[test]
    fn control_frame_hash_normalizes_reason_codes() {
        let noisy_codes = vec![
            ReasonCode::RcGvOrientTargetIntegrity as i32,
            ReasonCode::RcGvDwmStabilize as i32,
            ReasonCode::RcGvDwmReport as i32,
            ReasonCode::RcGvDwmReport as i32,
            ReasonCode::RcGvDwmExecPlan as i32,
            ReasonCode::RcGvDwmSimulate as i32,
            ReasonCode::RcGvSequenceSplitRequired as i32,
            ReasonCode::RcGvOrientTargetRecovery as i32,
            ReasonCode::RcGvOrientTargetApproval as i32,
            ReasonCode::RcGvOrientTargetDlp as i32,
        ];

        let mut shuffled_codes = noisy_codes.clone();
        shuffled_codes.reverse();

        let frame_a = control_frame_fixture(noisy_codes);
        let frame_b = control_frame_fixture(shuffled_codes);

        let digest_a = control_frame_digest(&frame_a);
        let digest_b = control_frame_digest(&frame_b);

        assert_eq!(digest_a, digest_b);

        let normalized_codes = normalized_reason_code_list(
            &frame_a.profile_reason_codes,
            CONTROL_FRAME_REASON_MAX_LEN,
        );
        assert!(normalized_codes.len() <= CONTROL_FRAME_REASON_MAX_LEN);
        assert_eq!(
            normalized_codes,
            normalized_reason_code_list(
                &frame_b.profile_reason_codes,
                CONTROL_FRAME_REASON_MAX_LEN
            )
        );
    }
}
