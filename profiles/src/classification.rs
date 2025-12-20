#![forbid(unsafe_code)]

use crate::config::{ThresholdConfig, WindowThresholds};
use serde::{Deserialize, Serialize};
use ucf::v1::{
    ExecStats, IntegrityStateClass, LevelClass, PolicyStats, ReasonCode, ReceiptStats, SignalFrame,
    WindowKind,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClassifiedSignals {
    pub window_kind: WindowKind,
    pub integrity_state: IntegrityStateClass,
    pub policy_pressure_class: LevelClass,
    pub policy_deny_count: u32,
    pub policy_allow_count: u32,
    pub receipt_failures_class: LevelClass,
    pub receipt_missing_class: LevelClass,
    pub receipt_invalid_class: LevelClass,
    pub receipt_missing_count: u32,
    pub receipt_invalid_count: u32,
    pub dlp_severity_class: LevelClass,
    pub replay_mismatch_class: LevelClass,
    pub exec_reliability_class: LevelClass,
    pub exec_timeout_count: u32,
    pub missing_data: bool,
}

impl ClassifiedSignals {
    pub fn conservative(window_kind: WindowKind) -> Self {
        ClassifiedSignals {
            window_kind,
            integrity_state: IntegrityStateClass::Degraded,
            policy_pressure_class: LevelClass::High,
            policy_deny_count: 0,
            policy_allow_count: 0,
            receipt_failures_class: LevelClass::High,
            receipt_missing_class: LevelClass::High,
            receipt_invalid_class: LevelClass::High,
            receipt_missing_count: 0,
            receipt_invalid_count: 0,
            dlp_severity_class: LevelClass::High,
            replay_mismatch_class: LevelClass::High,
            exec_reliability_class: LevelClass::High,
            exec_timeout_count: 0,
            missing_data: true,
        }
    }
}

pub fn classify_signal_frame(frame: &SignalFrame, cfg: &ThresholdConfig) -> ClassifiedSignals {
    let window_kind = WindowKind::try_from(frame.window_kind).unwrap_or(WindowKind::Unknown);

    if window_kind == WindowKind::Unknown {
        return ClassifiedSignals::conservative(window_kind);
    }

    let mut missing_data = false;

    let (policy_pressure_class, policy_deny_count, policy_allow_count) = classify_policy_stats(
        frame.policy_stats.as_ref(),
        cfg.policy_pressure.for_window_kind(window_kind),
        &mut missing_data,
    );

    let (exec_reliability_class, exec_timeout_count) = classify_exec(
        frame.exec_stats.as_ref(),
        cfg.exec_timeouts.for_window_kind(window_kind),
        &mut missing_data,
    );

    let (
        receipt_failures_class,
        receipt_missing_class,
        receipt_invalid_class,
        receipt_missing_count,
        receipt_invalid_count,
    ) = classify_receipts(
        frame.receipt_stats.as_ref(),
        cfg.receipt_missing.for_window_kind(window_kind),
        cfg.receipt_invalid.for_window_kind(window_kind),
        &mut missing_data,
    );

    let integrity_state = IntegrityStateClass::try_from(frame.integrity_state)
        .unwrap_or(IntegrityStateClass::Degraded);

    let dlp_severity_class = classify_dlp(
        &frame.top_reason_codes,
        cfg.dlp_events.as_ref(),
        window_kind,
        &mut missing_data,
    );

    let replay_mismatch_class = if has_replay_mismatch(frame) {
        LevelClass::High
    } else {
        LevelClass::Low
    };

    ClassifiedSignals {
        window_kind,
        integrity_state,
        policy_pressure_class,
        policy_deny_count,
        policy_allow_count,
        receipt_failures_class,
        receipt_missing_class,
        receipt_invalid_class,
        receipt_missing_count,
        receipt_invalid_count,
        dlp_severity_class,
        replay_mismatch_class,
        exec_reliability_class,
        exec_timeout_count,
        missing_data,
    }
}

fn classify_policy_stats(
    stats: Option<&PolicyStats>,
    thresholds: &crate::config::Thresholds,
    missing: &mut bool,
) -> (LevelClass, u32, u32) {
    if let Some(stats) = stats {
        let deny = stats.deny_count;
        (thresholds.classify(deny), deny, stats.allow_count)
    } else {
        *missing = true;
        (LevelClass::High, 0, 0)
    }
}

fn classify_exec(
    stats: Option<&ExecStats>,
    thresholds: &crate::config::Thresholds,
    missing: &mut bool,
) -> (LevelClass, u32) {
    if let Some(stats) = stats {
        (
            thresholds.classify(stats.timeout_count),
            stats.timeout_count,
        )
    } else {
        *missing = true;
        (LevelClass::High, 0)
    }
}

fn classify_receipts(
    stats: Option<&ReceiptStats>,
    missing_thresholds: &crate::config::Thresholds,
    invalid_thresholds: &crate::config::Thresholds,
    missing_flag: &mut bool,
) -> (LevelClass, LevelClass, LevelClass, u32, u32) {
    if let Some(stats) = stats {
        let missing_class = missing_thresholds.classify(stats.receipt_missing_count);
        let invalid_class = invalid_thresholds.classify(stats.receipt_invalid_count);
        let failures_class = max_level(missing_class, invalid_class);
        (
            failures_class,
            missing_class,
            invalid_class,
            stats.receipt_missing_count,
            stats.receipt_invalid_count,
        )
    } else {
        *missing_flag = true;
        (LevelClass::High, LevelClass::High, LevelClass::High, 0, 0)
    }
}

fn classify_dlp(
    top_reason_codes: &[i32],
    thresholds: Option<&WindowThresholds>,
    window_kind: WindowKind,
    missing_flag: &mut bool,
) -> LevelClass {
    let dlp_hits = top_reason_codes
        .iter()
        .filter_map(|code| ReasonCode::try_from(*code).ok())
        .filter(|code| {
            matches!(
                code,
                ReasonCode::RcCdDlpSecretPattern
                    | ReasonCode::RcCdDlpObfuscation
                    | ReasonCode::RcCdDlpStegano
            )
        })
        .count() as u32;

    match thresholds {
        Some(thresholds) => thresholds.for_window_kind(window_kind).classify(dlp_hits),
        None => {
            if dlp_hits > 0 {
                LevelClass::High
            } else {
                *missing_flag = true;
                LevelClass::High
            }
        }
    }
}

fn has_replay_mismatch(frame: &SignalFrame) -> bool {
    let mismatch_code = ReasonCode::RcReReplayMismatch as i32;

    frame.top_reason_codes.contains(&mismatch_code)
        || frame.reason_codes.contains(&mismatch_code)
        || frame
            .policy_stats
            .as_ref()
            .map(|stats| stats.top_reason_codes.contains(&mismatch_code))
            .unwrap_or(false)
        || frame
            .exec_stats
            .as_ref()
            .map(|stats| stats.top_reason_codes.contains(&mismatch_code))
            .unwrap_or(false)
}

fn max_level(a: LevelClass, b: LevelClass) -> LevelClass {
    if (a as i32) >= (b as i32) {
        a
    } else {
        b
    }
}
