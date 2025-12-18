#![forbid(unsafe_code)]

pub mod classification;
pub mod config;

use rsv::RsvState;
use serde::{de, Deserialize, Serialize};
use thiserror::Error;
use ucf::v1::{IntegrityStateClass, LevelClass, ReasonCode};

pub use classification::{classify_signal_frame, ClassifiedSignals};
pub use config::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ProfileState {
    M0Research,
    M1Restricted,
    M2Quarantine,
    M3Forensic,
}

impl<'de> Deserialize<'de> for ProfileState {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        deserialize_profile_state(deserializer)
    }
}

pub fn deserialize_profile_state<'de, D>(deserializer: D) -> Result<ProfileState, D::Error>
where
    D: de::Deserializer<'de>,
{
    let value: String = String::deserialize(deserializer)?;
    match value.to_ascii_uppercase().as_str() {
        "M0_RESEARCH" | "M0" => Ok(ProfileState::M0Research),
        "M1_RESTRICTED" | "M1" => Ok(ProfileState::M1Restricted),
        "M2_QUARANTINE" | "M2" => Ok(ProfileState::M2Quarantine),
        "M3_FORENSIC" | "M3" => Ok(ProfileState::M3Forensic),
        other => Err(de::Error::custom(format!(
            "unknown profile state: {}",
            other
        ))),
    }
}

impl ProfileState {
    pub fn as_str(&self) -> &'static str {
        match self {
            ProfileState::M0Research => "M0_RESEARCH",
            ProfileState::M1Restricted => "M1_RESTRICTED",
            ProfileState::M2Quarantine => "M2_QUARANTINE",
            ProfileState::M3Forensic => "M3_FORENSIC",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct OverlaySet {
    pub simulate_first: bool,
    pub export_lock: bool,
    pub novelty_lock: bool,
}

impl OverlaySet {
    pub fn merge(&self, other: &OverlaySet) -> OverlaySet {
        OverlaySet {
            simulate_first: self.simulate_first || other.simulate_first,
            export_lock: self.export_lock || other.export_lock,
            novelty_lock: self.novelty_lock || other.novelty_lock,
        }
    }

    pub fn all_enabled() -> OverlaySet {
        OverlaySet {
            simulate_first: true,
            export_lock: true,
            novelty_lock: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlDecision {
    pub profile: ProfileState,
    pub overlays: OverlaySet,
    pub deescalation_lock: bool,
    pub missing_frame_override: bool,
    pub profile_reason_codes: Vec<ReasonCode>,
}

#[derive(Debug, Error)]
pub enum DecisionError {
    #[error("no matching profile rule")]
    NoMatchingRule,
}

pub fn decide(
    rsv: &RsvState,
    now_ms: u64,
    config: &RegulationConfig,
) -> Result<ControlDecision, DecisionError> {
    let missing_frame = rsv
        .last_seen_frame_ts_ms
        .map(|ts| now_ms.saturating_sub(ts) > config.windowing.short.max_age_ms)
        .unwrap_or(true)
        || rsv.missing_frame_counter > 0
        || rsv.missing_data;

    let signals = ConditionSignals {
        integrity_state: rsv.integrity,
        receipt_failures: rsv.receipt_failures,
        policy_pressure: rsv.policy_pressure,
        arousal: rsv.arousal,
        stability: rsv.stability,
        divergence: rsv.divergence,
        budget_stress: rsv.budget_stress,
        missing_frame,
    };

    let mut overlays = OverlaySet::default();
    let mut profile: Option<ProfileState> = None;
    let mut deescalation_lock = false;
    let mut reason_codes = Vec::new();

    for rule in &config.update_tables.profile_switch {
        if rule.conditions.matches(&signals) {
            profile = Some(rule.set_profile);
            if let Some(lock) = rule.deescalation_lock {
                deescalation_lock = lock;
            }
            reason_codes.extend(rule.reason_codes.iter().copied());
            break;
        }
    }

    for rule in &config.update_tables.overlay_enable {
        if rule.conditions.matches(&signals) {
            overlays = overlays.merge(&rule.overlays);
        }
    }

    if missing_frame {
        reason_codes.push(ReasonCode::ReIntegrityDegraded);
        overlays = overlays.merge(&OverlaySet::all_enabled());
        deescalation_lock = true;
    }

    let profile = profile.ok_or(DecisionError::NoMatchingRule)?;

    Ok(ControlDecision {
        profile,
        overlays,
        deescalation_lock,
        missing_frame_override: missing_frame,
        profile_reason_codes: reason_codes,
    })
}

pub fn decide_with_fallback(
    rsv: &RsvState,
    now_ms: u64,
    config: &RegulationConfig,
) -> ControlDecision {
    decide(rsv, now_ms, config).unwrap_or_else(|_| ControlDecision {
        profile: ProfileState::M1Restricted,
        overlays: OverlaySet::all_enabled(),
        deescalation_lock: true,
        missing_frame_override: true,
        profile_reason_codes: vec![ReasonCode::ReIntegrityDegraded],
    })
}

pub fn apply_classification(rsv: &mut RsvState, classified: &ClassifiedSignals, timestamp_ms: u64) {
    rsv.last_seen_frame_ts_ms = Some(timestamp_ms);
    rsv.missing_frame_counter = 0;
    rsv.integrity = classified.integrity_state;
    rsv.policy_pressure = classified.policy_pressure_class;
    rsv.threat = classified.dlp_severity_class;
    rsv.arousal = classified.exec_reliability_class;
    rsv.stability = if classified.integrity_state != IntegrityStateClass::Ok {
        LevelClass::High
    } else {
        classified.exec_reliability_class
    };
    rsv.receipt_failures = classified.receipt_failures_class;
    rsv.receipt_missing_count_window = classified.receipt_missing_count;
    rsv.receipt_invalid_count_window = classified.receipt_invalid_count;
    rsv.missing_data = classified.missing_data;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use ucf::v1::{ExecStats, LevelClass, PolicyStats, ReceiptStats, SignalFrame, WindowKind};

    fn workspace_config_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("config")
    }

    #[test]
    fn config_files_load() {
        let config = RegulationConfig::load_from_dir(workspace_config_dir()).unwrap();
        assert_eq!(config.windowing.short.max_records, 50);
        assert!(config.profiles.m0_research.toolclass_mask.export);
        assert!(!config.character_baselines.cbv_influence_enabled);
    }

    #[test]
    fn missing_required_field_fails_parse() {
        let base_dir = workspace_config_dir();
        let tmp_dir = base_dir
            .parent()
            .unwrap()
            .join("target")
            .join(format!("config_invalid_{}", std::process::id()));

        if tmp_dir.exists() {
            let _ = fs::remove_dir_all(&tmp_dir);
        }

        fs::create_dir_all(&tmp_dir).unwrap();

        fs::write(
            tmp_dir.join("windowing.yaml"),
            "short:\n  min_records: 1\n  max_records: 1\nmedium:\n  min_records: 1\n  max_records: 1\n  max_age_ms: 1\nlong:\n  min_records: 1\n  max_records: 1\n  max_age_ms: 1\n",
        )
        .unwrap();
        for file in [
            "class_thresholds.yaml",
            "regulator_profiles.yaml",
            "regulator_overlays.yaml",
            "regulator_update_tables.yaml",
        ] {
            fs::copy(base_dir.join(file), tmp_dir.join(file)).unwrap();
        }

        let result = RegulationConfig::load_from_dir(&tmp_dir);
        assert!(result.is_err());
    }

    #[test]
    fn classification_uses_thresholds() {
        let config = RegulationConfig::load_from_dir(workspace_config_dir()).unwrap();
        let frame = SignalFrame {
            window_kind: WindowKind::Short as i32,
            window_index: Some(1),
            timestamp_ms: Some(1),
            policy_stats: Some(PolicyStats {
                deny_count: 6,
                allow_count: 0,
            }),
            exec_stats: Some(ExecStats { timeout_count: 0 }),
            receipt_stats: Some(ReceiptStats {
                receipt_missing_count: 0,
                receipt_invalid_count: 0,
            }),
            ..SignalFrame::default()
        };

        let classified = classify_signal_frame(&frame, &config.thresholds);
        assert_eq!(classified.policy_pressure_class, LevelClass::High);
        assert!(!classified.missing_data);
    }

    #[test]
    fn rules_drive_profiles_and_overlays() {
        let config = RegulationConfig::load_from_dir(workspace_config_dir()).unwrap();
        let mut rsv = RsvState::default();
        let frame = SignalFrame {
            window_kind: WindowKind::Short as i32,
            window_index: Some(1),
            timestamp_ms: Some(1),
            policy_stats: Some(PolicyStats {
                deny_count: 0,
                allow_count: 1,
            }),
            exec_stats: Some(ExecStats { timeout_count: 0 }),
            receipt_stats: Some(ReceiptStats {
                receipt_missing_count: 0,
                receipt_invalid_count: 2,
            }),
            ..SignalFrame::default()
        };

        let classified = classify_signal_frame(&frame, &config.thresholds);
        apply_classification(&mut rsv, &classified, 1);
        let decision = decide_with_fallback(&rsv, 1, &config);

        assert_eq!(decision.profile, ProfileState::M1Restricted);
        assert!(decision.overlays.export_lock);
        assert!(decision.overlays.novelty_lock);
    }
}
