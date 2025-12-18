#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use thiserror::Error;
use ucf::v1::{IntegrityStateClass, LevelClass, ReasonCode};

use crate::{OverlaySet, ProfileState};

const WINDOWING_FILE: &str = "windowing.yaml";
const CLASS_THRESHOLDS_FILE: &str = "class_thresholds.yaml";
const REGULATOR_PROFILES_FILE: &str = "regulator_profiles.yaml";
const REGULATOR_OVERLAYS_FILE: &str = "regulator_overlays.yaml";
const REGULATOR_UPDATE_TABLES_FILE: &str = "regulator_update_tables.yaml";
const BASELINE_MODIFIERS_FILE: &str = "baseline_modifiers.yaml";

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("failed to read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: serde_yaml::Error,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WindowSpec {
    pub min_records: u32,
    pub max_records: u32,
    pub max_age_ms: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WindowingConfig {
    pub short: WindowSpec,
    pub medium: WindowSpec,
    pub long: WindowSpec,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Thresholds {
    pub med: u32,
    pub high: u32,
}

impl Thresholds {
    pub fn classify(&self, value: u32) -> LevelClass {
        if value >= self.high {
            LevelClass::High
        } else if value >= self.med {
            LevelClass::Med
        } else {
            LevelClass::Low
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WindowThresholds {
    pub short: Thresholds,
    pub medium: Thresholds,
    pub long: Thresholds,
}

impl WindowThresholds {
    pub fn for_window_kind(&self, kind: ucf::v1::WindowKind) -> &Thresholds {
        match kind {
            ucf::v1::WindowKind::Short => &self.short,
            ucf::v1::WindowKind::Medium => &self.medium,
            ucf::v1::WindowKind::Long => &self.long,
            ucf::v1::WindowKind::Unknown => &self.short,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ThresholdConfig {
    pub policy_pressure: WindowThresholds,
    pub receipt_missing: WindowThresholds,
    pub receipt_invalid: WindowThresholds,
    pub exec_timeouts: WindowThresholds,
    pub dlp_events: Option<WindowThresholds>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GainProfileConfig {
    pub novelty: LevelClass,
    pub abstraction: LevelClass,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BudgetProfileConfig {
    pub k1: LevelClass,
    pub k2: LevelClass,
    pub k3: LevelClass,
    pub k4: LevelClass,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChainPolicyConfig {
    pub max_chain: u32,
    pub max_concurrency: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExportPolicyConfig {
    pub export_lock_default: bool,
    pub allowed_output_types: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToolClassMaskConfig {
    pub read: bool,
    pub write: bool,
    pub execute: bool,
    pub transform: bool,
    pub export: bool,
}

impl ToolClassMaskConfig {
    pub fn merge_overlay(&self, overlay: &ToolClassMaskOverlay) -> ToolClassMaskConfig {
        ToolClassMaskConfig {
            read: overlay.read.unwrap_or(self.read),
            write: overlay.write.unwrap_or(self.write),
            execute: overlay.execute.unwrap_or(self.execute),
            transform: overlay.transform.unwrap_or(self.transform),
            export: overlay.export.unwrap_or(self.export),
        }
    }

    pub fn to_tool_class_mask(&self) -> ucf::v1::ToolClassMask {
        ucf::v1::ToolClassMask {
            read: self.read,
            write: self.write,
            execute: self.execute,
            transform: self.transform,
            export: self.export,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeescalationConfig {
    pub lock: bool,
    pub cooldown_class: LevelClass,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileDefinition {
    pub approval_mode: String,
    pub gain_profile: GainProfileConfig,
    pub budgets: BudgetProfileConfig,
    pub chain_policy: ChainPolicyConfig,
    pub export_policy: ExportPolicyConfig,
    pub toolclass_mask: ToolClassMaskConfig,
    pub deescalation: DeescalationConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileSet {
    pub m0_research: ProfileDefinition,
    pub m1_restricted: ProfileDefinition,
    pub m2_quarantine: ProfileDefinition,
    pub m3_forensic: ProfileDefinition,
}

impl ProfileSet {
    pub fn get(&self, profile: ProfileState) -> &ProfileDefinition {
        match profile {
            ProfileState::M0Research => &self.m0_research,
            ProfileState::M1Restricted => &self.m1_restricted,
            ProfileState::M2Quarantine => &self.m2_quarantine,
            ProfileState::M3Forensic => &self.m3_forensic,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ToolClassMaskOverlay {
    pub read: Option<bool>,
    pub write: Option<bool>,
    pub execute: Option<bool>,
    pub transform: Option<bool>,
    pub export: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(deny_unknown_fields)]
pub struct OverlayEffect {
    pub toolclass_mask: Option<ToolClassMaskOverlay>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OverlayConfig {
    pub simulate_first: OverlayEffect,
    pub export_lock: OverlayEffect,
    pub novelty_lock: OverlayEffect,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CharacterBaselineConfig {
    #[serde(default)]
    pub cbv_influence_enabled: bool,
    #[serde(default = "default_strict_approval_mode")]
    pub strict_approval_mode: String,
    #[serde(default = "default_novelty_lock_on_cbv")]
    pub novelty_lock_on_cbv: bool,
}

impl Default for CharacterBaselineConfig {
    fn default() -> Self {
        CharacterBaselineConfig {
            cbv_influence_enabled: false,
            strict_approval_mode: default_strict_approval_mode(),
            novelty_lock_on_cbv: default_novelty_lock_on_cbv(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(deny_unknown_fields)]
pub struct RuleCondition {
    pub integrity_state: Option<IntegrityStateClass>,
    pub receipt_failures: Option<LevelClass>,
    pub policy_pressure: Option<LevelClass>,
    pub arousal: Option<LevelClass>,
    pub stability: Option<LevelClass>,
    pub divergence: Option<LevelClass>,
    pub budget_stress: Option<LevelClass>,
    pub missing_frame: Option<bool>,
    pub any: Option<bool>,
}

impl RuleCondition {
    pub fn matches(&self, signal: &ConditionSignals) -> bool {
        if self.any.unwrap_or(false) {
            return true;
        }

        if let Some(expected) = self.integrity_state {
            if signal.integrity_state != expected {
                return false;
            }
        }

        if let Some(expected) = self.receipt_failures {
            if !level_at_least(signal.receipt_failures, expected) {
                return false;
            }
        }

        if let Some(expected) = self.policy_pressure {
            if !level_at_least(signal.policy_pressure, expected) {
                return false;
            }
        }

        if let Some(expected) = self.arousal {
            if !level_at_least(signal.arousal, expected) {
                return false;
            }
        }

        if let Some(expected) = self.stability {
            if !level_at_least(signal.stability, expected) {
                return false;
            }
        }

        if let Some(expected) = self.divergence {
            if !level_at_least(signal.divergence, expected) {
                return false;
            }
        }

        if let Some(expected) = self.budget_stress {
            if !level_at_least(signal.budget_stress, expected) {
                return false;
            }
        }

        if let Some(missing_required) = self.missing_frame {
            if signal.missing_frame != missing_required {
                return false;
            }
        }

        true
    }
}

fn level_at_least(value: LevelClass, expected: LevelClass) -> bool {
    (value as i32) >= (expected as i32)
}

#[derive(Debug, Clone)]
pub struct ConditionSignals {
    pub integrity_state: IntegrityStateClass,
    pub receipt_failures: LevelClass,
    pub policy_pressure: LevelClass,
    pub arousal: LevelClass,
    pub stability: LevelClass,
    pub divergence: LevelClass,
    pub budget_stress: LevelClass,
    pub missing_frame: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileSwitchRule {
    pub name: String,
    #[serde(flatten)]
    pub conditions: RuleCondition,
    #[serde(deserialize_with = "crate::deserialize_profile_state")]
    pub set_profile: ProfileState,
    pub deescalation_lock: Option<bool>,
    #[serde(default)]
    pub reason_codes: Vec<ReasonCode>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OverlayEnableRule {
    pub name: String,
    #[serde(flatten)]
    pub conditions: RuleCondition,
    pub overlays: OverlaySet,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UpdateTablesConfig {
    pub profile_switch: Vec<ProfileSwitchRule>,
    pub overlay_enable: Vec<OverlayEnableRule>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RegulationConfig {
    pub windowing: WindowingConfig,
    pub thresholds: ThresholdConfig,
    pub profiles: ProfileSet,
    pub overlays: OverlayConfig,
    pub update_tables: UpdateTablesConfig,
    #[serde(default)]
    pub character_baselines: CharacterBaselineConfig,
}

impl RegulationConfig {
    pub fn load_from_dir<P: AsRef<Path>>(dir: P) -> Result<Self, ConfigError> {
        let base = dir.as_ref();
        Ok(RegulationConfig {
            windowing: load_file(base.join(WINDOWING_FILE))?,
            thresholds: load_file(base.join(CLASS_THRESHOLDS_FILE))?,
            profiles: load_file(base.join(REGULATOR_PROFILES_FILE))?,
            overlays: load_file(base.join(REGULATOR_OVERLAYS_FILE))?,
            update_tables: load_file(base.join(REGULATOR_UPDATE_TABLES_FILE))?,
            character_baselines: load_optional_file(base.join(BASELINE_MODIFIERS_FILE))?,
        })
    }

    pub fn fallback() -> Self {
        RegulationConfig {
            windowing: WindowingConfig {
                short: WindowSpec {
                    min_records: 1,
                    max_records: 1,
                    max_age_ms: 30_000,
                },
                medium: WindowSpec {
                    min_records: 1,
                    max_records: 1,
                    max_age_ms: 60_000,
                },
                long: WindowSpec {
                    min_records: 1,
                    max_records: 1,
                    max_age_ms: 120_000,
                },
            },
            thresholds: ThresholdConfig {
                policy_pressure: WindowThresholds {
                    short: Thresholds { med: 1, high: 1 },
                    medium: Thresholds { med: 1, high: 1 },
                    long: Thresholds { med: 1, high: 1 },
                },
                receipt_missing: WindowThresholds {
                    short: Thresholds { med: 1, high: 1 },
                    medium: Thresholds { med: 1, high: 1 },
                    long: Thresholds { med: 1, high: 1 },
                },
                receipt_invalid: WindowThresholds {
                    short: Thresholds { med: 1, high: 1 },
                    medium: Thresholds { med: 1, high: 1 },
                    long: Thresholds { med: 1, high: 1 },
                },
                exec_timeouts: WindowThresholds {
                    short: Thresholds { med: 1, high: 1 },
                    medium: Thresholds { med: 1, high: 1 },
                    long: Thresholds { med: 1, high: 1 },
                },
                dlp_events: None,
            },
            profiles: ProfileSet {
                m0_research: ProfileDefinition {
                    approval_mode: "monitor".to_string(),
                    gain_profile: GainProfileConfig {
                        novelty: LevelClass::Low,
                        abstraction: LevelClass::Low,
                    },
                    budgets: BudgetProfileConfig {
                        k1: LevelClass::Low,
                        k2: LevelClass::Low,
                        k3: LevelClass::Low,
                        k4: LevelClass::Low,
                    },
                    chain_policy: ChainPolicyConfig {
                        max_chain: 1,
                        max_concurrency: 1,
                    },
                    export_policy: ExportPolicyConfig {
                        export_lock_default: true,
                        allowed_output_types: Vec::new(),
                    },
                    toolclass_mask: ToolClassMaskConfig {
                        read: true,
                        write: false,
                        execute: false,
                        transform: true,
                        export: false,
                    },
                    deescalation: DeescalationConfig {
                        lock: true,
                        cooldown_class: LevelClass::High,
                    },
                },
                m1_restricted: ProfileDefinition {
                    approval_mode: "restricted".to_string(),
                    gain_profile: GainProfileConfig {
                        novelty: LevelClass::Low,
                        abstraction: LevelClass::Low,
                    },
                    budgets: BudgetProfileConfig {
                        k1: LevelClass::Low,
                        k2: LevelClass::Low,
                        k3: LevelClass::Low,
                        k4: LevelClass::Low,
                    },
                    chain_policy: ChainPolicyConfig {
                        max_chain: 1,
                        max_concurrency: 1,
                    },
                    export_policy: ExportPolicyConfig {
                        export_lock_default: true,
                        allowed_output_types: Vec::new(),
                    },
                    toolclass_mask: ToolClassMaskConfig {
                        read: true,
                        write: false,
                        execute: false,
                        transform: true,
                        export: false,
                    },
                    deescalation: DeescalationConfig {
                        lock: true,
                        cooldown_class: LevelClass::High,
                    },
                },
                m2_quarantine: ProfileDefinition {
                    approval_mode: "quarantine".to_string(),
                    gain_profile: GainProfileConfig {
                        novelty: LevelClass::Low,
                        abstraction: LevelClass::Low,
                    },
                    budgets: BudgetProfileConfig {
                        k1: LevelClass::Low,
                        k2: LevelClass::Low,
                        k3: LevelClass::Low,
                        k4: LevelClass::Low,
                    },
                    chain_policy: ChainPolicyConfig {
                        max_chain: 0,
                        max_concurrency: 0,
                    },
                    export_policy: ExportPolicyConfig {
                        export_lock_default: true,
                        allowed_output_types: Vec::new(),
                    },
                    toolclass_mask: ToolClassMaskConfig {
                        read: true,
                        write: false,
                        execute: false,
                        transform: false,
                        export: false,
                    },
                    deescalation: DeescalationConfig {
                        lock: true,
                        cooldown_class: LevelClass::High,
                    },
                },
                m3_forensic: ProfileDefinition {
                    approval_mode: "forensic".to_string(),
                    gain_profile: GainProfileConfig {
                        novelty: LevelClass::Low,
                        abstraction: LevelClass::Low,
                    },
                    budgets: BudgetProfileConfig {
                        k1: LevelClass::Low,
                        k2: LevelClass::Low,
                        k3: LevelClass::Low,
                        k4: LevelClass::Low,
                    },
                    chain_policy: ChainPolicyConfig {
                        max_chain: 0,
                        max_concurrency: 0,
                    },
                    export_policy: ExportPolicyConfig {
                        export_lock_default: true,
                        allowed_output_types: Vec::new(),
                    },
                    toolclass_mask: ToolClassMaskConfig {
                        read: true,
                        write: false,
                        execute: false,
                        transform: false,
                        export: false,
                    },
                    deescalation: DeescalationConfig {
                        lock: true,
                        cooldown_class: LevelClass::High,
                    },
                },
            },
            overlays: OverlayConfig {
                simulate_first: OverlayEffect {
                    toolclass_mask: None,
                },
                export_lock: OverlayEffect {
                    toolclass_mask: Some(ToolClassMaskOverlay {
                        export: Some(false),
                        ..Default::default()
                    }),
                },
                novelty_lock: OverlayEffect {
                    toolclass_mask: Some(ToolClassMaskOverlay {
                        transform: Some(false),
                        ..Default::default()
                    }),
                },
            },
            update_tables: UpdateTablesConfig {
                profile_switch: vec![ProfileSwitchRule {
                    name: "fallback".to_string(),
                    conditions: RuleCondition {
                        any: Some(true),
                        ..Default::default()
                    },
                    set_profile: ProfileState::M1Restricted,
                    deescalation_lock: Some(true),
                    reason_codes: vec![ReasonCode::ReIntegrityDegraded],
                }],
                overlay_enable: vec![OverlayEnableRule {
                    name: "fallback_overlays".to_string(),
                    conditions: RuleCondition {
                        any: Some(true),
                        ..Default::default()
                    },
                    overlays: OverlaySet {
                        simulate_first: true,
                        export_lock: true,
                        novelty_lock: true,
                    },
                }],
            },
            character_baselines: CharacterBaselineConfig::default(),
        }
    }
}

fn load_file<T: for<'de> Deserialize<'de>>(path: PathBuf) -> Result<T, ConfigError> {
    let reader = std::fs::File::open(&path).map_err(|source| ConfigError::Io {
        path: path.clone(),
        source,
    })?;
    serde_yaml::from_reader(reader).map_err(|source| ConfigError::Parse { path, source })
}

fn load_optional_file<T>(path: PathBuf) -> Result<T, ConfigError>
where
    T: for<'de> Deserialize<'de> + Default,
{
    if path.exists() {
        load_file(path)
    } else {
        Ok(T::default())
    }
}

fn default_strict_approval_mode() -> String {
    "STRICT".to_string()
}

fn default_novelty_lock_on_cbv() -> bool {
    true
}
