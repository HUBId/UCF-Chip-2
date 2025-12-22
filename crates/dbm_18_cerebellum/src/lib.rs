#![forbid(unsafe_code)]

use dbm_core::{DbmModule, IntegrityState, LevelClass, ReasonSet};
use std::collections::HashMap;

#[derive(Debug, Clone, Default)]
pub struct CerInput {
    pub timeout_count_medium: u32,
    pub partial_failure_count_medium: u32,
    pub tool_unavailable_count_medium: u32,
    pub receipt_invalid_present: bool,
    pub integrity: IntegrityState,
    pub tool_id: Option<String>,
    pub dlp_block_count_medium: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CerOutput {
    pub divergence: LevelClass,
    pub side_effect_suspected: bool,
    pub suspend_recommended: bool,
    pub reason_codes: ReasonSet,
}

impl Default for CerOutput {
    fn default() -> Self {
        Self {
            divergence: LevelClass::Low,
            side_effect_suspected: false,
            suspend_recommended: false,
            reason_codes: ReasonSet::default(),
        }
    }
}

#[derive(Debug, Default)]
pub struct Cerebellum {
    divergence: LevelClass,
    anomaly_counter_global: u32,
    anomaly_counter_per_tool: HashMap<String, u32>,
    cooldown_windows: u32,
}

impl Cerebellum {
    pub fn new() -> Self {
        Self::default()
    }

    fn stable_window(input: &CerInput) -> bool {
        input.timeout_count_medium == 0
            && input.partial_failure_count_medium == 0
            && input.tool_unavailable_count_medium == 0
            && !input.receipt_invalid_present
            && input.integrity == IntegrityState::Ok
            && input.dlp_block_count_medium == 0
    }

    fn determine_raw_divergence(input: &CerInput) -> LevelClass {
        if input.integrity != IntegrityState::Ok
            || input.tool_unavailable_count_medium >= 3
            || input.partial_failure_count_medium >= 5
            || input.timeout_count_medium >= 10
            || input.receipt_invalid_present
        {
            LevelClass::High
        } else if input.timeout_count_medium >= 2 || input.partial_failure_count_medium >= 1 {
            LevelClass::Med
        } else {
            LevelClass::Low
        }
    }

    fn update_tool_counter(&mut self, tool_id: &str) -> u32 {
        if self.anomaly_counter_per_tool.contains_key(tool_id) {
            if let Some(entry) = self.anomaly_counter_per_tool.get_mut(tool_id) {
                *entry = entry.saturating_add(1);
                return *entry;
            }
        }

        if self.anomaly_counter_per_tool.len() >= 32 {
            if let Some(max_key) = self.anomaly_counter_per_tool.keys().max().cloned() {
                self.anomaly_counter_per_tool.remove(&max_key);
            }
        }

        self.anomaly_counter_per_tool.insert(tool_id.to_string(), 1);
        1
    }
}

impl DbmModule for Cerebellum {
    type Input = CerInput;
    type Output = CerOutput;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        let mut output = CerOutput::default();
        let raw_divergence = Self::determine_raw_divergence(input);

        let divergence =
            if self.divergence == LevelClass::High && raw_divergence != LevelClass::High {
                if Self::stable_window(input) {
                    LevelClass::Low
                } else {
                    LevelClass::High
                }
            } else {
                raw_divergence
            };

        if divergence == LevelClass::High {
            self.cooldown_windows = self.cooldown_windows.saturating_add(1);
            self.anomaly_counter_global = self.anomaly_counter_global.saturating_add(1);
            output.reason_codes.insert("RC.GV.DIVERGENCE.HIGH");
        } else {
            self.cooldown_windows = 0;
        }

        let mut per_tool_count = 0;
        if let Some(tool_id) = &input.tool_id {
            if divergence == LevelClass::High {
                per_tool_count = self.update_tool_counter(tool_id);
            }
        }

        output.side_effect_suspected = divergence == LevelClass::High
            && (input.tool_unavailable_count_medium > 0 || input.partial_failure_count_medium > 0);

        output.suspend_recommended =
            divergence == LevelClass::High && (self.cooldown_windows >= 2 || per_tool_count >= 3);

        if output.suspend_recommended {
            output.reason_codes.insert("RC.GV.TOOL.SUSPEND_RECOMMENDED");
        }

        self.divergence = divergence;
        output.divergence = divergence;

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_input() -> CerInput {
        CerInput {
            integrity: IntegrityState::Ok,
            ..Default::default()
        }
    }

    #[test]
    fn timeouts_trigger_high() {
        let mut cerebellum = Cerebellum::new();
        let output = cerebellum.tick(&CerInput {
            timeout_count_medium: 10,
            ..base_input()
        });

        assert_eq!(output.divergence, LevelClass::High);
    }

    #[test]
    fn unavailable_trigger_high() {
        let mut cerebellum = Cerebellum::new();
        let output = cerebellum.tick(&CerInput {
            tool_unavailable_count_medium: 3,
            ..base_input()
        });

        assert_eq!(output.divergence, LevelClass::High);
    }

    #[test]
    fn stable_window_reduces_from_high() {
        let mut cerebellum = Cerebellum::new();
        let _ = cerebellum.tick(&CerInput {
            timeout_count_medium: 10,
            ..base_input()
        });

        let output = cerebellum.tick(&base_input());
        assert_eq!(output.divergence, LevelClass::Low);
    }

    #[test]
    fn repeated_high_recommends_suspend() {
        let mut cerebellum = Cerebellum::new();
        let input = CerInput {
            timeout_count_medium: 10,
            ..base_input()
        };

        let _ = cerebellum.tick(&input);
        let output = cerebellum.tick(&input);

        assert!(output.suspend_recommended);
        assert!(output
            .reason_codes
            .codes
            .iter()
            .any(|code| code == "RC.GV.TOOL.SUSPEND_RECOMMENDED"));
    }

    #[test]
    fn deterministic_tool_eviction() {
        let mut cerebellum = Cerebellum::new();

        for idx in 0..33 {
            let tool = format!("tool-{idx:02}");
            let _ = cerebellum.tick(&CerInput {
                timeout_count_medium: 10,
                tool_id: Some(tool.clone()),
                ..base_input()
            });
        }

        assert_eq!(cerebellum.anomaly_counter_per_tool.len(), 32);
        assert!(!cerebellum
            .anomaly_counter_per_tool
            .contains_key("tool-31"));
    }
}
