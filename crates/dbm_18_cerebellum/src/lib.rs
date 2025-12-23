#![forbid(unsafe_code)]

use dbm_core::{DbmModule, IntegrityState, LevelClass, ReasonSet, SuspendRecommendation, ToolKey};
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
    pub tool_failures: Vec<(ToolKey, ToolFailureCounts)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CerOutput {
    pub divergence: LevelClass,
    pub side_effect_suspected: bool,
    pub suspend_recommended: bool,
    pub tool_anomalies: Vec<(ToolKey, LevelClass)>,
    pub suspend_recommendations: Vec<SuspendRecommendation>,
    pub reason_codes: ReasonSet,
}

impl Default for CerOutput {
    fn default() -> Self {
        Self {
            divergence: LevelClass::Low,
            side_effect_suspected: false,
            suspend_recommended: false,
            tool_anomalies: Vec::new(),
            suspend_recommendations: Vec::new(),
            reason_codes: ReasonSet::default(),
        }
    }
}

#[derive(Debug, Default)]
pub struct Cerebellum {
    divergence: LevelClass,
    anomaly_counter_per_tool: HashMap<ToolKey, u32>,
}

#[derive(Debug, Clone, Default)]
pub struct ToolFailureCounts {
    pub timeouts: u32,
    pub partial_failures: u32,
    pub unavailable: u32,
}

impl ToolFailureCounts {
    #[allow(dead_code)]
    fn from_medium_window(timeouts: u32, partial_failures: u32, unavailable: u32) -> Self {
        Self {
            timeouts,
            partial_failures,
            unavailable,
        }
    }
}

impl Cerebellum {
    const MAX_TOOL_FAILURES: usize = 16;
    const MAX_TOOL_COUNTERS: usize = 32;
    const MAX_TOOL_ANOMALIES: usize = 16;
    const MAX_SUSPEND_RECOMMENDATIONS: usize = 8;

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

    fn determine_tool_anomaly(counts: &ToolFailureCounts) -> LevelClass {
        if counts.unavailable >= 1 || counts.partial_failures >= 3 || counts.timeouts >= 5 {
            LevelClass::High
        } else if counts.partial_failures >= 1 || counts.timeouts >= 2 {
            LevelClass::Med
        } else {
            LevelClass::Low
        }
    }

    fn update_tool_counter(&mut self, tool: &ToolKey, anomaly: LevelClass, stable_window: bool) {
        if !self.anomaly_counter_per_tool.contains_key(tool)
            && self.anomaly_counter_per_tool.len() >= Self::MAX_TOOL_COUNTERS
        {
            if let Some(evict) = self.anomaly_counter_per_tool.keys().max().cloned() {
                self.anomaly_counter_per_tool.remove(&evict);
            }
        }

        let entry = self
            .anomaly_counter_per_tool
            .entry(tool.clone())
            .or_insert(0);

        match anomaly {
            LevelClass::High => {
                *entry = entry.saturating_add(2);
            }
            LevelClass::Med => {
                *entry = entry.saturating_add(1);
            }
            LevelClass::Low => {
                if stable_window {
                    *entry = entry.saturating_sub(1);
                }
            }
        }
    }

    fn prepare_failures(
        tool_failures: &[(ToolKey, ToolFailureCounts)],
    ) -> Vec<(ToolKey, ToolFailureCounts)> {
        let mut failures: Vec<(ToolKey, ToolFailureCounts)> = tool_failures
            .iter()
            .map(|(key, counts)| (key.clone().normalized(), counts.clone()))
            .collect();
        failures.sort_by(|a, b| a.0.cmp(&b.0));
        failures.truncate(Self::MAX_TOOL_FAILURES);
        failures
    }

    fn build_recommendations(
        counters: &HashMap<ToolKey, u32>,
        anomalies: &[(ToolKey, LevelClass)],
    ) -> Vec<SuspendRecommendation> {
        let mut recommendations = Vec::new();

        for (tool, _) in anomalies.iter() {
            if let Some(count) = counters.get(tool) {
                let severity = if *count >= 6 {
                    LevelClass::High
                } else if *count >= 3 {
                    LevelClass::Med
                } else {
                    continue;
                };

                let mut reason_codes = ReasonSet::default();
                reason_codes.insert("RC.GV.TOOL.SUSPEND_RECOMMENDED");

                recommendations.push(SuspendRecommendation {
                    tool: tool.clone(),
                    severity,
                    reason_codes,
                });
            }
        }

        recommendations.sort_by(|a, b| {
            let sev_ord = level_severity(b.severity).cmp(&level_severity(a.severity));
            if sev_ord == std::cmp::Ordering::Equal {
                a.tool.cmp(&b.tool)
            } else {
                sev_ord
            }
        });
        recommendations.truncate(Self::MAX_SUSPEND_RECOMMENDATIONS);

        recommendations
    }
}

fn level_severity(level: LevelClass) -> u8 {
    match level {
        LevelClass::Low => 0,
        LevelClass::Med => 1,
        LevelClass::High => 2,
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
            output.reason_codes.insert("RC.GV.DIVERGENCE.HIGH");
        }

        let stable_window = Self::stable_window(input);
        let mut anomalies: Vec<(ToolKey, LevelClass)> = Vec::new();

        for (tool, counts) in Self::prepare_failures(&input.tool_failures) {
            let anomaly = Self::determine_tool_anomaly(&counts);
            self.update_tool_counter(&tool, anomaly, stable_window);
            if anomaly != LevelClass::Low {
                anomalies.push((tool, anomaly));
            }
        }

        anomalies.sort_by(|a, b| {
            let sev_ord = level_severity(b.1).cmp(&level_severity(a.1));
            if sev_ord == std::cmp::Ordering::Equal {
                a.0.cmp(&b.0)
            } else {
                sev_ord
            }
        });
        anomalies.truncate(Self::MAX_TOOL_ANOMALIES);

        let mut recommendations =
            Self::build_recommendations(&self.anomaly_counter_per_tool, &anomalies);

        // enrich recommendation reasons based on current anomalies
        for recommendation in &mut recommendations {
            if let Some((_, counts)) = input
                .tool_failures
                .iter()
                .find(|(tool, _)| tool == &recommendation.tool)
            {
                if counts.timeouts > 0 {
                    recommendation.reason_codes.insert("RC.GE.EXEC.TIMEOUT");
                }
                if counts.partial_failures > 0 {
                    recommendation
                        .reason_codes
                        .insert("RC.GE.EXEC.PARTIAL_FAILURE");
                }
            }
        }

        output.tool_anomalies = anomalies;
        output.suspend_recommendations = recommendations;
        output.suspend_recommended = !output.suspend_recommendations.is_empty();
        if output.suspend_recommended {
            output.reason_codes.insert("RC.GV.TOOL.SUSPEND_RECOMMENDED");
        }

        output.side_effect_suspected = output
            .tool_anomalies
            .iter()
            .any(|(_, level)| level == &LevelClass::High);

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

    fn tool_failure(
        tool: &str,
        timeouts: u32,
        partial_failures: u32,
    ) -> (ToolKey, ToolFailureCounts) {
        (
            ToolKey::new(tool.to_string(), "act".to_string()),
            ToolFailureCounts {
                timeouts,
                partial_failures,
                unavailable: 0,
            },
        )
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
    fn per_tool_counter_recommends_suspend() {
        let mut cerebellum = Cerebellum::new();
        let mut input = base_input();
        input.tool_failures = vec![tool_failure("tool-a", 5, 0)];

        let _ = cerebellum.tick(&input);
        let output = cerebellum.tick(&input);

        assert!(output.suspend_recommended);
        assert_eq!(output.suspend_recommendations[0].severity, LevelClass::Med);
        assert!(output.suspend_recommendations.iter().any(|rec| rec
            .reason_codes
            .codes
            .contains(&"RC.GE.EXEC.TIMEOUT".to_string())));
    }

    #[test]
    fn deterministic_tool_anomaly_ordering() {
        let mut cerebellum = Cerebellum::new();
        let mut input = base_input();
        input.tool_failures = vec![tool_failure("tool-b", 2, 0), tool_failure("tool-a", 2, 0)];

        let output = cerebellum.tick(&input);

        let ordered: Vec<String> = output
            .tool_anomalies
            .iter()
            .map(|(tool, _)| tool.tool_id.clone())
            .collect();
        assert_eq!(ordered, vec!["tool-a".to_string(), "tool-b".to_string()]);
    }

    #[test]
    fn bounded_tool_processing_and_recommendations() {
        let mut cerebellum = Cerebellum::new();
        let mut input = base_input();
        input.tool_failures = (0..50)
            .map(|idx| tool_failure(&format!("tool-{idx:02}"), 5, 0))
            .collect();

        for _ in 0..3 {
            let _ = cerebellum.tick(&input);
        }

        let output = cerebellum.tick(&input);

        assert!(output.tool_anomalies.len() <= Cerebellum::MAX_TOOL_ANOMALIES);
        assert_eq!(
            output.suspend_recommendations.len(),
            Cerebellum::MAX_SUSPEND_RECOMMENDATIONS
        );

        let tools: Vec<String> = output
            .suspend_recommendations
            .iter()
            .map(|rec| rec.tool.tool_id.clone())
            .collect();
        assert_eq!(tools[0], "tool-00".to_string());
    }
}
