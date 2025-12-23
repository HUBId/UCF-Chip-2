#![forbid(unsafe_code)]

use dbm_core::{DbmModule, IntegrityState, LevelClass, ReasonSet};
#[cfg(feature = "microcircuit-lc")]
use microcircuit_core::CircuitConfig;
use microcircuit_core::MicrocircuitBackend;
pub use microcircuit_lc_stub::{LcInput, LcOutput};
use std::fmt;

#[derive(Debug, Default)]
pub struct LcRules;

impl LcRules {
    fn severity(level: LevelClass) -> u8 {
        match level {
            LevelClass::Low => 0,
            LevelClass::Med => 1,
            LevelClass::High => 2,
        }
    }

    fn tick(&mut self, input: &LcInput) -> LcOutput {
        let mut reason_codes = ReasonSet::default();

        let mut arousal = if input.integrity != IntegrityState::Ok
            || input.receipt_invalid_count_short >= 1
            || input.dlp_critical_present_short
            || input.timeout_count_short >= 2
        {
            reason_codes.insert("lc_high_trigger");
            LevelClass::High
        } else if input.deny_count_short >= 2
            || input.receipt_missing_count_short >= 1
            || input.timeout_count_short == 1
        {
            reason_codes.insert("lc_med_trigger");
            LevelClass::Med
        } else {
            LevelClass::Low
        };

        if Self::severity(arousal) < Self::severity(input.arousal_floor) {
            arousal = input.arousal_floor;
            reason_codes.insert("lc_floor");
        }

        let (hint_simulate_first, hint_novelty_lock) = match arousal {
            LevelClass::High => (true, true),
            LevelClass::Med => (true, false),
            LevelClass::Low => (false, false),
        };

        LcOutput {
            arousal,
            hint_simulate_first,
            hint_novelty_lock,
            reason_codes,
        }
    }
}

pub enum LcBackend {
    Rules(LcRules),
    Micro(Box<dyn MicrocircuitBackend<LcInput, LcOutput>>),
}

impl fmt::Debug for LcBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LcBackend::Rules(_) => f.write_str("LcBackend::Rules"),
            LcBackend::Micro(_) => f.write_str("LcBackend::Micro"),
        }
    }
}

impl LcBackend {
    fn tick(&mut self, input: &LcInput) -> LcOutput {
        match self {
            LcBackend::Rules(rules) => rules.tick(input),
            LcBackend::Micro(backend) => backend.step(input, 0),
        }
    }
}

#[derive(Debug)]
pub struct Lc {
    backend: LcBackend,
}

impl Lc {
    pub fn new() -> Self {
        Self {
            backend: LcBackend::Rules(LcRules),
        }
    }

    #[cfg(feature = "microcircuit-lc-spike")]
    pub fn new_micro(config: CircuitConfig) -> Self {
        use microcircuit_lc_spike::LcMicrocircuit;

        Self {
            backend: LcBackend::Micro(Box::new(LcMicrocircuit::new(config))),
        }
    }

    #[cfg(all(feature = "microcircuit-lc", not(feature = "microcircuit-lc-spike")))]
    pub fn new_micro(config: CircuitConfig) -> Self {
        use microcircuit_lc_stub::LcMicrocircuit;

        Self {
            backend: LcBackend::Micro(Box::new(LcMicrocircuit::new(config))),
        }
    }

    pub fn snapshot_digest(&self) -> Option<[u8; 32]> {
        match &self.backend {
            LcBackend::Micro(backend) => Some(backend.snapshot_digest()),
            LcBackend::Rules(_) => None,
        }
    }
}

impl Default for Lc {
    fn default() -> Self {
        Self::new()
    }
}

impl DbmModule for Lc {
    type Input = LcInput;
    type Output = LcOutput;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        self.backend.tick(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_input() -> LcInput {
        LcInput {
            integrity: IntegrityState::Ok,
            arousal_floor: LevelClass::Low,
            ..Default::default()
        }
    }

    #[test]
    fn receipt_invalid_forces_high() {
        let mut lc = Lc::new();
        let output = lc.tick(&LcInput {
            receipt_invalid_count_short: 1,
            ..base_input()
        });

        assert_eq!(output.arousal, LevelClass::High);
        assert!(output.hint_simulate_first);
        assert!(output.hint_novelty_lock);
    }

    #[test]
    fn timeout_burst_forces_high() {
        let mut lc = Lc::new();
        let output = lc.tick(&LcInput {
            timeout_count_short: 2,
            ..base_input()
        });

        assert_eq!(output.arousal, LevelClass::High);
    }

    #[test]
    fn deny_threshold_sets_med() {
        let mut lc = Lc::new();
        let output = lc.tick(&LcInput {
            deny_count_short: 2,
            ..base_input()
        });

        assert_eq!(output.arousal, LevelClass::Med);
        assert!(output.hint_simulate_first);
        assert!(!output.hint_novelty_lock);
    }

    #[test]
    fn floor_applies() {
        let mut lc = Lc::new();
        let output = lc.tick(&LcInput {
            arousal_floor: LevelClass::Med,
            ..base_input()
        });

        assert_eq!(output.arousal, LevelClass::Med);
        assert!(output.reason_codes.codes.contains(&"lc_floor".to_string()));
    }

    #[cfg(feature = "microcircuit-lc")]
    #[test]
    fn micro_backend_matches_rules() {
        use microcircuit_lc_stub::LcMicrocircuit;

        let mut rules = Lc::new();
        let mut micro = Lc {
            backend: LcBackend::Micro(Box::new(LcMicrocircuit::new(CircuitConfig::default()))),
        };

        let cases = [
            LcInput {
                deny_count_short: 3,
                timeout_count_short: 1,
                arousal_floor: LevelClass::Low,
                ..Default::default()
            },
            LcInput {
                integrity: IntegrityState::Fail,
                receipt_missing_count_short: 1,
                arousal_floor: LevelClass::Med,
                ..Default::default()
            },
        ];

        for input in cases {
            let rules_out = rules.tick(&input);
            let micro_out = micro.tick(&input);

            assert_eq!(rules_out, micro_out);
        }
    }

    #[cfg(feature = "microcircuit-lc-spike")]
    #[test]
    fn receipt_invalid_invariant_holds() {
        let mut lc = Lc::new_micro(CircuitConfig::default());
        let output = lc.tick(&LcInput {
            receipt_invalid_count_short: 1,
            ..base_input()
        });

        assert!(matches!(output.arousal, LevelClass::Med | LevelClass::High));
        assert!(output.hint_simulate_first);
    }

    #[cfg(feature = "microcircuit-lc-spike")]
    #[test]
    fn integrity_fail_forces_high() {
        let mut lc = Lc::new_micro(CircuitConfig::default());
        let output = lc.tick(&LcInput {
            integrity: IntegrityState::Fail,
            ..base_input()
        });

        assert_eq!(output.arousal, LevelClass::High);
    }

    #[cfg(feature = "microcircuit-lc-spike")]
    #[test]
    fn micro_is_no_less_conservative_than_rules_for_critical_inputs() {
        fn severity(level: LevelClass) -> u8 {
            match level {
                LevelClass::Low => 0,
                LevelClass::Med => 1,
                LevelClass::High => 2,
            }
        }

        let mut rules = Lc::new();
        let mut micro = Lc::new_micro(CircuitConfig::default());

        let cases = [
            LcInput {
                integrity: IntegrityState::Degraded,
                arousal_floor: LevelClass::Low,
                ..Default::default()
            },
            LcInput {
                receipt_invalid_count_short: 2,
                arousal_floor: LevelClass::Low,
                ..Default::default()
            },
            LcInput {
                dlp_critical_present_short: true,
                arousal_floor: LevelClass::Med,
                ..Default::default()
            },
            LcInput {
                timeout_count_short: 2,
                arousal_floor: LevelClass::Low,
                ..Default::default()
            },
            LcInput {
                deny_count_short: 2,
                receipt_missing_count_short: 1,
                arousal_floor: LevelClass::Low,
                ..Default::default()
            },
        ];

        for input in cases {
            let rules_out = rules.tick(&input);
            let micro_out = micro.tick(&input);

            assert!(
                severity(micro_out.arousal) >= severity(rules_out.arousal),
                "micro arousal {:?} was less than rules {:?}",
                micro_out.arousal,
                rules_out.arousal
            );
            if rules_out.hint_simulate_first {
                assert!(micro_out.hint_simulate_first);
            }
        }
    }
}
