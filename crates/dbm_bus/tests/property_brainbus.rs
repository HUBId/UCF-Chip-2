use dbm_12_insula::InsulaInput;
use dbm_18_cerebellum::CerInput;
use dbm_6_dopamin_nacc::DopaInput;
use dbm_7_lc::LcInput;
use dbm_8_serotonin::SerInput;
use dbm_9_amygdala::AmyInput;
use dbm_bus::{BrainBus, BrainInput};
use dbm_core::limits::{MAX_LOCK_MS, MAX_REASON_CODES, MAX_SALIENCE_ITEMS, MAX_SUSPEND_RECS};
use dbm_core::{CooldownClass, IntegrityState, LevelClass, ProfileState, ThreatVector};
use dbm_hpa::{Hpa, HpaInput, HpaOutput};
use dbm_pag::PagInput;
use dbm_pmrf::PmrfInput;
use dbm_stn::StnInput;
use std::sync::atomic::{AtomicUsize, Ordering};
use ucf::v1::{CharacterBaselineVector, PolicyEcologyVector, WindowKind};

static HPA_STATE_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn make_bus() -> BrainBus {
    let id = HPA_STATE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let path =
        std::env::temp_dir().join(format!("hpa_state_test_{}_{}.json", std::process::id(), id));
    let _ = std::fs::remove_file(&path);
    BrainBus::with_hpa(Hpa::new(path), HpaOutput::default())
}

#[allow(clippy::too_many_arguments)]
fn make_input(
    now_ms: u64,
    deny_count_medium: u32,
    receipt_invalid_present: bool,
    dlp_critical_present: bool,
    integrity: IntegrityState,
    unlock_present: bool,
    tool_unavailable_count_medium: u32,
    timeout_count_medium: u32,
) -> BrainInput {
    BrainInput {
        now_ms,
        window_kind: WindowKind::Medium,
        hpa: HpaInput {
            integrity_state: integrity,
            replay_mismatch_present: false,
            dlp_critical_present,
            receipt_invalid_present,
            deny_storm_present: deny_count_medium >= 5,
            timeouts_burst_present: timeout_count_medium >= 10,
            unlock_present,
            stable_medium_window: false,
            calibrate_now: false,
        },
        cbv: Some(CharacterBaselineVector::default()),
        pev: Some(PolicyEcologyVector::default()),
        lc: LcInput {
            integrity,
            receipt_invalid_count_short: receipt_invalid_present as u32,
            receipt_missing_count_short: 0,
            dlp_critical_present_short: dlp_critical_present,
            timeout_count_short: timeout_count_medium.min(3),
            deny_count_short: deny_count_medium.min(3),
            arousal_floor: LevelClass::Low,
        },
        serotonin: SerInput {
            integrity,
            receipt_invalid_count_medium: receipt_invalid_present as u32,
            dlp_critical_count_medium: if dlp_critical_present { 2 } else { 0 },
            flapping_count_medium: 0,
            stability_floor: LevelClass::Low,
            unlock_present,
            ..Default::default()
        },
        amygdala: AmyInput {
            integrity,
            ..Default::default()
        },
        pag: PagInput {
            integrity,
            threat: if dlp_critical_present {
                LevelClass::High
            } else {
                LevelClass::Med
            },
            vectors: vec![ThreatVector::Exfil],
            unlock_present,
            stability: LevelClass::Med,
            serotonin_cooldown: CooldownClass::Base,
        },
        cerebellum: Some(CerInput {
            timeout_count_medium,
            partial_failure_count_medium: timeout_count_medium.saturating_sub(1),
            tool_unavailable_count_medium,
            receipt_invalid_present,
            integrity,
            tool_id: Some("tool-a".to_string()),
            dlp_block_count_medium: deny_count_medium,
            tool_failures: Vec::new(),
        }),
        stn: StnInput {
            integrity,
            ..Default::default()
        },
        pmrf: PmrfInput {
            divergence: if deny_count_medium >= 20 {
                LevelClass::High
            } else {
                LevelClass::Med
            },
            ..Default::default()
        },
        dopamin: Some(DopaInput {
            integrity,
            threat: if deny_count_medium >= 20 {
                LevelClass::High
            } else {
                LevelClass::Med
            },
            policy_pressure: LevelClass::Med,
            receipt_invalid_present,
            dlp_critical_present,
            replay_mismatch_present: false,
            exec_success_count_medium: 1,
            exec_failure_count_medium: timeout_count_medium,
            deny_count_medium,
            budget_stress: LevelClass::Med,
            macro_finalized_count_long: 0,
        }),
        insula: InsulaInput {
            integrity,
            receipt_invalid_present,
            policy_pressure: LevelClass::Med,
            receipt_failures: if receipt_invalid_present {
                LevelClass::Med
            } else {
                LevelClass::Low
            },
            exec_reliability: LevelClass::Med,
            timeout_burst: timeout_count_medium >= 10,
            ..Default::default()
        },
        sc_unlock_present: unlock_present,
        sc_replay_planned_present: receipt_invalid_present,
        pprf_cooldown_class: CooldownClass::Base,
    }
}

fn generated_inputs() -> Vec<BrainInput> {
    let mut inputs = Vec::new();
    let mut idx: u64 = 0;
    for deny_count_medium in [0, 5, 20] {
        for receipt_invalid_present in [false, true] {
            for dlp_critical_present in [false, true] {
                for integrity in [
                    IntegrityState::Ok,
                    IntegrityState::Degraded,
                    IntegrityState::Fail,
                ] {
                    for unlock_present in [false, true] {
                        for tool_unavailable_count_medium in [0, 3] {
                            for timeout_count_medium in [0, 10] {
                                if inputs.len() >= 50 {
                                    return inputs;
                                }
                                let now_ms = 10_000 + idx * 137;
                                idx += 1;
                                inputs.push(make_input(
                                    now_ms,
                                    deny_count_medium,
                                    receipt_invalid_present,
                                    dlp_critical_present,
                                    integrity,
                                    unlock_present,
                                    tool_unavailable_count_medium,
                                    timeout_count_medium,
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
    inputs
}

#[test]
fn brainbus_outputs_are_deterministic_and_bounded() {
    for (idx, input) in generated_inputs().into_iter().enumerate() {
        let mut bus_a = make_bus();
        let mut bus_b = make_bus();

        let output_a = bus_a.tick(input.clone());
        let output_b = bus_b.tick(input);
        assert_eq!(
            output_a.decision.profile_state,
            output_b.decision.profile_state
        );
        assert_eq!(output_a.dwm, output_b.dwm);
        assert_eq!(
            output_a.reason_codes.len(),
            output_b.reason_codes.len(),
            "reason code length mismatch at input #{idx}"
        );
        assert!(output_a
            .reason_codes
            .windows(2)
            .all(|pair| pair[0] <= pair[1]));
        assert!(output_b
            .reason_codes
            .windows(2)
            .all(|pair| pair[0] <= pair[1]));

        assert!(output_a.reason_codes.len() <= MAX_REASON_CODES);
        assert!(output_a.salience_items.len() <= MAX_SALIENCE_ITEMS);
        assert!(output_a.suspend_recommendations.len() <= MAX_SUSPEND_RECS);
    }
}

#[test]
fn profile_does_not_downgrade_after_fail_without_unlock() {
    let inputs = generated_inputs();
    let mut bus = make_bus();
    let mut fail_seen = false;

    for input in inputs {
        let unlock_ready = input.hpa.unlock_present || input.sc_unlock_present;
        let output = bus.tick(input);

        if fail_seen && !unlock_ready {
            assert_ne!(output.decision.profile_state, ProfileState::M0);
        }

        if output.isv.integrity == IntegrityState::Fail {
            fail_seen = true;
        }

        if unlock_ready {
            fail_seen = false;
        }
    }
}

#[test]
fn pprf_lock_is_bounded_in_generated_cases() {
    let mut bus = make_bus();
    let inputs = generated_inputs();

    for input in inputs.into_iter().take(8) {
        bus.tick(input.clone());
        let lock_cap = input.now_ms.saturating_add(MAX_LOCK_MS);
        assert!(bus.pprf_lock_until_ms() <= lock_cap);
    }
}
