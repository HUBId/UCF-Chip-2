use biophys_core::ModulatorField;
use dbm_12_insula::InsulaInput;
use dbm_18_cerebellum::CerInput;
use dbm_6_dopamin_nacc::DopaInput;
use dbm_7_lc::LcInput;
use dbm_8_serotonin::SerInput;
use dbm_9_amygdala::AmyInput;
use dbm_bus::{BrainBus, BrainInput};
use dbm_core::{CooldownClass, IntegrityState, LevelClass, ReasonSet, ThreatVector};
use dbm_hpa::HpaInput;
use dbm_pag::PagInput;
use dbm_pmrf::PmrfInput;
use dbm_stn::StnInput;
use ucf::v1::{CharacterBaselineVector, PolicyEcologyVector, WindowKind};

fn brain_bus_with_clean_state(label: &str) -> BrainBus {
    let path = std::env::temp_dir().join(format!(
        "brain_bus_hpa_state_{}_{}_{}.json",
        std::process::id(),
        label,
        7
    ));
    let _ = std::fs::remove_file(&path);
    BrainBus::with_hpa(dbm_hpa::Hpa::new(path), Default::default())
}

fn canonical_brain_input() -> BrainInput {
    BrainInput {
        now_ms: 10_000,
        window_kind: WindowKind::Medium,
        hpa: HpaInput {
            integrity_state: IntegrityState::Fail,
            replay_mismatch_present: true,
            dlp_critical_present: true,
            receipt_invalid_present: true,
            deny_storm_present: true,
            timeouts_burst_present: true,
            unlock_present: false,
            stable_medium_window: false,
            calibrate_now: false,
        },
        cbv: Some(CharacterBaselineVector::default()),
        pev: Some(PolicyEcologyVector::default()),
        lc: LcInput {
            integrity: IntegrityState::Fail,
            receipt_invalid_count_short: 2,
            receipt_missing_count_short: 1,
            dlp_critical_present_short: true,
            timeout_count_short: 2,
            deny_count_short: 3,
            arousal_floor: LevelClass::Med,
            modulators: ModulatorField::default(),
        },
        serotonin: SerInput {
            stability_floor: LevelClass::Med,
            ..Default::default()
        },
        amygdala: AmyInput {
            integrity: IntegrityState::Fail,
            ..Default::default()
        },
        pag: PagInput {
            integrity: IntegrityState::Fail,
            threat: LevelClass::High,
            vectors: vec![ThreatVector::Exfil],
            unlock_present: false,
            stability: LevelClass::Med,
            serotonin_cooldown: CooldownClass::Base,
            modulators: ModulatorField::default(),
        },
        cerebellum: Some(CerInput {
            timeout_count_medium: 3,
            partial_failure_count_medium: 2,
            tool_unavailable_count_medium: 1,
            receipt_invalid_present: true,
            integrity: IntegrityState::Fail,
            tool_id: Some("tool-a".to_string()),
            dlp_block_count_medium: 1,
            tool_failures: Vec::new(),
        }),
        stn: StnInput {
            integrity: IntegrityState::Fail,
            modulators: ModulatorField::default(),
            ..Default::default()
        },
        pmrf: PmrfInput {
            divergence: LevelClass::High,
            modulators: ModulatorField::default(),
            ..Default::default()
        },
        dopamin: Some(DopaInput {
            integrity: IntegrityState::Fail,
            threat: LevelClass::High,
            policy_pressure: LevelClass::High,
            receipt_invalid_present: true,
            dlp_critical_present: true,
            replay_mismatch_present: true,
            exec_success_count_medium: 1,
            exec_failure_count_medium: 3,
            deny_count_medium: 4,
            budget_stress: LevelClass::High,
            macro_finalized_count_long: 2,
        }),
        insula: InsulaInput {
            policy_pressure: LevelClass::High,
            receipt_failures: LevelClass::High,
            receipt_invalid_present: true,
            exec_reliability: LevelClass::High,
            integrity: IntegrityState::Fail,
            timeout_burst: true,
            ..Default::default()
        },
        sc_unlock_present: false,
        sc_replay_planned_present: true,
        pprf_cooldown_class: CooldownClass::Longer,
        trace_fail_present: false,
        trace_pass_present: false,
        trace_fail_streak: 0,
    }
}

#[test]
fn tick_outputs_stable_and_reason_codes_sorted() {
    let input = canonical_brain_input();
    let mut brain_a = brain_bus_with_clean_state("a");
    let mut brain_b = brain_bus_with_clean_state("b");

    let output_a = brain_a.tick(input.clone());
    let output_b = brain_b.tick(input);

    assert_eq!(output_a.hpa, output_b.hpa);
    assert_eq!(output_a.baseline, output_b.baseline);
    assert_eq!(output_a.decision, output_b.decision);
    assert_eq!(output_a.dwm, output_b.dwm);
    assert_eq!(output_a.focus_target, output_b.focus_target);

    assert_eq!(output_a.reason_codes, output_b.reason_codes);
    assert!(output_a
        .reason_codes
        .windows(2)
        .all(|pair| pair[0] <= pair[1]));
    assert!(output_a.reason_codes.len() <= ReasonSet::DEFAULT_MAX_LEN);

    let expected = if output_a
        .reason_codes
        .iter()
        .any(|code| code == "RC.CD.DLP.EXPORT_BLOCKED")
    {
        vec![
            "RC.CD.DLP.EXPORT_BLOCKED".to_string(),
            "RC.GV.DIVERGENCE.HIGH".to_string(),
            "RC.GV.DWM.REPORT".to_string(),
            "RC.GV.FOCUS_SHIFT.EXECUTED".to_string(),
            "RC.GV.HOLD.ON".to_string(),
            "RC.GV.ORIENT.TARGET_INTEGRITY".to_string(),
            "RC.GV.PROGRESS.REWARD_BLOCKED".to_string(),
            "RC.GV.SEQUENCE.SPLIT_REQUIRED".to_string(),
        ]
    } else if output_a
        .reason_codes
        .iter()
        .any(|code| code == "RC.RE.INTEGRITY.DEGRADED/FAIL")
    {
        vec![
            "RC.GV.DIVERGENCE.HIGH".to_string(),
            "RC.GV.DWM.REPORT".to_string(),
            "RC.GV.FOCUS_SHIFT.EXECUTED".to_string(),
            "RC.GV.HOLD.ON".to_string(),
            "RC.GV.ORIENT.TARGET_INTEGRITY".to_string(),
            "RC.GV.PROGRESS.REWARD_BLOCKED".to_string(),
            "RC.GV.SEQUENCE.SPLIT_REQUIRED".to_string(),
            "RC.RE.INTEGRITY.DEGRADED/FAIL".to_string(),
        ]
    } else if output_a
        .reason_codes
        .iter()
        .any(|code| code == "RC.RE.INTEGRITY.FAIL")
    {
        vec![
            "RC.GV.DIVERGENCE.HIGH".to_string(),
            "RC.GV.DWM.REPORT".to_string(),
            "RC.GV.FOCUS_SHIFT.EXECUTED".to_string(),
            "RC.GV.HOLD.ON".to_string(),
            "RC.GV.ORIENT.TARGET_INTEGRITY".to_string(),
            "RC.GV.PROGRESS.REWARD_BLOCKED".to_string(),
            "RC.GV.SEQUENCE.SPLIT_REQUIRED".to_string(),
            "RC.RE.INTEGRITY.FAIL".to_string(),
        ]
    } else {
        vec![
            "RC.GV.DIVERGENCE.HIGH".to_string(),
            "RC.GV.DWM.REPORT".to_string(),
            "RC.GV.FOCUS_SHIFT.EXECUTED".to_string(),
            "RC.GV.HOLD.ON".to_string(),
            "RC.GV.ORIENT.TARGET_INTEGRITY".to_string(),
            "RC.GV.PROGRESS.REWARD_BLOCKED".to_string(),
            "RC.GV.SEQUENCE.SPLIT_REQUIRED".to_string(),
            "RC.RG.STATE.AROUSAL_UP".to_string(),
        ]
    };
    assert_eq!(output_a.reason_codes, expected);
}
