use baseline_resolver::{resolve_baseline, BaselineInputs, HbvOffsets};
use dbm_core::{IntegrityState, LevelClass};
use dbm_hpa::{Hpa, HpaInput};
use engine::RegulationEngine;
use pvgs_client::MockPvgsReader;
use std::time::{SystemTime, UNIX_EPOCH};
use ucf::v1::{
    CharacterBaselineVector, ExecStats, IntegrityStateClass, PolicyEcologyVector, PolicyStats,
    ReasonCode, ReceiptStats, SignalFrame, WindowKind,
};

fn medium_frame() -> SignalFrame {
    SignalFrame {
        window_kind: WindowKind::Medium as i32,
        window_index: Some(42),
        timestamp_ms: None,
        policy_stats: Some(PolicyStats {
            deny_count: 6,
            allow_count: 0,
            top_reason_codes: vec![],
        }),
        exec_stats: Some(ExecStats {
            timeout_count: 3,
            partial_failure_count: 0,
            tool_unavailable_count: 0,
            tool_id: None,
            dlp_block_count: 1,
            top_reason_codes: vec![],
        }),
        integrity_state: IntegrityStateClass::Fail as i32,
        top_reason_codes: vec![],
        signal_frame_digest: None,
        receipt_stats: Some(ReceiptStats {
            receipt_missing_count: 0,
            receipt_invalid_count: 1,
        }),
        reason_codes: vec![],
    }
}

fn cbv_high_bias() -> CharacterBaselineVector {
    CharacterBaselineVector {
        baseline_caution_offset: 3,
        baseline_novelty_dampening_offset: 3,
        baseline_approval_strictness_offset: 3,
        baseline_export_strictness_offset: 3,
        baseline_chain_conservatism_offset: 3,
        baseline_cooldown_multiplier_class: 3,
    }
}

fn pev_high_bias() -> PolicyEcologyVector {
    PolicyEcologyVector {
        conservatism_bias: 3,
        novelty_penalty_bias: 3,
        manipulation_aversion_bias: 3,
        reversibility_bias: 3,
    }
}

fn hbv_from_hpa() -> HbvOffsets {
    let path = std::env::temp_dir().join(format!(
        "hbv_hpa_state_{}_{}.json",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    let _ = std::fs::remove_file(&path);
    let mut hpa = Hpa::new(&path);
    let output = hpa.tick(&HpaInput {
        integrity_state: IntegrityState::Fail,
        replay_mismatch_present: false,
        dlp_critical_present: false,
        receipt_invalid_present: false,
        deny_storm_present: false,
        timeouts_burst_present: true,
        unlock_present: false,
        stable_medium_window: false,
        calibrate_now: false,
    });

    HbvOffsets {
        baseline_caution_offset: output.baseline_caution_offset as i32,
        baseline_novelty_dampening_offset: output.baseline_novelty_dampening_offset as i32,
        baseline_approval_strictness_offset: output.baseline_approval_strictness_offset as i32,
        baseline_export_strictness_offset: output.baseline_export_strictness_offset as i32,
        baseline_chain_conservatism_offset: output.baseline_chain_conservatism_offset as i32,
        baseline_cooldown_multiplier_class: output.baseline_cooldown_multiplier_class as i32,
        reward_block_bias: Some(LevelClass::High),
        reason_codes: output.reason_codes.codes.clone(),
    }
}

fn mock_reader() -> MockPvgsReader {
    MockPvgsReader {
        cbv: Some(cbv_high_bias()),
        pev: Some(pev_high_bias()),
        cbv_digest: Some([7u8; 32]),
        pev_digest: Some([8u8; 32]),
        ..Default::default()
    }
}

fn drive_frame(
    engine: &mut RegulationEngine,
    frame: SignalFrame,
    now_ms: u64,
) -> ucf::v1::ControlFrame {
    engine.enqueue_signal_frame(frame).expect("frame enqueued");
    engine.tick(now_ms)
}

#[test]
fn baseline_resolver_max_merge_reaches_high_and_biases() {
    let hbv = hbv_from_hpa();
    let baseline = resolve_baseline(&BaselineInputs {
        cbv: Some(cbv_high_bias()),
        pev: Some(pev_high_bias()),
        hbv: Some(hbv.clone()),
        integrity: Some(IntegrityState::Fail),
    });

    assert_eq!(baseline.caution_floor, LevelClass::High);
    assert_eq!(baseline.chain_conservatism, LevelClass::High);
    assert_eq!(baseline.export_strictness, LevelClass::High);
    assert_eq!(baseline.reward_block_bias, LevelClass::High);
    assert!(baseline
        .reason_codes
        .codes
        .contains(&"cbv_influence".to_string()));
    assert!(baseline
        .reason_codes
        .codes
        .contains(&"pev_influence".to_string()));
    assert!(baseline
        .reason_codes
        .codes
        .iter()
        .any(|reason| reason.starts_with("hpa_")));
    assert!(hbv
        .reason_codes
        .iter()
        .any(|reason| reason.starts_with("hpa_")));
}

#[test]
fn identical_frames_produce_deterministic_control_and_digest() {
    let path_a = std::env::temp_dir().join(format!(
        "hpa_state_engine_a_{}_{}.json",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    let path_b = std::env::temp_dir().join(format!(
        "hpa_state_engine_b_{}_{}.json",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));

    let _ = std::fs::remove_file(&path_a);
    std::env::set_var("HPA_STATE_PATH", &path_a);
    let mut engine_a = RegulationEngine::default();
    engine_a.set_pvgs_reader(mock_reader());
    let _ = std::fs::remove_file(&path_b);
    std::env::set_var("HPA_STATE_PATH", &path_b);
    let mut engine_b = RegulationEngine::default();
    engine_b.set_pvgs_reader(mock_reader());

    // Prime the baseline resolver by ensuring hbv offsets are present.
    let frame = medium_frame();
    let control_a = drive_frame(&mut engine_a, frame.clone(), 1_000);
    let control_b = drive_frame(&mut engine_b, frame, 1_000);

    let overlays_a = control_a.overlays.as_ref().expect("overlays present");
    assert!(overlays_a.simulate_first);
    assert!(overlays_a.export_lock);
    assert!(overlays_a.novelty_lock);

    let overlays_b = control_b.overlays.as_ref().expect("overlays present");
    assert_eq!(overlays_a, overlays_b);

    let reason_codes_a = &control_a.profile_reason_codes;
    let mut sorted_a = reason_codes_a.clone();
    sorted_a.sort();
    assert_eq!(reason_codes_a, &sorted_a);

    assert!(reason_codes_a.contains(&(ReasonCode::RcGvCbvUpdated as i32)));
    assert!(reason_codes_a.contains(&(ReasonCode::RcGvPevUpdated as i32)));
    assert!(reason_codes_a.contains(&(ReasonCode::RcGvProgressRewardBlocked as i32)));

    assert_eq!(
        control_a.control_frame_digest,
        control_b.control_frame_digest
    );
    assert_eq!(
        control_a.profile_reason_codes,
        control_b.profile_reason_codes
    );
}
