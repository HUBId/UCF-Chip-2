use engine::RegulationEngine;
use ucf::v1::{
    ExecStats, IntegrityStateClass, Overlays, PolicyStats, ReasonCode, ReceiptStats, SignalFrame,
};

fn profile_rank(profile: &str) -> u8 {
    match profile.split('_').next().unwrap_or(profile) {
        "M0" => 0,
        "M1" => 1,
        "M2" => 2,
        "M3" => 3,
        _ => 0,
    }
}

fn medium_trace_frame(now_ms: u64, reason_codes: Vec<i32>) -> SignalFrame {
    SignalFrame {
        window_kind: ucf::v1::WindowKind::Medium as i32,
        window_index: Some(now_ms),
        timestamp_ms: Some(now_ms),
        policy_stats: Some(PolicyStats {
            deny_count: 0,
            allow_count: 0,
            top_reason_codes: vec![],
        }),
        exec_stats: Some(ExecStats {
            timeout_count: 0,
            partial_failure_count: 0,
            tool_unavailable_count: 0,
            tool_id: None,
            dlp_block_count: 0,
            top_reason_codes: vec![],
        }),
        integrity_state: IntegrityStateClass::Ok as i32,
        top_reason_codes: vec![],
        signal_frame_digest: None,
        receipt_stats: Some(ReceiptStats {
            receipt_missing_count: 0,
            receipt_invalid_count: 0,
        }),
        reason_codes,
    }
}

#[test]
fn trace_fail_sets_m1_overlays_and_reason_code() {
    let mut engine = RegulationEngine::default();
    let frame = medium_trace_frame(1_000, vec![ReasonCode::RcGvTraceFail as i32]);
    engine.enqueue_signal_frame(frame).expect("frame enqueued");
    let control = engine.tick(1_000);

    let profile = control
        .active_profile
        .as_ref()
        .expect("profile present")
        .profile
        .as_str();
    assert!(profile_rank(profile) >= 1);

    let overlays = control.overlays.as_ref().expect("overlays present");
    assert!(overlays.simulate_first);
    assert!(overlays.novelty_lock);

    let mut sorted = control.profile_reason_codes.clone();
    sorted.sort();
    assert_eq!(control.profile_reason_codes, sorted);
    assert!(control
        .profile_reason_codes
        .contains(&(ReasonCode::RcGvTraceFail as i32)));
}

#[test]
fn consecutive_trace_fail_escalates_to_m2() {
    let mut engine = RegulationEngine::default();

    let frame_a = medium_trace_frame(1_000, vec![ReasonCode::RcGvTraceFail as i32]);
    engine
        .enqueue_signal_frame(frame_a)
        .expect("frame enqueued");
    let _ = engine.tick(1_000);

    let frame_b = medium_trace_frame(2_000, vec![ReasonCode::RcGvTraceFail as i32]);
    engine
        .enqueue_signal_frame(frame_b)
        .expect("frame enqueued");
    let control = engine.tick(2_000);

    let profile = control
        .active_profile
        .as_ref()
        .expect("profile present")
        .profile
        .as_str();
    assert!(profile_rank(profile) >= 2);
}

#[test]
fn trace_pass_is_informational_only() {
    let mut engine_pass = RegulationEngine::default();
    let frame_pass = medium_trace_frame(1_000, vec![ReasonCode::RcGvTracePass as i32]);
    engine_pass
        .enqueue_signal_frame(frame_pass)
        .expect("frame enqueued");
    let control_pass = engine_pass.tick(1_000);

    let mut engine_plain = RegulationEngine::default();
    let frame_plain = medium_trace_frame(1_000, vec![]);
    engine_plain
        .enqueue_signal_frame(frame_plain)
        .expect("frame enqueued");
    let control_plain = engine_plain.tick(1_000);

    assert_eq!(control_pass.active_profile, control_plain.active_profile);
    assert_eq!(
        control_pass.overlays,
        Some(Overlays {
            simulate_first: false,
            export_lock: false,
            novelty_lock: false,
            chain_tightening: false,
        })
    );
}
