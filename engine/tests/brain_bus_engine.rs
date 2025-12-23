use engine::RegulationEngine;
use pvgs_client::MockPvgsReader;
use ucf::v1::{
    ExecStats, IntegrityStateClass, LevelClass, PolicyStats, ReceiptStats, SignalFrame, WindowKind,
};

fn mock_reader() -> MockPvgsReader {
    MockPvgsReader {
        cbv: Some(Default::default()),
        pev: Some(Default::default()),
        cbv_digest: Some([0xAA; 32]),
        pev_digest: Some([0xBB; 32]),
        ..Default::default()
    }
}

fn canonical_frames() -> Vec<SignalFrame> {
    vec![
        SignalFrame {
            window_kind: WindowKind::Short as i32,
            window_index: Some(7),
            timestamp_ms: Some(1_000),
            policy_stats: Some(PolicyStats {
                deny_count: 3,
                allow_count: 0,
                top_reason_codes: vec![],
            }),
            exec_stats: Some(ExecStats {
                timeout_count: 1,
                partial_failure_count: 0,
                tool_unavailable_count: 0,
                tool_id: None,
                dlp_block_count: 1,
                top_reason_codes: vec![],
            }),
            integrity_state: IntegrityStateClass::Degraded as i32,
            top_reason_codes: vec![],
            signal_frame_digest: None,
            receipt_stats: Some(ReceiptStats {
                receipt_missing_count: 1,
                receipt_invalid_count: 1,
            }),
            reason_codes: vec![],
        },
        SignalFrame {
            window_kind: WindowKind::Medium as i32,
            window_index: Some(8),
            timestamp_ms: Some(2_000),
            policy_stats: Some(PolicyStats {
                deny_count: 1,
                allow_count: 2,
                top_reason_codes: vec![],
            }),
            exec_stats: Some(ExecStats {
                timeout_count: 2,
                partial_failure_count: 1,
                tool_unavailable_count: 0,
                tool_id: None,
                dlp_block_count: 0,
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
        },
    ]
}

fn build_engine(tag: &str) -> RegulationEngine {
    let path = std::env::temp_dir().join(format!(
        "brain_bus_engine_hpa_{}_{}_{}.json",
        tag,
        std::process::id(),
        99
    ));
    let _ = std::fs::remove_file(&path);
    std::env::set_var("HPA_STATE_PATH", &path);

    let mut engine = RegulationEngine::default();
    engine.set_pvgs_reader(mock_reader());
    engine
}

fn run_canonical_sequence(tag: &str) -> (RegulationEngine, ucf::v1::ControlFrame) {
    let mut engine = build_engine(tag);
    for frame in canonical_frames() {
        engine.enqueue_signal_frame(frame).expect("frame enqueued");
    }

    let control = engine.tick(2_500);
    (engine, control)
}

#[test]
fn brain_bus_control_matches_golden_digest_and_reasons() {
    let (_, control) = run_canonical_sequence("golden_a");

    let digest = control
        .control_frame_digest
        .as_ref()
        .expect("digest present");
    let expected_digest: [u8; 32] = [
        0x7a, 0x3b, 0xe5, 0xa8, 0x46, 0x71, 0xa6, 0x0f, 0xf8, 0x55, 0x8d, 0xf1, 0x6a, 0x59, 0xe0,
        0xfa, 0x10, 0x48, 0xd3, 0xb6, 0x15, 0x2f, 0x3e, 0x82, 0x49, 0x8b, 0xed, 0xf0, 0x64, 0x90,
        0x2d, 0xa0,
    ];
    assert_eq!(digest.as_slice(), expected_digest);

    let expected_reasons = vec![1, 30, 40, 40, 44, 45, 47, 53, 56, 58];
    assert_eq!(control.profile_reason_codes, expected_reasons);

    let overlays = control.overlays.as_ref().expect("overlays present");
    assert!(overlays.simulate_first);
    assert!(overlays.export_lock);
    assert_eq!(control.cooldown_class, Some(LevelClass::High as i32));
}

#[test]
fn brain_bus_control_is_reproducible() {
    let (_, control_a) = run_canonical_sequence("repeat_a");
    let (_, control_b) = run_canonical_sequence("repeat_b");

    assert_eq!(
        control_a.control_frame_digest,
        control_b.control_frame_digest
    );
    assert_eq!(
        control_a.profile_reason_codes,
        control_b.profile_reason_codes
    );
    assert_eq!(control_a.cooldown_class, control_b.cooldown_class);
}
