#![forbid(unsafe_code)]

use engine::RegulationEngine;
use hex::encode as hex_encode;
use ucf::v1::{ExecStats, IntegrityStateClass, PolicyStats, ReasonCode, SignalFrame, WindowKind};
use wire::FrameIngestor;

fn main() {
    let now_ms = 1;
    let mut ingestor = FrameIngestor::new();
    let mut engine = RegulationEngine::default();

    let signal_frame = SignalFrame {
        window_kind: WindowKind::Short as i32,
        window_index: Some(1),
        timestamp_ms: Some(now_ms),
        policy_stats: Some(PolicyStats {
            deny_count: 0,
            allow_count: 10,
        }),
        exec_stats: Some(ExecStats { timeout_count: 0 }),
        integrity_state: IntegrityStateClass::Ok as i32,
        top_reason_codes: vec![ReasonCode::Unknown as i32],
        signal_frame_digest: None,
    };

    ingestor
        .ingest_signal_frame(signal_frame.clone())
        .expect("signal frame should be accepted");

    let control_frame = engine.on_signal_frame(signal_frame, now_ms);
    println!(
        "ControlFrame => profile: {}, overlays: {:?}, toolclass_mask: {:?}, digest: {}",
        control_frame
            .active_profile
            .as_ref()
            .map(|p| p.profile.clone())
            .unwrap_or_default(),
        control_frame.overlays,
        control_frame.toolclass_mask,
        control_frame
            .control_frame_digest
            .as_ref()
            .map(hex_encode)
            .unwrap_or_else(|| "missing".to_string())
    );

    let missing_control = engine.on_tick(now_ms + 60_000);
    println!(
        "Missing frame decision => profile: {}, overlays: {:?}, toolclass_mask: {:?}",
        missing_control
            .active_profile
            .as_ref()
            .map(|p| p.profile.clone())
            .unwrap_or_default(),
        missing_control.overlays,
        missing_control.toolclass_mask,
    );
}
