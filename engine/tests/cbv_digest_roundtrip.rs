use chip4::pvgs::{Cbv, Digest32, InMemoryPvgs};
use engine::RegulationEngine;
use pvgs_client::LocalPvgsReader;
use ucf::v1::{ExecStats, IntegrityStateClass, PolicyStats, ReceiptStats, SignalFrame, WindowKind};

fn base_frame() -> SignalFrame {
    SignalFrame {
        window_kind: WindowKind::Short as i32,
        window_index: Some(1),
        timestamp_ms: None,
        policy_stats: Some(PolicyStats {
            deny_count: 0,
            allow_count: 1,
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
        reason_codes: vec![],
    }
}

fn cbv_update(epoch: u64, byte: u8) -> Cbv {
    Cbv {
        epoch,
        cbv_digest: Some(Digest32::from_array([byte; 32])),
        proof_receipt_ref: Some(vec![0xAA]),
        signature: Some(vec![0xBB]),
        cbv: None,
    }
}

#[test]
fn cbv_digest_roundtrip_in_control_frame() {
    let store = InMemoryPvgs::new();
    let digest_epoch_one = [1u8; 32];
    store.commit_cbv_update(cbv_update(1, 1));

    let mut engine = RegulationEngine::default();
    engine.set_pvgs_reader(LocalPvgsReader::new(store.clone()));

    let control_one = engine.on_signal_frame(base_frame(), 10);
    assert_eq!(
        control_one.character_epoch_digest.as_deref(),
        Some(digest_epoch_one.as_slice())
    );
    let first_control_digest = control_one.control_frame_digest.clone();

    let digest_epoch_two = [2u8; 32];
    store.commit_cbv_update(cbv_update(2, 2));
    let control_two = engine.on_signal_frame(base_frame(), 20);

    assert_eq!(
        control_two.character_epoch_digest.as_deref(),
        Some(digest_epoch_two.as_slice())
    );
    assert_ne!(first_control_digest, control_two.control_frame_digest);
}

#[test]
fn control_frame_stable_when_local_cbv_absent() {
    let store = InMemoryPvgs::new();
    let reader_a = LocalPvgsReader::new(store.clone());
    let reader_b = LocalPvgsReader::new(store);
    let mut engine_a = RegulationEngine::default();
    let mut engine_b = RegulationEngine::default();
    engine_a.set_pvgs_reader(reader_a);
    engine_b.set_pvgs_reader(reader_b);

    let frame = base_frame();
    let control_a = engine_a.on_signal_frame(frame.clone(), 1);
    let control_b = engine_b.on_signal_frame(frame, 1);

    assert!(control_a.character_epoch_digest.is_none());
    assert_eq!(
        control_a.control_frame_digest,
        control_b.control_frame_digest
    );
}
