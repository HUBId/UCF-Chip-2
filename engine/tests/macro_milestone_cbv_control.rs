mod common;

use common::{setup_pvgs_with_keys_and_ruleset, PvgsHandle};
use engine::RegulationEngine;
use pvgs_client::LocalPvgsReader;
use ucf::v1::{
    ConsistencyClass, ExecStats, IntegrityStateClass, LevelClass, MacroMilestone,
    MacroMilestoneState, PolicyStats, ReceiptStats, SignalFrame, TraitUpdate, TraitUpdateDirection,
    WindowKind,
};

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

fn finalized_macro_milestone() -> MacroMilestone {
    MacroMilestone {
        state: MacroMilestoneState::Finalized as i32,
        consistency_class: ConsistencyClass::ConsistencyHigh as i32,
        trait_updates: vec![
            TraitUpdate {
                trait_name: "approval_strictness".to_string(),
                magnitude: LevelClass::High as i32,
                direction: TraitUpdateDirection::IncreaseStrictness as i32,
            },
            TraitUpdate {
                trait_name: "novelty_dampening".to_string(),
                magnitude: LevelClass::Med as i32,
                direction: TraitUpdateDirection::IncreaseStrictness as i32,
            },
        ],
        macro_digest: Some(vec![0xAB; 32]),
    }
}

fn build_engine_with_pvgs(handle: &PvgsHandle) -> RegulationEngine {
    let mut engine = RegulationEngine::default();
    engine.set_pvgs_reader(LocalPvgsReader::new(handle.store()));
    engine
}

#[test]
fn control_frame_unchanged_without_macro_cbv() {
    let handle = setup_pvgs_with_keys_and_ruleset();
    let frame = base_frame();

    let mut engine = build_engine_with_pvgs(&handle);
    let control_a = engine.on_signal_frame(frame.clone(), 1_000);
    let mut engine_again = build_engine_with_pvgs(&handle);
    let control_b = engine_again.on_signal_frame(frame, 1_000);

    let overlays = control_a.overlays.as_ref().expect("overlays present");
    assert!(!overlays.novelty_lock);
    assert!(control_a
        .approval_mode
        .as_deref()
        .map(|mode| !mode.eq_ignore_ascii_case("STRICT"))
        .unwrap_or(true));
    assert!(control_a.character_epoch_digest.is_none());
    assert_eq!(
        control_a.control_frame_digest,
        control_b.control_frame_digest
    );
}

#[test]
fn macro_milestone_commits_cbv_and_tightens_control_frame() {
    let handle = setup_pvgs_with_keys_and_ruleset();
    let milestone = finalized_macro_milestone();
    handle.append_macro_milestone(milestone);

    let frame = base_frame();
    let mut engine = build_engine_with_pvgs(&handle);
    let control = engine.on_signal_frame(frame.clone(), 2_000);

    let overlays = control.overlays.as_ref().expect("overlays present");
    assert!(control
        .approval_mode
        .as_deref()
        .map(|mode| mode.eq_ignore_ascii_case("STRICT"))
        .unwrap_or(false));
    assert!(overlays.novelty_lock);
    assert!(control.character_epoch_digest.is_some());

    let mut engine_again = build_engine_with_pvgs(&handle);
    let control_repeat = engine_again.on_signal_frame(frame, 2_000);

    assert_eq!(
        control.control_frame_digest,
        control_repeat.control_frame_digest
    );
    assert_eq!(
        control.character_epoch_digest,
        control_repeat.character_epoch_digest
    );
}
