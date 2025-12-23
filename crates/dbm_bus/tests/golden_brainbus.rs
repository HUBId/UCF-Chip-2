use dbm_12_insula::InsulaInput;
use dbm_7_lc::LcInput;
use dbm_8_serotonin::SerInput;
use dbm_9_amygdala::AmyInput;
use dbm_bus::{BrainBus, BrainInput};
use dbm_core::{CooldownClass, LevelClass, OrientTarget, OverlaySet, ProfileState};
use dbm_hpa::HpaInput;
use dbm_pag::PagInput;
use dbm_pmrf::PmrfInput;
use dbm_stn::StnInput;
use ucf::v1::WindowKind;

#[derive(Debug, PartialEq, Eq)]
struct GoldenSnapshot {
    profile: ProfileState,
    overlays: OverlaySet,
    dwm: dbm_core::DwmMode,
    target: OrientTarget,
    reason_codes: Vec<String>,
}

fn brain_bus_with_clean_state(label: &str) -> BrainBus {
    let path = std::env::temp_dir().join(format!(
        "brain_bus_hpa_state_golden_{}_{}_{}.json",
        std::process::id(),
        label,
        7
    ));
    let _ = std::fs::remove_file(&path);
    BrainBus::with_hpa(dbm_hpa::Hpa::new(path), Default::default())
}

fn base_input(now_ms: u64) -> BrainInput {
    BrainInput {
        now_ms,
        window_kind: WindowKind::Medium,
        hpa: HpaInput::default(),
        cbv: None,
        pev: None,
        lc: LcInput::default(),
        serotonin: SerInput::default(),
        amygdala: AmyInput::default(),
        pag: PagInput::default(),
        cerebellum: None,
        stn: StnInput::default(),
        pmrf: PmrfInput::default(),
        dopamin: None,
        insula: InsulaInput::default(),
        sc_unlock_present: false,
        sc_replay_planned_present: false,
        pprf_cooldown_class: CooldownClass::Base,
    }
}

fn golden_sequence() -> Vec<BrainInput> {
    let mut inputs = Vec::new();

    // 1) Normal state.
    let mut normal = base_input(1_000);
    normal.pag.stability = LevelClass::Low;
    inputs.push(normal);

    // 2) Deny storm / policy pressure.
    let mut deny = base_input(2_000);
    deny.hpa.deny_storm_present = true;
    deny.lc.deny_count_short = 5;
    deny.pag.threat = LevelClass::Med;
    inputs.push(deny);

    // 3) Receipt invalid spike.
    let mut receipt_invalid = base_input(3_000);
    receipt_invalid.hpa.receipt_invalid_present = true;
    receipt_invalid.lc.receipt_invalid_count_short = 4;
    receipt_invalid.amygdala.receipt_invalid_medium = 3;
    receipt_invalid.pag.threat = LevelClass::Med;
    inputs.push(receipt_invalid);

    // 4) DLP critical spike.
    let mut dlp = base_input(4_000);
    dlp.hpa.dlp_critical_present = true;
    dlp.amygdala.dlp_secret_present = true;
    dlp.lc.dlp_critical_present_short = true;
    dlp.pag.threat = LevelClass::High;
    inputs.push(dlp);

    // 5) Integrity fail / forensic latch.
    let mut integrity_fail = base_input(5_000);
    integrity_fail.hpa.integrity_state = dbm_core::IntegrityState::Fail;
    integrity_fail.lc.integrity = dbm_core::IntegrityState::Fail;
    integrity_fail.amygdala.integrity = dbm_core::IntegrityState::Fail;
    integrity_fail.pag.integrity = dbm_core::IntegrityState::Fail;
    integrity_fail.stn.integrity = dbm_core::IntegrityState::Fail;
    integrity_fail.pag.threat = LevelClass::High;
    inputs.push(integrity_fail);

    // 6) Unlock present + stable medium windows.
    let mut unlock = base_input(6_000);
    unlock.hpa.unlock_present = true;
    unlock.hpa.stable_medium_window = true;
    unlock.pag.unlock_present = true;
    unlock.pag.stability = LevelClass::Med;
    unlock.pag.threat = LevelClass::Low;
    inputs.push(unlock);

    inputs
}

fn snapshot_from_output(output: dbm_bus::BrainOutput) -> GoldenSnapshot {
    GoldenSnapshot {
        profile: output.decision.profile_state,
        overlays: output.decision.overlays,
        dwm: output.dwm,
        target: output.focus_target,
        reason_codes: output.reason_codes,
    }
}

#[test]
fn golden_sequence_is_deterministic() {
    let inputs = golden_sequence();

    let mut bus_a = brain_bus_with_clean_state("golden_a");
    let mut bus_b = brain_bus_with_clean_state("golden_b");

    let snapshots_a: Vec<GoldenSnapshot> = inputs
        .iter()
        .cloned()
        .map(|input| snapshot_from_output(bus_a.tick(input)))
        .collect();
    let snapshots_b: Vec<GoldenSnapshot> = inputs
        .into_iter()
        .map(|input| snapshot_from_output(bus_b.tick(input)))
        .collect();

    assert_eq!(snapshots_a, snapshots_b);

    let expected = vec![
        GoldenSnapshot {
            profile: ProfileState::M0,
            overlays: OverlaySet {
                simulate_first: false,
                export_lock: false,
                novelty_lock: false,
            },
            dwm: dbm_core::DwmMode::ExecPlan,
            target: OrientTarget::Approval,
            reason_codes: vec![
                "RC.GV.DWM.EXEC_PLAN".to_string(),
                "RC.GV.FOCUS_SHIFT.EXECUTED".to_string(),
                "RC.GV.ORIENT.TARGET_APPROVAL".to_string(),
                "baseline".to_string(),
                "integrity_ok".to_string(),
            ],
        },
        GoldenSnapshot {
            profile: ProfileState::M0,
            overlays: OverlaySet {
                simulate_first: true,
                export_lock: false,
                novelty_lock: true,
            },
            dwm: dbm_core::DwmMode::ExecPlan,
            target: OrientTarget::Approval,
            reason_codes: vec![
                "RC.GV.DWM.EXEC_PLAN".to_string(),
                "RC.GV.FOCUS_SHIFT.BLOCKED_BY_LOCK".to_string(),
                "RC.GV.ORIENT.TARGET_APPROVAL".to_string(),
                "RC.GV.PROGRESS.REWARD_BLOCKED".to_string(),
                "baseline".to_string(),
                "baseline_approval_strict".to_string(),
                "baseline_chain_conservatism".to_string(),
                "baseline_novelty_dampening".to_string(),
            ],
        },
        GoldenSnapshot {
            profile: ProfileState::M2,
            overlays: OverlaySet {
                simulate_first: true,
                export_lock: true,
                novelty_lock: true,
            },
            dwm: dbm_core::DwmMode::ExecPlan,
            target: OrientTarget::Approval,
            reason_codes: vec![
                "RC.GV.DWM.STABILIZE".to_string(),
                "RC.GV.FOCUS_SHIFT.BLOCKED_BY_LOCK".to_string(),
                "RC.GV.HOLD.ON".to_string(),
                "RC.GV.ORIENT.TARGET_APPROVAL".to_string(),
                "RC.GV.PROGRESS.REWARD_BLOCKED".to_string(),
                "RC.GV.SEQUENCE.SLOW".to_string(),
                "RC.RG.STATE.AROUSAL_UP".to_string(),
                "RC.RG.STATE.STABILITY_UP".to_string(),
            ],
        },
        GoldenSnapshot {
            profile: ProfileState::M3,
            overlays: OverlaySet {
                simulate_first: true,
                export_lock: true,
                novelty_lock: true,
            },
            dwm: dbm_core::DwmMode::Stabilize,
            target: OrientTarget::Dlp,
            reason_codes: vec![
                "RC.GV.DWM.STABILIZE".to_string(),
                "RC.GV.FOCUS_SHIFT.EXECUTED".to_string(),
                "RC.GV.HOLD.ON".to_string(),
                "RC.GV.ORIENT.TARGET_DLP".to_string(),
                "RC.GV.PROGRESS.REWARD_BLOCKED".to_string(),
                "RC.GV.SEQUENCE.SLOW".to_string(),
                "RC.RG.STATE.AROUSAL_UP".to_string(),
                "RC.RG.STATE.STABILITY_UP".to_string(),
            ],
        },
        GoldenSnapshot {
            profile: ProfileState::M3,
            overlays: OverlaySet {
                simulate_first: true,
                export_lock: true,
                novelty_lock: true,
            },
            dwm: dbm_core::DwmMode::Stabilize,
            target: OrientTarget::Dlp,
            reason_codes: vec![
                "RC.GV.DWM.STABILIZE".to_string(),
                "RC.GV.FOCUS_SHIFT.BLOCKED_BY_LOCK".to_string(),
                "RC.GV.HOLD.ON".to_string(),
                "RC.GV.ORIENT.TARGET_APPROVAL".to_string(),
                "RC.GV.PROGRESS.REWARD_BLOCKED".to_string(),
                "RC.GV.SEQUENCE.SLOW".to_string(),
                "RC.RG.STATE.AROUSAL_UP".to_string(),
                "RC.RG.STATE.STABILITY_UP".to_string(),
            ],
        },
        GoldenSnapshot {
            profile: ProfileState::M0,
            overlays: OverlaySet {
                simulate_first: true,
                export_lock: true,
                novelty_lock: true,
            },
            dwm: dbm_core::DwmMode::Stabilize,
            target: OrientTarget::Dlp,
            reason_codes: vec![
                "RC.GV.DWM.STABILIZE".to_string(),
                "RC.GV.FOCUS_SHIFT.BLOCKED_BY_LOCK".to_string(),
                "RC.GV.ORIENT.TARGET_APPROVAL".to_string(),
                "RC.GV.PROGRESS.REWARD_BLOCKED".to_string(),
                "RC.RG.STATE.AROUSAL_UP".to_string(),
                "RC.RG.STATE.STABILITY_UP".to_string(),
                "baseline".to_string(),
                "baseline_approval_strict".to_string(),
            ],
        },
    ];

    assert_eq!(snapshots_a, expected);
}
