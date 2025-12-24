use dbm_8_serotonin::{SerInput, Serotonin};
use dbm_core::limits::MAX_LOCK_MS;
use dbm_core::{CooldownClass, DbmModule, IntegrityState, LevelClass, ReasonSet};
use dbm_hpa::HpaInput;
use dbm_pprf::{Pprf, PprfInput};
use dbm_sc::ScOutput;

#[test]
fn hpa_counters_saturate_under_stress() {
    let mut hpa = dbm_hpa::Hpa::default();
    let input = HpaInput {
        integrity_state: IntegrityState::Fail,
        replay_mismatch_present: true,
        dlp_critical_present: true,
        receipt_invalid_present: true,
        deny_storm_present: true,
        timeouts_burst_present: true,
        unlock_present: false,
        stable_medium_window: false,
        calibrate_now: false,
    };

    for _ in 0..200 {
        let output = hpa.tick(&input);
        assert!(output.baseline_caution_offset <= 100);
        assert!(output.baseline_export_strictness_offset <= 100);
        assert!(output.baseline_novelty_dampening_offset <= 100);
        assert!(output.baseline_chain_conservatism_offset <= 100);
    }
}

#[test]
fn serotonin_stability_does_not_exceed_high() {
    let mut ser = Serotonin::default();
    let mut stability_levels = Vec::new();
    let input = SerInput {
        integrity: IntegrityState::Fail,
        replay_mismatch_present: true,
        receipt_invalid_count_medium: 3,
        dlp_critical_count_medium: 6,
        flapping_count_medium: 10,
        unlock_present: false,
        stability_floor: LevelClass::Low,
    };

    for _ in 0..50 {
        let output = ser.tick(&input);
        stability_levels.push(output.stability);
        assert!(matches!(
            output.stability,
            LevelClass::Low | LevelClass::Med | LevelClass::High
        ));
    }
}

#[test]
fn pprf_lock_does_not_exceed_cap() {
    let mut pprf = Pprf::default();
    let mut now_ms = 0;
    let orient = ScOutput {
        target: dbm_core::OrientTarget::Integrity,
        urgency: dbm_core::UrgencyClass::Low,
        recommended_dwm: dbm_core::DwmMode::Report,
        reason_codes: ReasonSet::default(),
    };

    for _ in 0..5 {
        let input = PprfInput {
            orient: orient.clone(),
            current_target: dbm_core::OrientTarget::Approval,
            current_dwm: dbm_core::DwmMode::ExecPlan,
            cooldown_class: CooldownClass::Base,
            stability: LevelClass::Low,
            arousal: LevelClass::Low,
            now_ms,
        };
        let output = pprf.tick(&input);
        let lock_cap = now_ms.saturating_add(MAX_LOCK_MS);
        assert!(output.lock_until_ms <= lock_cap);
        now_ms += 2_000;
    }
}
