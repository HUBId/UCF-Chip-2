#![forbid(unsafe_code)]

use dbm_core::{CooldownClass, DbmModule, DwmMode, LevelClass, OrientTarget, ReasonSet};
use dbm_sc::ScOutput;

#[derive(Debug, Clone, Default)]
pub struct PprfInput {
    pub orient: ScOutput,
    pub current_target: OrientTarget,
    pub current_dwm: DwmMode,
    pub cooldown_class: CooldownClass,
    pub stability: LevelClass,
    pub arousal: LevelClass,
    pub now_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PprfOutput {
    pub active_target: OrientTarget,
    pub active_dwm: DwmMode,
    pub shift_executed: bool,
    pub lock_until_ms: u64,
    pub reason_codes: ReasonSet,
}

impl Default for PprfOutput {
    fn default() -> Self {
        Self {
            active_target: OrientTarget::Approval,
            active_dwm: DwmMode::ExecPlan,
            shift_executed: false,
            lock_until_ms: 0,
            reason_codes: ReasonSet::default(),
        }
    }
}

#[derive(Debug, Default)]
pub struct Pprf {
    pub active_target: OrientTarget,
    pub active_dwm: DwmMode,
    pub lock_until_ms: u64,
    pub shift_count_medium: u32,
}

impl Pprf {
    pub fn new() -> Self {
        Self {
            active_target: OrientTarget::Approval,
            active_dwm: DwmMode::ExecPlan,
            lock_until_ms: 0,
            shift_count_medium: 0,
        }
    }

    fn target_severity(target: OrientTarget) -> u8 {
        match target {
            OrientTarget::Integrity => 5,
            OrientTarget::Dlp => 4,
            OrientTarget::Recovery => 3,
            OrientTarget::Replay => 2,
            OrientTarget::PolicyPressure => 1,
            OrientTarget::Approval => 0,
        }
    }

    fn dwm_severity(mode: DwmMode) -> u8 {
        match mode {
            DwmMode::Report => 3,
            DwmMode::Stabilize => 2,
            DwmMode::Simulate => 1,
            DwmMode::ExecPlan => 0,
        }
    }

    fn lock_duration(class: CooldownClass) -> u64 {
        match class {
            CooldownClass::Base => 10_000,
            CooldownClass::Longer => 30_000,
        }
    }

    fn should_shift(&self, orient: &ScOutput, now_ms: u64) -> bool {
        let stricter =
            Self::target_severity(orient.target) > Self::target_severity(self.active_target);
        stricter || now_ms >= self.lock_until_ms
    }

    fn apply_shift(
        &mut self,
        orient: &ScOutput,
        now_ms: u64,
        lock_duration: u64,
        current_dwm: DwmMode,
    ) {
        self.active_target = orient.target;
        let recommended = orient.recommended_dwm;
        self.active_dwm = if Self::dwm_severity(recommended) >= Self::dwm_severity(current_dwm) {
            recommended
        } else {
            current_dwm
        };
        self.lock_until_ms = now_ms.saturating_add(lock_duration);
        self.shift_count_medium = self.shift_count_medium.saturating_add(1);
    }

    fn apply_flapping_penalty(&mut self, reason_codes: &mut ReasonSet) {
        if self.shift_count_medium > 6 {
            if Self::dwm_severity(DwmMode::Report) > Self::dwm_severity(self.active_dwm) {
                self.active_dwm = DwmMode::Report;
            }
            self.lock_until_ms = self.lock_until_ms.saturating_add(30_000);
            reason_codes.insert("RC.GV.FLAPPING.PENALTY");
        }
    }

    pub fn on_medium_window_rollover(&mut self) {
        self.shift_count_medium = 0;
    }
}

impl DbmModule for Pprf {
    type Input = PprfInput;
    type Output = PprfOutput;

    fn tick(&mut self, input: &Self::Input) -> Self::Output {
        self.active_target = input.current_target;
        self.active_dwm = input.current_dwm;

        let lock_duration = Self::lock_duration(input.cooldown_class);
        let mut output = PprfOutput::default();
        output
            .reason_codes
            .extend(input.orient.reason_codes.codes.clone());

        if self.should_shift(&input.orient, input.now_ms) {
            self.apply_shift(
                &input.orient,
                input.now_ms,
                lock_duration,
                input.current_dwm,
            );
            output.shift_executed = true;
            output.reason_codes.insert("RC.GV.FOCUS_SHIFT.EXECUTED");
        } else {
            output
                .reason_codes
                .insert("RC.GV.FOCUS_SHIFT.BLOCKED_BY_LOCK");
        }

        self.apply_flapping_penalty(&mut output.reason_codes);

        output.active_target = self.active_target;
        output.active_dwm = self.active_dwm;
        output.lock_until_ms = self.lock_until_ms;

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dbm_core::UrgencyClass;

    fn orient(target: OrientTarget, recommended_dwm: DwmMode, reason: &str) -> ScOutput {
        let mut reason_codes = ReasonSet::default();
        reason_codes.insert(reason);
        ScOutput {
            target,
            urgency: UrgencyClass::Low,
            recommended_dwm,
            reason_codes,
        }
    }

    #[test]
    fn stricter_target_preempts_lock() {
        let mut pprf = Pprf {
            active_target: OrientTarget::Approval,
            active_dwm: DwmMode::ExecPlan,
            lock_until_ms: 50_000,
            shift_count_medium: 0,
        };

        let input = PprfInput {
            orient: orient(
                OrientTarget::Integrity,
                DwmMode::Report,
                "RC.GV.ORIENT.TARGET_INTEGRITY",
            ),
            current_target: pprf.active_target,
            current_dwm: pprf.active_dwm,
            cooldown_class: CooldownClass::Base,
            stability: LevelClass::Low,
            arousal: LevelClass::Low,
            now_ms: 10_000,
        };

        let output = pprf.tick(&input);

        assert!(output.shift_executed);
        assert_eq!(output.active_target, OrientTarget::Integrity);
        assert_eq!(output.active_dwm, DwmMode::Report);
    }

    #[test]
    fn non_strict_shift_blocked_by_lock() {
        let mut pprf = Pprf {
            active_target: OrientTarget::Recovery,
            active_dwm: DwmMode::Report,
            lock_until_ms: 20_000,
            shift_count_medium: 0,
        };

        let input = PprfInput {
            orient: orient(
                OrientTarget::Approval,
                DwmMode::ExecPlan,
                "RC.GV.ORIENT.TARGET_APPROVAL",
            ),
            current_target: pprf.active_target,
            current_dwm: pprf.active_dwm,
            cooldown_class: CooldownClass::Base,
            stability: LevelClass::Low,
            arousal: LevelClass::Low,
            now_ms: 10_000,
        };

        let output = pprf.tick(&input);

        assert!(!output.shift_executed);
        assert_eq!(output.active_target, OrientTarget::Recovery);
        assert_eq!(output.active_dwm, DwmMode::Report);
        assert!(output
            .reason_codes
            .codes
            .contains(&"RC.GV.FOCUS_SHIFT.BLOCKED_BY_LOCK".to_string()));
    }

    #[test]
    fn flapping_penalty_forces_report_and_longer_lock() {
        let mut pprf = Pprf::new();
        let orient = orient(
            OrientTarget::PolicyPressure,
            DwmMode::Simulate,
            "RC.GV.ORIENT.TARGET_POLICY_PRESSURE",
        );

        let mut now = 0u64;
        for _ in 0..7 {
            let input = PprfInput {
                orient: orient.clone(),
                current_target: pprf.active_target,
                current_dwm: pprf.active_dwm,
                cooldown_class: CooldownClass::Base,
                stability: LevelClass::Low,
                arousal: LevelClass::Low,
                now_ms: now,
            };
            let _ = pprf.tick(&input);
            now += 20_000;
        }

        let input = PprfInput {
            orient,
            current_target: pprf.active_target,
            current_dwm: pprf.active_dwm,
            cooldown_class: CooldownClass::Base,
            stability: LevelClass::Low,
            arousal: LevelClass::Low,
            now_ms: now,
        };

        let output = pprf.tick(&input);

        assert_eq!(output.active_dwm, DwmMode::Report);
        assert!(output.lock_until_ms >= now + 10_000 + 30_000);
        assert!(output
            .reason_codes
            .codes
            .contains(&"RC.GV.FLAPPING.PENALTY".to_string()));
    }
}
