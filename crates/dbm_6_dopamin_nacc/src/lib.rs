#![forbid(unsafe_code)]

use dbm_core::{DbmComponent, DwmMode, IntegrityState, ReasonSet};

#[derive(Debug, Clone, Default)]
pub struct DopaOutput {
    pub progress: bool,
    pub incentive_focus_hint: bool,
    pub replay_hint: bool,
    pub reward_block: bool,
    pub reason_codes: ReasonSet,
}

#[derive(Debug, Default)]
pub struct DopaminNacc {
    reward_block: bool,
    diminishing_returns_counter: u32,
}

impl DopaminNacc {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn tick(&mut self) -> DopaOutput {
        // TODO: replace with module-specific processing once available.
        let progress = false;
        let mut reason_codes = ReasonSet::default();

        let replay_hint = self.diminishing_returns_counter >= 3;
        if replay_hint {
            reason_codes.insert("RC.GV.REPLAY.DIMINISHING_RETURNS");
        }

        let incentive_focus_hint = Self::incentive_hint(progress, self.reward_block);

        DopaOutput {
            progress,
            incentive_focus_hint,
            replay_hint,
            reward_block: self.reward_block,
            reason_codes,
        }
    }

    fn incentive_hint(progress: bool, reward_block: bool) -> bool {
        progress || reward_block
    }
}

impl DbmComponent for DopaminNacc {
    fn mode(&self) -> DwmMode {
        // TODO: return active mode when implemented.
        DwmMode::Simulate
    }

    fn integrity(&self) -> IntegrityState {
        // TODO: track integrity state when implemented.
        IntegrityState::Ok
    }
}
