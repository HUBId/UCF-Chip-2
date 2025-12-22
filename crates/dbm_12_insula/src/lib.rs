#![forbid(unsafe_code)]

use dbm_core::{DbmComponent, DwmMode, IntegrityState};

#[derive(Debug, Default)]
pub struct Insula {}

impl Insula {
    pub fn new() -> Self {
        Self {}
    }

    pub fn tick(&mut self) {
        // TODO: replace with module-specific processing once available.
    }
}

impl DbmComponent for Insula {
    fn mode(&self) -> DwmMode {
        // TODO: return active mode when implemented.
        DwmMode::Simulate
    }

    fn integrity(&self) -> IntegrityState {
        // TODO: track integrity state when implemented.
        IntegrityState::Ok
    }
}
