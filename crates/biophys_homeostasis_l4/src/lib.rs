#![forbid(unsafe_code)]

use blake3::Hasher;

const SCALE_UNIT_Q: u32 = 1000;

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum HomeoMode {
    OFF = 0,
    REPLAY_ONLY = 1,
    ALWAYS = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HomeostasisConfig {
    pub enabled: bool,
    pub mode: HomeoMode,
    pub target_rate_q: u16,
    pub window_steps: u16,
    pub gain_up_q: u16,
    pub gain_down_q: u16,
    pub scale_min_q: u16,
    pub scale_max_q: u16,
}

impl Default for HomeostasisConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: HomeoMode::OFF,
            target_rate_q: 100,
            window_steps: 100,
            gain_up_q: 5,
            gain_down_q: 5,
            scale_min_q: 800,
            scale_max_q: 1200,
        }
    }
}

impl HomeostasisConfig {
    pub fn is_active(&self, in_replay: bool) -> bool {
        if !self.enabled {
            return false;
        }
        match self.mode {
            HomeoMode::OFF => false,
            HomeoMode::REPLAY_ONLY => in_replay,
            HomeoMode::ALWAYS => true,
        }
    }

    fn scale_bounds(&self) -> (u16, u16) {
        let min = self.scale_min_q.min(self.scale_max_q);
        let max = self.scale_min_q.max(self.scale_max_q);
        (min, max)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HomeostasisState {
    pub scale_q: u16,
    pub spike_count_window: u32,
    pub step_in_window: u16,
}

impl Default for HomeostasisState {
    fn default() -> Self {
        Self {
            scale_q: 1000,
            spike_count_window: 0,
            step_in_window: 0,
        }
    }
}

pub fn homeostasis_tick(
    config: HomeostasisConfig,
    state: &mut HomeostasisState,
    spike_count: u32,
    in_replay: bool,
) -> bool {
    if !config.is_active(in_replay) {
        return false;
    }

    state.spike_count_window = state.spike_count_window.saturating_add(spike_count);
    state.step_in_window = state.step_in_window.saturating_add(1);

    let window = config.window_steps.max(1);
    if state.step_in_window < window {
        return false;
    }

    let (min_scale, max_scale) = config.scale_bounds();
    let target = config.target_rate_q as u32;
    let mut updated = false;
    if state.spike_count_window > target {
        let next = state
            .scale_q
            .saturating_sub(config.gain_down_q)
            .max(min_scale);
        updated = next != state.scale_q;
        state.scale_q = next;
    } else if state.spike_count_window < target {
        let next = state
            .scale_q
            .saturating_add(config.gain_up_q)
            .min(max_scale);
        updated = next != state.scale_q;
        state.scale_q = next;
    }

    state.step_in_window = 0;
    state.spike_count_window = 0;

    updated
}

pub fn scale_g_max_fixed(g_max_eff_q: u32, scale_q: u16, max_fixed: u32) -> u32 {
    let scaled = (g_max_eff_q as u64 * scale_q as u64) / SCALE_UNIT_Q as u64;
    scaled.min(max_fixed as u64) as u32
}

pub fn homeostasis_snapshot_digest(state: &HomeostasisState) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"UCF:BIO:L4:HOMEOSTASIS:SNAP");
    hasher.update(&state.scale_q.to_le_bytes());
    hasher.update(&state.spike_count_window.to_le_bytes());
    hasher.update(&state.step_in_window.to_le_bytes());
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_config() -> HomeostasisConfig {
        HomeostasisConfig {
            enabled: true,
            mode: HomeoMode::ALWAYS,
            target_rate_q: 5,
            window_steps: 2,
            gain_up_q: 10,
            gain_down_q: 10,
            scale_min_q: 900,
            scale_max_q: 1100,
        }
    }

    #[test]
    fn deterministic_scale_evolution() {
        let config = base_config();
        let mut state_a = HomeostasisState::default();
        let mut state_b = HomeostasisState::default();
        let spikes = [2u32, 9, 5, 0, 8, 2, 3, 7];
        let mut history_a = Vec::new();
        let mut history_b = Vec::new();
        for spike in spikes {
            homeostasis_tick(config, &mut state_a, spike, true);
            homeostasis_tick(config, &mut state_b, spike, true);
            history_a.push(state_a.scale_q);
            history_b.push(state_b.scale_q);
        }
        assert_eq!(history_a, history_b);
    }

    #[test]
    fn scale_decreases_above_target() {
        let config = HomeostasisConfig {
            target_rate_q: 1,
            window_steps: 2,
            ..base_config()
        };
        let mut state = HomeostasisState::default();
        homeostasis_tick(config, &mut state, 2, true);
        homeostasis_tick(config, &mut state, 2, true);
        assert_eq!(state.scale_q, 990);
    }

    #[test]
    fn scale_increases_below_target() {
        let config = HomeostasisConfig {
            target_rate_q: 5,
            window_steps: 2,
            ..base_config()
        };
        let mut state = HomeostasisState::default();
        homeostasis_tick(config, &mut state, 0, true);
        homeostasis_tick(config, &mut state, 0, true);
        assert_eq!(state.scale_q, 1010);
    }

    #[test]
    fn scale_clamped_to_bounds() {
        let config = HomeostasisConfig {
            target_rate_q: 0,
            window_steps: 1,
            gain_down_q: 200,
            scale_min_q: 900,
            ..base_config()
        };
        let mut state = HomeostasisState::default();
        homeostasis_tick(config, &mut state, 10, true);
        homeostasis_tick(config, &mut state, 10, true);
        assert_eq!(state.scale_q, 900);
    }

    #[test]
    fn replay_only_gate_blocks_updates() {
        let config = HomeostasisConfig {
            enabled: true,
            mode: HomeoMode::REPLAY_ONLY,
            target_rate_q: 1,
            window_steps: 1,
            ..base_config()
        };
        let mut state = HomeostasisState::default();
        homeostasis_tick(config, &mut state, 10, false);
        assert_eq!(state.scale_q, 1000);
        assert_eq!(state.step_in_window, 0);
        assert_eq!(state.spike_count_window, 0);
    }
}
