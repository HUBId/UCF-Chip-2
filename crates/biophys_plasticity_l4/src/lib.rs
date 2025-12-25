#![forbid(unsafe_code)]

use blake3::Hasher;

pub const TRACE_SCALE_Q: u16 = 1000;
pub const TRACE_SPIKE_INCREMENT_Q: u16 = 300;

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum LearningMode {
    OFF = 0,
    REPLAY_ONLY = 1,
    ALWAYS = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StdpConfig {
    pub enabled: bool,
    pub learning_mode: LearningMode,
    pub a_plus_q: u16,
    pub a_minus_q: u16,
    pub tau_plus_steps: u16,
    pub tau_minus_steps: u16,
    pub w_min: i32,
    pub w_max: i32,
}

impl Default for StdpConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            learning_mode: LearningMode::OFF,
            a_plus_q: 0,
            a_minus_q: 0,
            tau_plus_steps: 1,
            tau_minus_steps: 1,
            w_min: 0,
            w_max: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StdpTrace {
    pub pre_trace_q: u16,
    pub post_trace_q: u16,
}

impl StdpTrace {
    pub fn decay_tick(&mut self, tau_plus_steps: u16, tau_minus_steps: u16) {
        self.pre_trace_q = decay_trace_q(self.pre_trace_q, tau_plus_steps);
        self.post_trace_q = decay_trace_q(self.post_trace_q, tau_minus_steps);
    }

    pub fn on_pre_spike(&mut self) {
        self.pre_trace_q = self
            .pre_trace_q
            .saturating_add(TRACE_SPIKE_INCREMENT_Q)
            .min(TRACE_SCALE_Q);
    }

    pub fn on_post_spike(&mut self) {
        self.post_trace_q = self
            .post_trace_q
            .saturating_add(TRACE_SPIKE_INCREMENT_Q)
            .min(TRACE_SCALE_Q);
    }
}

pub fn plasticity_snapshot_digest(step_count: u64, g_max_values: &[u32]) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"UCF:BIO:L4:PLASTICITY:SNAP");
    for value in g_max_values {
        hasher.update(&value.to_le_bytes());
    }
    hasher.update(&step_count.to_le_bytes());
    *hasher.finalize().as_bytes()
}

fn decay_trace_q(value: u16, tau_steps: u16) -> u16 {
    let tau = tau_steps.max(1) as u32;
    let current = value.min(TRACE_SCALE_Q) as u32;
    let decay = current / tau;
    current.saturating_sub(decay).min(TRACE_SCALE_Q as u32) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_decay_and_spike_update() {
        let mut trace = StdpTrace {
            pre_trace_q: 1000,
            post_trace_q: 500,
        };
        trace.decay_tick(2, 4);
        assert_eq!(trace.pre_trace_q, 500);
        assert_eq!(trace.post_trace_q, 375);
        trace.on_pre_spike();
        trace.on_post_spike();
        assert_eq!(trace.pre_trace_q, 800);
        assert_eq!(trace.post_trace_q, 675);
    }

    #[test]
    fn snapshot_digest_stable() {
        let values = [10u32, 20u32, 30u32];
        let digest_a = plasticity_snapshot_digest(5, &values);
        let digest_b = plasticity_snapshot_digest(5, &values);
        assert_eq!(digest_a, digest_b);
    }
}
