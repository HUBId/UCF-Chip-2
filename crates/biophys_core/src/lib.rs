#![forbid(unsafe_code)]

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NeuronId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SynapseId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CompartmentId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LifParams {
    pub tau_ms: u16,
    pub v_rest: i32,
    pub v_reset: i32,
    pub v_threshold: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LifState {
    pub v: i32,
    pub refractory_steps: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SynapseEdge {
    pub pre: NeuronId,
    pub post: NeuronId,
    pub weight: i32,
    pub delay_steps: u16,
    pub stp: StpState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StpParams {
    pub u: u16,
    pub tau_rec_steps: u16,
    pub tau_fac_steps: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StpState {
    pub x: u16,
    pub u: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PopCode {
    pub spikes: Vec<NeuronId>,
}

pub const DEFAULT_REFRACTORY_STEPS: u16 = 0;
pub const STP_SCALE: u16 = 1000;

pub fn clamp_i32(value: i32, min: i32, max: i32) -> i32 {
    value.max(min).min(max)
}

pub fn clamp_usize(value: usize, max: usize) -> usize {
    if value > max {
        max
    } else {
        value
    }
}

fn clamp_u16(value: i32, min: u16, max: u16) -> u16 {
    value.clamp(min as i32, max as i32) as u16
}

impl StpState {
    pub fn update_between_spikes(&mut self, params: StpParams) {
        if params.tau_rec_steps > 0 {
            let increment = (STP_SCALE.saturating_sub(self.x)) / params.tau_rec_steps;
            self.x = clamp_u16(self.x as i32 + increment as i32, 0, STP_SCALE);
        } else {
            self.x = STP_SCALE;
        }

        if params.tau_fac_steps > 0 {
            let delta = (params.u as i32 - self.u as i32) / params.tau_fac_steps as i32;
            self.u = clamp_u16(self.u as i32 + delta, 0, STP_SCALE);
        } else {
            self.u = clamp_u16(params.u as i32, 0, STP_SCALE);
        }
    }

    pub fn on_spike(&mut self, params: StpParams) -> u16 {
        let available = (STP_SCALE.saturating_sub(self.u)) as u32;
        let increment = (params.u as u32 * available) / STP_SCALE as u32;
        let updated_u = self.u as u32 + increment;
        self.u = clamp_u16(updated_u as i32, 0, STP_SCALE);

        let released = (self.u as u32 * self.x as u32) / STP_SCALE as u32;
        let remaining_x = self.x as i32 - released as i32;
        self.x = clamp_u16(remaining_x, 0, STP_SCALE);
        released as u16
    }
}
