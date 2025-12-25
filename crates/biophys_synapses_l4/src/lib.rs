#![forbid(unsafe_code)]

#[cfg(feature = "biophys-l4-modulation")]
use biophys_core::ModLevel;
use biophys_core::{level_mul, ModChannel, ModulatorField, STP_SCALE};
use biophys_plasticity_l4::{StdpConfig, StdpTrace, TRACE_SCALE_Q};

pub const FIXED_POINT_SCALE: u32 = 1 << 16;
const FIXED_POINT_SCALE_I64: i64 = 1 << 16;
const DECAY_SCALE: u32 = 1024;
const MAX_SYNAPSE_G: f32 = 1000.0;
const MAX_ACCUMULATOR_G: f32 = 5000.0;
/// STDP delta scaling: dw_q (0..1000) maps to Q16.16 by shifting left 6 bits.
pub const STDP_WEIGHT_SHIFT: u32 = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SynKind {
    AMPA,
    NMDA,
    GABA,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum NmdaVDepMode {
    #[default]
    PiecewiseLinear,
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StpMode {
    STP_NONE = 0,
    STP_TM = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StpParamsL4 {
    pub mode: StpMode,
    pub u_base_q: u16,
    pub tau_rec_steps: u16,
    pub tau_fac_steps: u16,
}

impl StpParamsL4 {
    pub fn disabled() -> Self {
        Self {
            mode: StpMode::STP_NONE,
            u_base_q: 0,
            tau_rec_steps: 1,
            tau_fac_steps: 1,
        }
    }
}

impl Default for StpParamsL4 {
    fn default() -> Self {
        Self::disabled()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StpStateL4 {
    pub x_q: u16,
    pub u_q: u16,
}

impl StpStateL4 {
    pub fn new(params: StpParamsL4) -> Self {
        Self {
            x_q: STP_SCALE,
            u_q: params.u_base_q.min(STP_SCALE),
        }
    }
}

impl Default for StpStateL4 {
    fn default() -> Self {
        Self {
            x_q: STP_SCALE,
            u_q: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SynapseL4 {
    pub pre_neuron: u32,
    pub post_neuron: u32,
    pub post_compartment: u32,
    pub kind: SynKind,
    pub mod_channel: ModChannel,
    pub g_max_base_q: u32,
    pub g_nmda_base_q: u32,
    pub g_max_min_q: u32,
    pub g_max_max_q: u32,
    pub e_rev: f32,
    pub tau_rise_ms: f32,
    pub tau_decay_ms: f32,
    pub tau_decay_nmda_steps: u16,
    pub nmda_vdep_mode: NmdaVDepMode,
    pub delay_steps: u16,
    pub stp_params: StpParamsL4,
    pub stp_state: StpStateL4,
    pub stdp_enabled: bool,
    pub stdp_trace: StdpTrace,
}

impl SynapseL4 {
    pub fn g_max_base_fixed(&self) -> u32 {
        match self.kind {
            SynKind::NMDA => self.g_nmda_base_q,
            _ => self.g_max_base_q,
        }
    }

    pub fn effective_g_max_fixed(&self, mods: ModulatorField) -> u32 {
        let mult = mod_channel_multiplier(self.mod_channel, self.kind, mods);
        let base = self.g_max_base_fixed();
        let scaled = (base as u64 * mult as u64) / 100;
        let max_fixed = max_synapse_g_fixed();
        scaled.min(max_fixed as u64) as u32
    }

    pub fn e_rev_fixed(&self) -> i32 {
        f32_to_fixed_i32(self.e_rev)
    }

    pub fn stp_effective_params(&self, mods: ModulatorField) -> StpParamsL4 {
        let mut params = self.stp_params;
        if params.mode != StpMode::STP_TM {
            return params;
        }
        #[cfg(not(feature = "biophys-l4-modulation"))]
        let _mods = mods;
        params.u_base_q = params.u_base_q.min(STP_SCALE);
        params.tau_rec_steps = params.tau_rec_steps.max(1);
        params.tau_fac_steps = params.tau_fac_steps.max(1);

        #[cfg(feature = "biophys-l4-modulation")]
        {
            if mods.da == ModLevel::High {
                params.u_base_q = params.u_base_q.saturating_add(50).min(STP_SCALE);
            }
            if mods.ht == ModLevel::High {
                params.tau_rec_steps = params.tau_rec_steps.saturating_add(2).max(1);
            }
        }

        params
    }

    pub fn stp_recover_tick(&mut self, params: StpParamsL4) {
        if self.kind != SynKind::AMPA || params.mode != StpMode::STP_TM {
            return;
        }
        let u_base = params.u_base_q.min(STP_SCALE);
        let tau_rec = params.tau_rec_steps.max(1);
        let tau_fac = params.tau_fac_steps.max(1);

        let x_q = self.stp_state.x_q.min(STP_SCALE);
        let increment = (STP_SCALE - x_q) as u32 / tau_rec as u32;
        let x_next = (x_q as u32 + increment).min(STP_SCALE as u32) as u16;

        let u_q = self.stp_state.u_q.min(STP_SCALE);
        let delta = (u_base as i32 - u_q as i32) / tau_fac as i32;
        let u_next = clamp_stp_q(u_q as i32 + delta);

        self.stp_state.x_q = x_next;
        self.stp_state.u_q = u_next;
    }

    pub fn stp_release_on_spike(&mut self, params: StpParamsL4) -> u16 {
        if self.kind != SynKind::AMPA || params.mode != StpMode::STP_TM {
            return STP_SCALE;
        }
        let u_base = params.u_base_q.min(STP_SCALE);
        let mut u_q = self.stp_state.u_q.min(STP_SCALE);
        let mut x_q = self.stp_state.x_q.min(STP_SCALE);

        let delta_u = (u_base as u32 * (STP_SCALE - u_q) as u32) / STP_SCALE as u32;
        u_q = (u_q as u32 + delta_u).min(STP_SCALE as u32) as u16;
        let released_q = (u_q as u32 * x_q as u32) / STP_SCALE as u32;
        x_q = x_q.saturating_sub(released_q as u16);

        self.stp_state.u_q = u_q;
        self.stp_state.x_q = x_q;

        released_q.min(STP_SCALE as u32) as u16
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SynapseState {
    pub g_ampa_q: u32,
    pub g_nmda_q: u32,
    pub g_gaba_q: u32,
}

impl SynapseState {
    pub fn g_fixed_for(&self, kind: SynKind) -> u32 {
        match kind {
            SynKind::AMPA => self.g_ampa_q,
            SynKind::NMDA => self.g_nmda_q,
            SynKind::GABA => self.g_gaba_q,
        }
    }

    pub fn apply_spike(&mut self, kind: SynKind, g_max_eff_fixed: u32, release_gain_q: u16) {
        let max_fixed = max_synapse_g_fixed();
        let add_fixed = g_max_eff_fixed.min(max_fixed);
        let scaled_add = (add_fixed as u64 * release_gain_q as u64) / STP_SCALE as u64;
        let target = match kind {
            SynKind::AMPA => &mut self.g_ampa_q,
            SynKind::NMDA => &mut self.g_nmda_q,
            SynKind::GABA => &mut self.g_gaba_q,
        };
        *target = target.saturating_add(scaled_add as u32).min(max_fixed);
    }

    pub fn decay(&mut self, kind: SynKind, decay_k: u16, tau_decay_nmda_steps: u16) {
        if kind == SynKind::NMDA {
            let tau_steps = tau_decay_nmda_steps.max(1) as u32;
            let decay = self.g_nmda_q / tau_steps;
            self.g_nmda_q = self.g_nmda_q.saturating_sub(decay);
            return;
        }
        if decay_k as u32 >= DECAY_SCALE {
            match kind {
                SynKind::AMPA => self.g_ampa_q = 0,
                SynKind::GABA => self.g_gaba_q = 0,
                SynKind::NMDA => {}
            }
            return;
        }
        let decay_target = match kind {
            SynKind::AMPA => &mut self.g_ampa_q,
            SynKind::GABA => &mut self.g_gaba_q,
            SynKind::NMDA => &mut self.g_nmda_q,
        };
        let decay = (*decay_target as u64 * decay_k as u64) / DECAY_SCALE as u64;
        *decay_target = decay_target.saturating_sub(decay as u32);
    }
}

pub fn apply_stdp_updates(
    synapses: &mut [SynapseL4],
    spike_flags: &[bool],
    traces: &[StdpTrace],
    config: StdpConfig,
) {
    if spike_flags.len() != traces.len() {
        return;
    }
    let scale = TRACE_SCALE_Q as u32;
    for synapse in synapses.iter_mut() {
        if !synapse.stdp_enabled {
            continue;
        }
        if synapse.kind == SynKind::NMDA {
            continue;
        }
        let pre = synapse.pre_neuron as usize;
        if pre >= spike_flags.len() {
            continue;
        }
        if !spike_flags[pre] {
            continue;
        }
        let post = synapse.post_neuron as usize;
        if post >= traces.len() {
            continue;
        }
        let post_trace = traces[post].post_trace_q.min(TRACE_SCALE_Q) as u32;
        let dw_q = (config.a_minus_q as u32 * post_trace) / scale;
        let dw_g_q = dw_q << STDP_WEIGHT_SHIFT;
        let updated = synapse.g_max_base_q.saturating_sub(dw_g_q);
        synapse.g_max_base_q = updated.max(synapse.g_max_min_q).min(synapse.g_max_max_q);
    }
    for synapse in synapses.iter_mut() {
        if !synapse.stdp_enabled {
            continue;
        }
        if synapse.kind == SynKind::NMDA {
            continue;
        }
        let post = synapse.post_neuron as usize;
        if post >= spike_flags.len() {
            continue;
        }
        if !spike_flags[post] {
            continue;
        }
        let pre = synapse.pre_neuron as usize;
        if pre >= traces.len() {
            continue;
        }
        let pre_trace = traces[pre].pre_trace_q.min(TRACE_SCALE_Q) as u32;
        let dw_q = (config.a_plus_q as u32 * pre_trace) / scale;
        let dw_g_q = dw_q << STDP_WEIGHT_SHIFT;
        let updated = synapse.g_max_base_q.saturating_add(dw_g_q);
        synapse.g_max_base_q = updated.max(synapse.g_max_min_q).min(synapse.g_max_max_q);
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SynapseConductance {
    pub g_fixed: u32,
    pub g_e_rev_fixed: i64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SynapseAccumulator {
    pub ampa: SynapseConductance,
    pub nmda: SynapseConductance,
    pub gaba: SynapseConductance,
    pub nmda_vdep_mode: NmdaVDepMode,
}

impl SynapseAccumulator {
    pub fn add(&mut self, kind: SynKind, g_fixed: u32, e_rev: f32, nmda_vdep_mode: NmdaVDepMode) {
        let e_rev_fixed = f32_to_fixed_i32(e_rev) as i64;
        let max_fixed = f32_to_fixed_u32(MAX_ACCUMULATOR_G);
        let target = match kind {
            SynKind::AMPA => &mut self.ampa,
            SynKind::NMDA => &mut self.nmda,
            SynKind::GABA => &mut self.gaba,
        };
        let remaining = max_fixed.saturating_sub(target.g_fixed);
        let add_g = g_fixed.min(remaining);
        if add_g == 0 {
            return;
        }
        target.g_fixed = target.g_fixed.saturating_add(add_g);
        let delta = (add_g as i64 * e_rev_fixed) >> 16;
        target.g_e_rev_fixed = target.g_e_rev_fixed.saturating_add(delta);
        if kind == SynKind::NMDA {
            self.nmda_vdep_mode = nmda_vdep_mode;
        }
    }

    pub fn total_current(&self, v: f32) -> f32 {
        syn_current(self.ampa, v)
            + syn_current_nmda(self.nmda, v, self.nmda_vdep_mode)
            + syn_current(self.gaba, v)
    }
}

pub fn decay_k(dt_ms: f32, tau_decay_ms: f32) -> u16 {
    if tau_decay_ms <= 0.0 {
        return DECAY_SCALE as u16;
    }
    let ratio = dt_ms / tau_decay_ms;
    let scaled = (ratio * DECAY_SCALE as f32).round();
    scaled.clamp(0.0, DECAY_SCALE as f32) as u16
}

pub fn max_synapse_g_fixed() -> u32 {
    f32_to_fixed_u32(MAX_SYNAPSE_G)
}

fn syn_current(conductance: SynapseConductance, v: f32) -> f32 {
    let g = fixed_to_f32(conductance.g_fixed);
    if g == 0.0 {
        return 0.0;
    }
    let g_e_rev = fixed_to_f32_i64(conductance.g_e_rev_fixed);
    g_e_rev - g * v
}

fn syn_current_nmda(conductance: SynapseConductance, v: f32, nmda_vdep_mode: NmdaVDepMode) -> f32 {
    if conductance.g_fixed == 0 {
        return 0.0;
    }
    let alpha_q = nmda_alpha_q(v, nmda_vdep_mode) as u64;
    let g_eff_fixed = (conductance.g_fixed as u64 * alpha_q) / 1000;
    if g_eff_fixed == 0 {
        return 0.0;
    }
    let g_e_rev_eff = (conductance.g_e_rev_fixed * alpha_q as i64) / 1000;
    let g = g_eff_fixed as f32 / FIXED_POINT_SCALE as f32;
    let g_e_rev = fixed_to_f32_i64(g_e_rev_eff);
    g_e_rev - g * v
}

const NMDA_MIN_MV: i32 = -120;
const NMDA_MAX_MV: i32 = 60;
const NMDA_ALPHA_TABLE_LEN: usize = (NMDA_MAX_MV - NMDA_MIN_MV + 1) as usize;

const fn nmda_alpha_table() -> [u16; NMDA_ALPHA_TABLE_LEN] {
    let mut table = [0_u16; NMDA_ALPHA_TABLE_LEN];
    let mut idx = 0;
    while idx < NMDA_ALPHA_TABLE_LEN {
        let v_mv = NMDA_MIN_MV + idx as i32;
        let alpha = if v_mv <= -70 {
            200
        } else if v_mv >= -20 {
            1000
        } else {
            200 + ((v_mv + 70) * 800) / 50
        };
        table[idx] = alpha as u16;
        idx += 1;
    }
    table
}

const NMDA_ALPHA_TABLE: [u16; NMDA_ALPHA_TABLE_LEN] = nmda_alpha_table();

pub fn nmda_alpha_q(v: f32, mode: NmdaVDepMode) -> u16 {
    match mode {
        NmdaVDepMode::PiecewiseLinear => {
            let rounded = v.round() as i32;
            let clamped = rounded.clamp(NMDA_MIN_MV, NMDA_MAX_MV);
            let index = (clamped - NMDA_MIN_MV) as usize;
            NMDA_ALPHA_TABLE[index]
        }
    }
}

pub fn f32_to_fixed_u32(value: f32) -> u32 {
    if value <= 0.0 {
        return 0;
    }
    (value * FIXED_POINT_SCALE as f32).round() as u32
}

fn f32_to_fixed_i32(value: f32) -> i32 {
    (value * FIXED_POINT_SCALE as f32).round() as i32
}

fn fixed_to_f32(value: u32) -> f32 {
    value as f32 / FIXED_POINT_SCALE as f32
}

fn fixed_to_f32_i64(value: i64) -> f32 {
    value as f32 / FIXED_POINT_SCALE_I64 as f32
}

fn mod_channel_multiplier(channel: ModChannel, kind: SynKind, mods: ModulatorField) -> u32 {
    match kind {
        SynKind::AMPA | SynKind::NMDA => match channel {
            ModChannel::None => 100,
            ModChannel::Na => level_mul(mods.na) as u32,
            ModChannel::Da => level_mul(mods.da) as u32,
            ModChannel::Ht => 100,
            ModChannel::NaDa => {
                let na = level_mul(mods.na) as u64;
                let da = level_mul(mods.da) as u64;
                ((na * da) / 100) as u32
            }
        },
        SynKind::GABA => match channel {
            ModChannel::Ht => level_mul(mods.ht) as u32,
            _ => 100,
        },
    }
}

fn clamp_stp_q(value: i32) -> u16 {
    value.clamp(0, STP_SCALE as i32) as u16
}
