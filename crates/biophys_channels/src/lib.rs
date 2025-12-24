#![forbid(unsafe_code)]

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Leak {
    pub g: f32,
    pub e_rev: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NaK {
    pub g_na: f32,
    pub g_k: f32,
    pub e_na: f32,
    pub e_k: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GatingState {
    pub m: f32,
    pub h: f32,
    pub n: f32,
}

impl GatingState {
    pub fn from_voltage(v: f32) -> Self {
        let m = m_inf(v);
        let h = h_inf(v);
        let n = n_inf(v);
        Self { m, h, n }
    }

    pub fn update(&mut self, v: f32, dt_ms: f32) {
        self.m = euler_update(self.m, m_inf(v), tau_m(v), dt_ms);
        self.h = euler_update(self.h, h_inf(v), tau_h(v), dt_ms);
        self.n = euler_update(self.n, n_inf(v), tau_n(v), dt_ms);
        self.m = self.m.clamp(0.0, 1.0);
        self.h = self.h.clamp(0.0, 1.0);
        self.n = self.n.clamp(0.0, 1.0);
    }
}

pub fn leak_current(leak: Leak, v: f32) -> f32 {
    leak.g * (v - leak.e_rev)
}

pub fn nak_current(channel: NaK, gates: GatingState, v: f32) -> f32 {
    let m3 = gates.m * gates.m * gates.m;
    let n4 = gates.n * gates.n * gates.n * gates.n;
    let i_na = channel.g_na * m3 * gates.h * (v - channel.e_na);
    let i_k = channel.g_k * n4 * (v - channel.e_k);
    i_na + i_k
}

fn euler_update(x: f32, x_inf: f32, tau: f32, dt_ms: f32) -> f32 {
    let tau = tau.max(0.01);
    x + dt_ms * (x_inf - x) / tau
}

fn linear_clamped(v: f32, v_min: f32, v_max: f32) -> f32 {
    if v_min == v_max {
        return 0.0;
    }
    ((v - v_min) / (v_max - v_min)).clamp(0.0, 1.0)
}

fn m_inf(v: f32) -> f32 {
    linear_clamped(v, -60.0, -20.0)
}

fn h_inf(v: f32) -> f32 {
    1.0 - linear_clamped(v, -70.0, -40.0)
}

fn n_inf(v: f32) -> f32 {
    linear_clamped(v, -55.0, -25.0)
}

fn tau_m(v: f32) -> f32 {
    0.5 + 2.0 * (1.0 - linear_clamped(v, -60.0, -20.0))
}

fn tau_h(v: f32) -> f32 {
    1.0 + 4.0 * linear_clamped(v, -80.0, -40.0)
}

fn tau_n(v: f32) -> f32 {
    1.0 + 3.0 * (1.0 - linear_clamped(v, -55.0, -25.0))
}
