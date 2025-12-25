#![cfg(all(
    feature = "biophys",
    feature = "biophys-l4",
    feature = "biophys-l4-synapses",
    feature = "biophys-l4-plasticity"
))]

use biophys_core::{ModLevel, ModulatorField};
use biophys_plasticity_l4::{plasticity_snapshot_digest, LearningMode, StdpConfig, StdpTrace};
use biophys_synapses_l4::{
    apply_stdp_updates, f32_to_fixed_u32, max_synapse_g_fixed, SynKind, SynapseL4,
};

struct PlasticityHarness {
    synapses: Vec<SynapseL4>,
    traces: Vec<StdpTrace>,
    spike_flags: Vec<bool>,
    config: StdpConfig,
    step_count: u64,
    learning_enabled: bool,
    in_replay_mode: bool,
}

impl PlasticityHarness {
    fn new(neuron_count: usize, synapses: Vec<SynapseL4>, config: StdpConfig) -> Self {
        Self {
            synapses,
            traces: vec![StdpTrace::default(); neuron_count],
            spike_flags: vec![false; neuron_count],
            config,
            step_count: 0,
            learning_enabled: false,
            in_replay_mode: false,
        }
    }

    fn set_learning_context(&mut self, in_replay: bool, mods: ModulatorField, reward_block: bool) {
        self.in_replay_mode = in_replay;
        if !self.config.enabled {
            self.learning_enabled = false;
            return;
        }
        let mode_allowed = match self.config.learning_mode {
            LearningMode::OFF => false,
            LearningMode::REPLAY_ONLY => in_replay,
            LearningMode::ALWAYS => true,
        };
        let da_allowed = matches!(mods.da, ModLevel::Med | ModLevel::High);
        self.learning_enabled = mode_allowed && da_allowed && !reward_block;
    }

    fn tick(&mut self, spikes: &[usize]) {
        for trace in &mut self.traces {
            trace.decay_tick(self.config.tau_plus_steps, self.config.tau_minus_steps);
        }
        for &idx in spikes {
            if let Some(trace) = self.traces.get_mut(idx) {
                trace.on_pre_spike();
                trace.on_post_spike();
            }
        }
        if self.learning_enabled {
            self.spike_flags.fill(false);
            for &idx in spikes {
                if let Some(flag) = self.spike_flags.get_mut(idx) {
                    *flag = true;
                }
            }
            apply_stdp_updates(
                &mut self.synapses,
                &self.spike_flags,
                &self.traces,
                self.config,
            );
        }
        self.step_count = self.step_count.saturating_add(1);
    }

    fn digest(&self) -> [u8; 32] {
        let g_max_values = self
            .synapses
            .iter()
            .map(|synapse| synapse.g_max_base_q)
            .collect::<Vec<_>>();
        plasticity_snapshot_digest(self.step_count, &g_max_values)
    }
}

fn build_synapse(pre: u32, post: u32, g_max: f32, min: f32, max: f32) -> SynapseL4 {
    SynapseL4 {
        pre_neuron: pre,
        post_neuron: post,
        post_compartment: 0,
        kind: SynKind::AMPA,
        mod_channel: biophys_core::ModChannel::None,
        g_max_base_q: f32_to_fixed_u32(g_max),
        g_max_min_q: f32_to_fixed_u32(min),
        g_max_max_q: f32_to_fixed_u32(max).min(max_synapse_g_fixed()),
        e_rev: 0.0,
        tau_rise_ms: 0.0,
        tau_decay_ms: 8.0,
        delay_steps: 0,
        stp_params: Default::default(),
        stp_state: Default::default(),
        stdp_enabled: true,
        stdp_trace: StdpTrace::default(),
    }
}

#[test]
fn deterministic_runs_match_and_digest_is_stable() {
    let config = StdpConfig {
        enabled: true,
        learning_mode: LearningMode::ALWAYS,
        a_plus_q: 400,
        a_minus_q: 200,
        tau_plus_steps: 10,
        tau_minus_steps: 12,
        w_min: 0,
        w_max: 0,
    };
    let synapses = vec![build_synapse(0, 1, 2.0, 0.0, 10.0)];
    let spikes = vec![vec![0usize], vec![1usize], vec![0usize, 1usize], vec![]];

    let run = || {
        let mut harness = PlasticityHarness::new(2, synapses.clone(), config);
        harness.set_learning_context(
            false,
            ModulatorField {
                da: ModLevel::High,
                ..Default::default()
            },
            false,
        );
        for tick in &spikes {
            harness.tick(tick);
        }
        (harness.synapses[0].g_max_base_q, harness.digest())
    };

    let (weight_a, digest_a) = run();
    let (weight_b, digest_b) = run();

    assert_eq!(weight_a, weight_b);
    assert_eq!(digest_a, digest_b);
}

#[test]
fn replay_only_gate_controls_learning() {
    let config = StdpConfig {
        enabled: true,
        learning_mode: LearningMode::REPLAY_ONLY,
        a_plus_q: 500,
        a_minus_q: 200,
        tau_plus_steps: 8,
        tau_minus_steps: 8,
        w_min: 0,
        w_max: 0,
    };
    let synapses = vec![build_synapse(0, 1, 2.0, 0.0, 10.0)];

    let mut no_replay = PlasticityHarness::new(2, synapses.clone(), config);
    no_replay.set_learning_context(
        false,
        ModulatorField {
            da: ModLevel::High,
            ..Default::default()
        },
        false,
    );
    no_replay.tick(&[0]);
    no_replay.tick(&[1]);
    let weight_no_replay = no_replay.synapses[0].g_max_base_q;

    let mut replay = PlasticityHarness::new(2, synapses, config);
    replay.set_learning_context(
        true,
        ModulatorField {
            da: ModLevel::High,
            ..Default::default()
        },
        false,
    );
    replay.tick(&[0]);
    replay.tick(&[1]);
    let weight_replay = replay.synapses[0].g_max_base_q;

    assert_eq!(weight_no_replay, f32_to_fixed_u32(2.0));
    assert!(weight_replay > weight_no_replay);
}

#[test]
fn dopamine_gate_blocks_learning() {
    let config = StdpConfig {
        enabled: true,
        learning_mode: LearningMode::ALWAYS,
        a_plus_q: 500,
        a_minus_q: 200,
        tau_plus_steps: 6,
        tau_minus_steps: 6,
        w_min: 0,
        w_max: 0,
    };
    let synapses = vec![build_synapse(0, 1, 2.0, 0.0, 10.0)];

    let mut low_da = PlasticityHarness::new(2, synapses.clone(), config);
    low_da.set_learning_context(
        false,
        ModulatorField {
            da: ModLevel::Low,
            ..Default::default()
        },
        false,
    );
    low_da.tick(&[0]);
    low_da.tick(&[1]);
    let weight_low = low_da.synapses[0].g_max_base_q;

    let mut high_da = PlasticityHarness::new(2, synapses, config);
    high_da.set_learning_context(
        false,
        ModulatorField {
            da: ModLevel::High,
            ..Default::default()
        },
        false,
    );
    high_da.tick(&[0]);
    high_da.tick(&[1]);
    let weight_high = high_da.synapses[0].g_max_base_q;

    assert_eq!(weight_low, f32_to_fixed_u32(2.0));
    assert!(weight_high > weight_low);
}

#[test]
fn reward_block_disables_learning() {
    let config = StdpConfig {
        enabled: true,
        learning_mode: LearningMode::ALWAYS,
        a_plus_q: 500,
        a_minus_q: 200,
        tau_plus_steps: 6,
        tau_minus_steps: 6,
        w_min: 0,
        w_max: 0,
    };
    let synapses = vec![build_synapse(0, 1, 2.0, 0.0, 10.0)];

    let mut blocked = PlasticityHarness::new(2, synapses, config);
    blocked.set_learning_context(
        false,
        ModulatorField {
            da: ModLevel::High,
            ..Default::default()
        },
        true,
    );
    blocked.tick(&[0]);
    blocked.tick(&[1]);

    assert_eq!(blocked.synapses[0].g_max_base_q, f32_to_fixed_u32(2.0));
}

#[test]
fn weights_stay_within_bounds() {
    let config = StdpConfig {
        enabled: true,
        learning_mode: LearningMode::ALWAYS,
        a_plus_q: 1000,
        a_minus_q: 1000,
        tau_plus_steps: 2,
        tau_minus_steps: 2,
        w_min: 0,
        w_max: 0,
    };
    let synapses = vec![build_synapse(0, 1, 1.0, 0.5, 1.0)];
    let mut harness = PlasticityHarness::new(2, synapses, config);
    harness.set_learning_context(
        false,
        ModulatorField {
            da: ModLevel::High,
            ..Default::default()
        },
        false,
    );

    for _ in 0..5 {
        harness.tick(&[0]);
        harness.tick(&[1]);
    }

    let weight = harness.synapses[0].g_max_base_q;
    assert_eq!(weight, f32_to_fixed_u32(1.0));

    for _ in 0..5 {
        harness.tick(&[1]);
        harness.tick(&[0]);
    }

    let weight_after_ltd = harness.synapses[0].g_max_base_q;
    assert!(weight_after_ltd >= f32_to_fixed_u32(0.5));
}

#[test]
fn stdp_is_directional_for_pre_post_pairing() {
    let config = StdpConfig {
        enabled: true,
        learning_mode: LearningMode::ALWAYS,
        a_plus_q: 400,
        a_minus_q: 200,
        tau_plus_steps: 10,
        tau_minus_steps: 10,
        w_min: 0,
        w_max: 0,
    };

    let synapses = vec![build_synapse(0, 1, 2.0, 0.0, 10.0)];
    let mut paired = PlasticityHarness::new(2, synapses.clone(), config);
    paired.set_learning_context(
        false,
        ModulatorField {
            da: ModLevel::High,
            ..Default::default()
        },
        false,
    );
    for _ in 0..5 {
        paired.tick(&[0]);
        paired.tick(&[1]);
    }
    let paired_weight = paired.synapses[0].g_max_base_q;

    let mut post_only = PlasticityHarness::new(2, synapses, config);
    post_only.set_learning_context(
        false,
        ModulatorField {
            da: ModLevel::High,
            ..Default::default()
        },
        false,
    );
    for _ in 0..5 {
        post_only.tick(&[1]);
    }
    let post_only_weight = post_only.synapses[0].g_max_base_q;

    assert!(paired_weight > post_only_weight);
}
