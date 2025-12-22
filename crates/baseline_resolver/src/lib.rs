#![forbid(unsafe_code)]

use dbm_core::{BaselineVector, CooldownClass, IntegrityState, LevelClass};
use ucf::v1::{CharacterBaselineVector, PolicyEcologyVector};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct HbvOffsets {
    pub baseline_caution_offset: i32,
    pub baseline_novelty_dampening_offset: i32,
    pub baseline_approval_strictness_offset: i32,
    pub baseline_export_strictness_offset: i32,
    pub baseline_chain_conservatism_offset: i32,
    pub baseline_cooldown_multiplier_class: i32,
    pub reward_block_bias: Option<LevelClass>,
    pub reason_codes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct BaselineInputs {
    pub cbv: Option<CharacterBaselineVector>,
    pub pev: Option<PolicyEcologyVector>,
    pub hbv: Option<HbvOffsets>,
    pub integrity: Option<IntegrityState>,
}

pub fn resolve_baseline(inputs: &BaselineInputs) -> BaselineVector {
    let mut baseline = BaselineVector::default();

    if let Some(hbv) = &inputs.hbv {
        apply_offsets(
            &mut baseline,
            hbv.baseline_caution_offset,
            hbv.baseline_novelty_dampening_offset,
            hbv.baseline_approval_strictness_offset,
            hbv.baseline_export_strictness_offset,
            hbv.baseline_chain_conservatism_offset,
            hbv.baseline_cooldown_multiplier_class,
        );

        if let Some(bias) = hbv.reward_block_bias {
            baseline.reward_block_bias = level_max(baseline.reward_block_bias, bias);
        }

        baseline.reason_codes.extend(hbv.reason_codes.clone());
    }

    if let Some(cbv) = &inputs.cbv {
        apply_offsets(
            &mut baseline,
            cbv.baseline_caution_offset,
            cbv.baseline_novelty_dampening_offset,
            cbv.baseline_approval_strictness_offset,
            cbv.baseline_export_strictness_offset,
            cbv.baseline_chain_conservatism_offset,
            cbv.baseline_cooldown_multiplier_class,
        );

        baseline.reason_codes.insert("cbv_influence".to_string());
    }

    if let Some(pev) = &inputs.pev {
        apply_offsets(
            &mut baseline,
            pev.conservatism_bias,
            pev.novelty_penalty_bias,
            pev.manipulation_aversion_bias,
            pev.manipulation_aversion_bias,
            pev.reversibility_bias,
            pev.conservatism_bias,
        );

        baseline.reason_codes.insert("pev_influence".to_string());
    }

    if let Some(integrity) = inputs.integrity {
        let integrity_level = match integrity {
            IntegrityState::Ok => LevelClass::Low,
            IntegrityState::Degraded => LevelClass::Med,
            IntegrityState::Fail => LevelClass::High,
        };

        baseline.caution_floor = level_max(baseline.caution_floor, integrity_level);
        baseline.novelty_dampening = level_max(baseline.novelty_dampening, integrity_level);
        baseline.approval_strictness = level_max(baseline.approval_strictness, integrity_level);
        baseline.export_strictness = level_max(baseline.export_strictness, integrity_level);
        baseline.chain_conservatism = level_max(baseline.chain_conservatism, integrity_level);
        baseline.cooldown_bias = cooldown_max(
            baseline.cooldown_bias,
            cooldown_from_offset(integrity_level as i32),
        );
        baseline.reward_block_bias = level_max(baseline.reward_block_bias, integrity_level);

        baseline.reason_codes.insert(match integrity {
            IntegrityState::Ok => "integrity_ok".to_string(),
            IntegrityState::Degraded => "integrity_degraded".to_string(),
            IntegrityState::Fail => "integrity_fail".to_string(),
        });
    }

    aggregate_reward_block_bias(&mut baseline);
    baseline
}

fn apply_offsets(
    baseline: &mut BaselineVector,
    caution_offset: i32,
    novelty_offset: i32,
    approval_offset: i32,
    export_offset: i32,
    chain_offset: i32,
    cooldown_class: i32,
) {
    baseline.caution_floor = level_max(baseline.caution_floor, level_from_offset(caution_offset));
    baseline.novelty_dampening = level_max(
        baseline.novelty_dampening,
        level_from_offset(novelty_offset),
    );
    baseline.approval_strictness = level_max(
        baseline.approval_strictness,
        level_from_offset(approval_offset),
    );
    baseline.export_strictness =
        level_max(baseline.export_strictness, level_from_offset(export_offset));
    baseline.chain_conservatism =
        level_max(baseline.chain_conservatism, level_from_offset(chain_offset));
    baseline.cooldown_bias =
        cooldown_max(baseline.cooldown_bias, cooldown_from_offset(cooldown_class));
}

fn level_from_offset(offset: i32) -> LevelClass {
    match offset.cmp(&0) {
        std::cmp::Ordering::Less => LevelClass::Low,
        std::cmp::Ordering::Equal => LevelClass::Low,
        std::cmp::Ordering::Greater => {
            if offset >= 2 {
                LevelClass::High
            } else {
                LevelClass::Med
            }
        }
    }
}

fn level_max(a: LevelClass, b: LevelClass) -> LevelClass {
    if (a as i32) >= (b as i32) {
        a
    } else {
        b
    }
}

fn cooldown_from_offset(offset: i32) -> CooldownClass {
    if offset > 0 {
        CooldownClass::Longer
    } else {
        CooldownClass::Base
    }
}

fn cooldown_max(a: CooldownClass, b: CooldownClass) -> CooldownClass {
    if matches!(a, CooldownClass::Longer) || matches!(b, CooldownClass::Longer) {
        CooldownClass::Longer
    } else {
        CooldownClass::Base
    }
}

fn aggregate_reward_block_bias(baseline: &mut BaselineVector) {
    let mut bias = baseline.reward_block_bias;
    bias = level_max(bias, baseline.caution_floor);
    bias = level_max(bias, baseline.novelty_dampening);
    bias = level_max(bias, baseline.approval_strictness);
    bias = level_max(bias, baseline.export_strictness);
    bias = level_max(bias, baseline.chain_conservatism);
    baseline.reward_block_bias = bias;
}

#[cfg(test)]
mod tests {
    use super::*;
    use dbm_core::ReasonSet;

    fn hbv_offsets() -> HbvOffsets {
        HbvOffsets {
            baseline_caution_offset: 1,
            baseline_novelty_dampening_offset: 0,
            baseline_approval_strictness_offset: 0,
            baseline_export_strictness_offset: 0,
            baseline_chain_conservatism_offset: 0,
            baseline_cooldown_multiplier_class: 0,
            reward_block_bias: Some(LevelClass::Med),
            reason_codes: vec!["hbv_reason_b".into(), "hbv_reason_a".into()],
        }
    }

    #[test]
    fn max_merge_prefers_higher_offset() {
        let mut inputs = BaselineInputs::default();
        inputs.hbv = Some(hbv_offsets());
        inputs.cbv = Some(CharacterBaselineVector {
            baseline_caution_offset: 3,
            baseline_novelty_dampening_offset: 0,
            baseline_approval_strictness_offset: 0,
            baseline_export_strictness_offset: 0,
            baseline_chain_conservatism_offset: 0,
            baseline_cooldown_multiplier_class: 0,
        });

        let baseline = resolve_baseline(&inputs);

        assert_eq!(baseline.caution_floor, LevelClass::High);
        assert_eq!(baseline.reward_block_bias, LevelClass::High);
    }

    #[test]
    fn negative_offsets_do_not_loosen_baseline() {
        let mut inputs = BaselineInputs::default();
        inputs.hbv = Some(HbvOffsets {
            reward_block_bias: Some(LevelClass::High),
            ..hbv_offsets()
        });
        inputs.cbv = Some(CharacterBaselineVector {
            baseline_caution_offset: -4,
            baseline_novelty_dampening_offset: -4,
            baseline_approval_strictness_offset: -4,
            baseline_export_strictness_offset: -4,
            baseline_chain_conservatism_offset: -4,
            baseline_cooldown_multiplier_class: -4,
        });

        let baseline = resolve_baseline(&inputs);

        assert_eq!(baseline.caution_floor, LevelClass::Med);
        assert_eq!(baseline.reward_block_bias, LevelClass::High);
    }

    #[test]
    fn resolution_is_deterministic() {
        let mut inputs = BaselineInputs::default();
        inputs.hbv = Some(hbv_offsets());
        inputs.pev = Some(PolicyEcologyVector {
            conservatism_bias: 1,
            novelty_penalty_bias: 0,
            manipulation_aversion_bias: 0,
            reversibility_bias: 2,
        });
        inputs.integrity = Some(IntegrityState::Degraded);

        let a = resolve_baseline(&inputs);
        let b = resolve_baseline(&inputs);

        assert_eq!(a, b);
    }

    #[test]
    fn reason_codes_are_sorted_and_truncated() {
        let mut inputs = BaselineInputs::default();
        inputs.hbv = Some(HbvOffsets {
            reason_codes: vec![
                "zeta".into(),
                "alpha".into(),
                "alpha".into(),
                "theta".into(),
                "beta".into(),
                "gamma".into(),
                "delta".into(),
                "epsilon".into(),
                "eta".into(),
                "iota".into(),
            ],
            ..hbv_offsets()
        });
        inputs.integrity = Some(IntegrityState::Fail);

        let baseline = resolve_baseline(&inputs);

        assert_eq!(
            baseline.reason_codes.codes.len(),
            ReasonSet::DEFAULT_MAX_LEN
        );
        assert_eq!(
            baseline.reason_codes.codes,
            vec![
                "alpha".to_string(),
                "beta".to_string(),
                "delta".to_string(),
                "epsilon".to_string(),
                "eta".to_string(),
                "gamma".to_string(),
                "integrity_fail".to_string(),
                "iota".to_string(),
            ]
        );
    }
}
