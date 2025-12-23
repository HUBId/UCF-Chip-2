#![forbid(unsafe_code)]

use baseline_resolver::{resolve_baseline, BaselineInputs, HbvOffsets};
use dbm_0_sn::{SnInput, SubstantiaNigra};
use dbm_12_insula::{Insula, InsulaInput};
use dbm_13_hypothalamus::{ControlDecision, Hypothalamus, HypothalamusInput};
use dbm_18_cerebellum::{CerInput, Cerebellum};
use dbm_6_dopamin_nacc::{DopaInput, DopaOutput, DopaminNacc};
use dbm_7_lc::{Lc, LcInput};
use dbm_8_serotonin::{SerInput, Serotonin};
use dbm_9_amygdala::{AmyInput, Amygdala};
use dbm_core::{
    BaselineVector, CooldownClass, DbmModule, DwmMode, EmotionField, IsvSnapshot, LevelClass,
    OrientTarget, ProfileState, ReasonSet,
};
use dbm_hpa::{Hpa, HpaInput, HpaOutput};
use dbm_pag::{DefensePattern, Pag, PagInput};
use dbm_pmrf::{Pmrf, PmrfInput};
use dbm_pprf::{Pprf, PprfInput};
use dbm_sc::{Sc, ScInput};
use dbm_stn::{Stn, StnInput};
use emotion_field::{EmotionFieldInput, EmotionFieldModule};
use ucf::v1::{CharacterBaselineVector, PolicyEcologyVector, WindowKind};

#[derive(Debug, Clone, Default)]
pub struct BrainInput {
    pub now_ms: u64,
    pub window_kind: ucf::v1::WindowKind,
    pub hpa: HpaInput,
    pub cbv: Option<CharacterBaselineVector>,
    pub pev: Option<PolicyEcologyVector>,
    pub lc: LcInput,
    pub serotonin: SerInput,
    pub amygdala: AmyInput,
    pub pag: PagInput,
    pub cerebellum: Option<CerInput>,
    pub stn: StnInput,
    pub pmrf: PmrfInput,
    pub dopamin: Option<DopaInput>,
    pub insula: InsulaInput,
    pub sc_unlock_present: bool,
    pub sc_replay_planned_present: bool,
    pub pprf_cooldown_class: CooldownClass,
}

#[derive(Debug, Clone, Default)]
pub struct BrainOutput {
    pub decision: ControlDecision,
    pub emotion_field: EmotionField,
    pub dwm: DwmMode,
    pub focus_target: OrientTarget,
    pub baseline: BaselineVector,
    pub hpa: HpaOutput,
    pub cerebellum: Option<dbm_18_cerebellum::CerOutput>,
    pub dopamin: dbm_6_dopamin_nacc::DopaOutput,
    pub isv: IsvSnapshot,
    pub reason_codes: Vec<String>,
}

#[derive(Debug, Default)]
pub struct BrainBus {
    hpa: Hpa,
    lc: Lc,
    serotonin: Serotonin,
    amygdala: Amygdala,
    pag: Pag,
    cerebellum: Cerebellum,
    stn: Stn,
    pmrf: Pmrf,
    dopamin: DopaminNacc,
    insula: Insula,
    sn: SubstantiaNigra,
    sc: Sc,
    pprf: Pprf,
    hypothalamus: Hypothalamus,
    emotion_field: EmotionFieldModule,
    current_dwm: DwmMode,
    current_target: OrientTarget,
    last_window_kind: Option<WindowKind>,
    last_cerebellum_output: Option<dbm_18_cerebellum::CerOutput>,
    last_dopa_output: Option<DopaOutput>,
    last_hpa_output: HpaOutput,
    last_baseline_vector: BaselineVector,
    last_emotion_field: Option<EmotionField>,
}

impl BrainBus {
    pub fn new() -> Self {
        Self {
            current_dwm: DwmMode::ExecPlan,
            current_target: OrientTarget::Approval,
            ..Self::default()
        }
    }

    pub fn with_hpa(hpa: Hpa, last_hpa_output: HpaOutput) -> Self {
        Self {
            hpa,
            last_hpa_output,
            ..Self::new()
        }
    }

    pub fn lc_mut(&mut self) -> &mut Lc {
        &mut self.lc
    }

    pub fn serotonin_mut(&mut self) -> &mut Serotonin {
        &mut self.serotonin
    }

    pub fn amygdala_mut(&mut self) -> &mut Amygdala {
        &mut self.amygdala
    }

    pub fn pag_mut(&mut self) -> &mut Pag {
        &mut self.pag
    }

    pub fn stn_mut(&mut self) -> &mut Stn {
        &mut self.stn
    }

    pub fn pmrf_mut(&mut self) -> &mut Pmrf {
        &mut self.pmrf
    }

    pub fn sn_mut(&mut self) -> &mut SubstantiaNigra {
        &mut self.sn
    }

    pub fn sc_mut(&mut self) -> &mut Sc {
        &mut self.sc
    }

    pub fn pprf(&self) -> &Pprf {
        &self.pprf
    }

    pub fn pprf_mut(&mut self) -> &mut Pprf {
        &mut self.pprf
    }

    pub fn cerebellum_mut(&mut self) -> &mut Cerebellum {
        &mut self.cerebellum
    }

    pub fn dopamin_mut(&mut self) -> &mut DopaminNacc {
        &mut self.dopamin
    }

    pub fn hpa_mut(&mut self) -> &mut Hpa {
        &mut self.hpa
    }

    pub fn insula_mut(&mut self) -> &mut Insula {
        &mut self.insula
    }

    pub fn hypothalamus_mut(&mut self) -> &mut Hypothalamus {
        &mut self.hypothalamus
    }

    pub fn emotion_field_mut(&mut self) -> &mut EmotionFieldModule {
        &mut self.emotion_field
    }

    pub fn current_dwm(&self) -> DwmMode {
        self.current_dwm
    }

    pub fn set_current_dwm(&mut self, dwm: DwmMode) {
        self.current_dwm = dwm;
    }

    pub fn current_target(&self) -> OrientTarget {
        self.current_target
    }

    pub fn set_current_target(&mut self, target: OrientTarget) {
        self.current_target = target;
    }

    pub fn last_cerebellum_output(&self) -> Option<dbm_18_cerebellum::CerOutput> {
        self.last_cerebellum_output.clone()
    }

    pub fn set_last_cerebellum_output(&mut self, output: Option<dbm_18_cerebellum::CerOutput>) {
        self.last_cerebellum_output = output;
    }

    pub fn last_dopa_output(&self) -> Option<DopaOutput> {
        self.last_dopa_output.clone()
    }

    pub fn set_last_dopa_output(&mut self, output: Option<DopaOutput>) {
        self.last_dopa_output = output;
    }

    pub fn last_hpa_output(&self) -> HpaOutput {
        self.last_hpa_output.clone()
    }

    pub fn set_last_hpa_output(&mut self, output: HpaOutput) {
        self.last_hpa_output = output;
    }

    pub fn last_baseline_vector(&self) -> BaselineVector {
        self.last_baseline_vector.clone()
    }

    pub fn set_last_baseline_vector(&mut self, vector: BaselineVector) {
        self.last_baseline_vector = vector;
    }

    pub fn last_emotion_field(&self) -> Option<EmotionField> {
        self.last_emotion_field.clone()
    }

    pub fn set_last_emotion_field(&mut self, field: Option<EmotionField>) {
        self.last_emotion_field = field;
    }

    pub fn tick(&mut self, input: BrainInput) -> BrainOutput {
        let medium_window = input.window_kind == WindowKind::Medium;
        if medium_window && self.last_window_kind != Some(WindowKind::Medium) {
            self.pprf.on_medium_window_rollover();
        }
        self.last_window_kind = Some(input.window_kind);

        let hpa_output = if medium_window {
            let output = self.hpa.tick(&input.hpa);
            self.set_last_hpa_output(output.clone());
            output
        } else {
            self.last_hpa_output()
        };

        let baseline = self.resolve_baseline(&input, &hpa_output);
        self.set_last_baseline_vector(baseline.clone());

        let cerebellum_output = input
            .cerebellum
            .as_ref()
            .map(|cer_input| self.cerebellum.tick(cer_input));
        self.set_last_cerebellum_output(cerebellum_output.clone());

        let lc_input = LcInput {
            arousal_floor: level_max(
                input.lc.arousal_floor,
                level_max(
                    baseline.caution_floor,
                    cerebellum_output
                        .as_ref()
                        .map_or(LevelClass::Low, |c| c.divergence),
                ),
            ),
            ..input.lc
        };
        let lc_output = self.lc.tick(&lc_input);

        let ser_input = SerInput {
            stability_floor: level_max(input.serotonin.stability_floor, baseline.caution_floor),
            ..input.serotonin
        };
        let mut ser_output = self.serotonin.tick(&ser_input);
        ser_output = apply_baseline_cooldown_bias(ser_output, &baseline);

        let amy_output = self.amygdala.tick(&input.amygdala);

        let pag_input = PagInput {
            threat: amy_output.threat,
            vectors: amy_output.vectors.clone(),
            stability: ser_output.stability,
            ..input.pag
        };
        let pag_output = self.pag.tick(&pag_input);

        let stn_input = StnInput {
            threat: amy_output.threat,
            arousal: lc_output.arousal,
            ..input.stn
        };
        let stn_output = self.stn.tick(&stn_input);

        let cerebellum_divergence = cerebellum_output
            .as_ref()
            .map_or(LevelClass::Low, |output| output.divergence);

        let pmrf_input = PmrfInput {
            hold_active: stn_output.hold_active,
            stability: ser_output.stability,
            divergence: level_max(input.pmrf.divergence, cerebellum_divergence),
            ..input.pmrf
        };
        let pmrf_output = self.pmrf.tick(&pmrf_input);

        let dopamin_input = input.dopamin.map(|mut dopa_input| {
            dopa_input.threat = amy_output.threat;
            dopa_input
        });

        let mut dopa_output = dopamin_input
            .as_ref()
            .map(|dopa_input| self.dopamin.tick(dopa_input))
            .or_else(|| self.last_dopa_output())
            .unwrap_or_default();
        if level_at_least(baseline.reward_block_bias, LevelClass::Med) {
            dopa_output.reward_block = true;
            dopa_output
                .reason_codes
                .insert("baseline_reward_block".to_string());
        }
        if dopamin_input.is_some() {
            self.set_last_dopa_output(Some(dopa_output.clone()));
        }

        let mut insula_input = input.insula;
        insula_input.hbv_present = true;
        insula_input
            .dominant_reason_codes
            .extend(amy_output.reason_codes.codes.clone());
        insula_input
            .dominant_reason_codes
            .extend(pag_output.reason_codes.codes.clone());
        insula_input.progress = dopa_output.progress;
        let mut isv = self.insula.tick(&insula_input);
        isv.threat = level_max(isv.threat, amy_output.threat);
        isv.threat_vectors = Some(amy_output.vectors.clone());
        isv.dominant_reason_codes
            .extend(amy_output.reason_codes.codes.clone());
        isv.dominant_reason_codes
            .extend(pag_output.reason_codes.codes.clone());
        isv.arousal = level_max(isv.arousal, lc_output.arousal);
        isv.stability = level_max(isv.stability, ser_output.stability);
        isv.replay_hint = dopa_output.replay_hint;
        isv.dominant_reason_codes
            .extend(dopa_output.reason_codes.codes.clone());

        let cooldown_level = Some(cooldown_to_level(ser_output.cooldown_class));
        let sn_output = self.sn.tick(&SnInput {
            isv: isv.clone(),
            cooldown_class: cooldown_level,
            current_dwm: Some(self.current_dwm),
            replay_hint: dopa_output.replay_hint,
            reward_block: dopa_output.reward_block,
        });

        let sc_output = self.sc.tick(&ScInput {
            isv: isv.clone(),
            salience_items: sn_output.salience_items.clone(),
            unlock_present: input.sc_unlock_present,
            replay_planned_present: input.sc_replay_planned_present,
            integrity: isv.integrity,
        });

        let isv_snapshot = isv.clone();

        let pprf_output = self.pprf.tick(&PprfInput {
            orient: sc_output.clone(),
            current_target: self.current_target,
            current_dwm: self.current_dwm,
            cooldown_class: ser_output.cooldown_class,
            stability: ser_output.stability,
            arousal: lc_output.arousal,
            now_ms: input.now_ms,
        });
        self.current_target = pprf_output.active_target;
        self.current_dwm = pprf_output.active_dwm;

        let mut decision = self.hypothalamus.tick(&HypothalamusInput {
            isv: isv.clone(),
            export_lock_bias: baseline.export_strictness == LevelClass::High,
            simulate_first_bias: baseline.chain_conservatism == LevelClass::High,
            approval_strict: baseline.approval_strictness == LevelClass::High,
            novelty_lock_bias: baseline.novelty_dampening == LevelClass::High,
        });
        merge_secondary_outputs(&mut decision, &lc_output, &ser_output, &sn_output);
        decision
            .reason_codes
            .extend(baseline.reason_codes.codes.clone());
        decision
            .reason_codes
            .extend(pprf_output.reason_codes.codes.clone());
        apply_defense_pattern(&mut decision, &pag_output, input.stn.policy_pressure);
        decision
            .reason_codes
            .extend(dopa_output.reason_codes.codes.clone());
        decision
            .reason_codes
            .extend(stn_output.hold_reason_codes.codes.clone());
        decision
            .reason_codes
            .extend(pmrf_output.reason_codes.codes.clone());
        if let Some(cer_output) = &cerebellum_output {
            decision
                .reason_codes
                .extend(cer_output.reason_codes.codes.clone());
        }

        decision.overlays.simulate_first |= stn_output.hint_simulate_first;
        decision.overlays.novelty_lock |= stn_output.hint_novelty_lock;
        decision.overlays.export_lock |= stn_output.hint_export_lock;
        if pmrf_output.checkpoint_required {
            decision.overlays.simulate_first = true;
        }

        let mut emotion_field = self.emotion_field.tick(&EmotionFieldInput {
            isv: isv.clone(),
            dwm: pprf_output.active_dwm,
            profile: decision.profile_state,
            overlays: decision.overlays.clone(),
            progress: dopa_output.progress,
            reward_block: dopa_output.reward_block,
            defense_pattern: Some(pag_output.pattern),
            replay_hint: dopa_output.replay_hint,
        });
        emotion_field
            .reason_codes
            .extend(decision.reason_codes.codes.clone());
        self.set_last_emotion_field(Some(emotion_field.clone()));

        let reason_codes = merge_reason_codes([
            decision.reason_codes.codes.clone(),
            emotion_field.reason_codes.codes.clone(),
            pprf_output.reason_codes.codes.clone(),
            sc_output.reason_codes.codes.clone(),
            sn_output.reason_codes.codes.clone(),
        ]);

        BrainOutput {
            decision,
            emotion_field,
            dwm: pprf_output.active_dwm,
            focus_target: pprf_output.active_target,
            baseline,
            hpa: hpa_output,
            cerebellum: cerebellum_output,
            dopamin: dopa_output,
            isv: isv_snapshot,
            reason_codes,
        }
    }

    fn resolve_baseline(&self, input: &BrainInput, hpa: &HpaOutput) -> BaselineVector {
        let hbv = HbvOffsets {
            baseline_caution_offset: hpa.baseline_caution_offset as i32,
            baseline_novelty_dampening_offset: hpa.baseline_novelty_dampening_offset as i32,
            baseline_approval_strictness_offset: hpa.baseline_approval_strictness_offset as i32,
            baseline_export_strictness_offset: hpa.baseline_export_strictness_offset as i32,
            baseline_chain_conservatism_offset: hpa.baseline_chain_conservatism_offset as i32,
            baseline_cooldown_multiplier_class: hpa.baseline_cooldown_multiplier_class as i32,
            reward_block_bias: None,
            reason_codes: hpa.reason_codes.codes.clone(),
        };

        resolve_baseline(&BaselineInputs {
            cbv: input.cbv.clone(),
            pev: input.pev.clone(),
            hbv: Some(hbv),
            integrity: Some(input.lc.integrity),
        })
    }
}

fn apply_baseline_cooldown_bias(
    mut ser_output: dbm_8_serotonin::SerOutput,
    baseline: &BaselineVector,
) -> dbm_8_serotonin::SerOutput {
    if matches!(baseline.cooldown_bias, CooldownClass::Longer) {
        ser_output.cooldown_class = CooldownClass::Longer;
        ser_output
            .reason_codes
            .insert("baseline_cooldown_bias".to_string());
    }

    ser_output
}

fn merge_secondary_outputs(
    decision: &mut ControlDecision,
    lc_output: &dbm_7_lc::LcOutput,
    ser_output: &dbm_8_serotonin::SerOutput,
    sn_output: &dbm_0_sn::SnOutput,
) {
    decision.overlays.simulate_first |= lc_output.hint_simulate_first;
    decision.overlays.novelty_lock |= lc_output.hint_novelty_lock;
    decision.deescalation_lock |= ser_output.deescalation_lock;
    decision.cooldown_class = level_max(
        decision.cooldown_class,
        cooldown_to_level(ser_output.cooldown_class),
    );

    decision
        .reason_codes
        .extend(lc_output.reason_codes.codes.clone());
    decision
        .reason_codes
        .extend(ser_output.reason_codes.codes.clone());
    decision
        .reason_codes
        .extend(sn_output.reason_codes.codes.clone());
}

fn apply_defense_pattern(
    decision: &mut ControlDecision,
    pag_output: &dbm_pag::PagOutput,
    policy_pressure: LevelClass,
) {
    decision
        .reason_codes
        .extend(pag_output.reason_codes.codes.clone());

    match pag_output.pattern {
        DefensePattern::DP3_FORENSIC => {
            decision.profile_state = profile_max(decision.profile_state, ProfileState::M3);
            decision.overlays.simulate_first = true;
            decision.overlays.export_lock = true;
            decision.overlays.novelty_lock = true;
            decision.deescalation_lock = true;
        }
        DefensePattern::DP2_QUARANTINE => {
            decision.profile_state = profile_max(decision.profile_state, ProfileState::M2);
            decision.overlays.simulate_first = true;
            decision.overlays.export_lock = true;
            decision.overlays.novelty_lock = true;
            decision.deescalation_lock = true;
        }
        DefensePattern::DP1_FREEZE | DefensePattern::DP4_CONTAINED_CONTINUE => {
            if policy_pressure == LevelClass::High {
                decision.overlays.simulate_first = true;
            }
        }
    }
}

fn profile_max(a: ProfileState, b: ProfileState) -> ProfileState {
    match (a, b) {
        (ProfileState::M3, _) | (_, ProfileState::M3) => ProfileState::M3,
        (ProfileState::M2, _) | (_, ProfileState::M2) => ProfileState::M2,
        (ProfileState::M1, _) | (_, ProfileState::M1) => ProfileState::M1,
        _ => ProfileState::M0,
    }
}

fn cooldown_to_level(class: CooldownClass) -> LevelClass {
    match class {
        CooldownClass::Base => LevelClass::Low,
        CooldownClass::Longer => LevelClass::High,
    }
}

fn level_max(a: LevelClass, b: LevelClass) -> LevelClass {
    match (a, b) {
        (LevelClass::High, _) | (_, LevelClass::High) => LevelClass::High,
        (LevelClass::Med, _) | (_, LevelClass::Med) => LevelClass::Med,
        _ => LevelClass::Low,
    }
}

fn level_at_least(level: LevelClass, threshold: LevelClass) -> bool {
    level_severity(level) >= level_severity(threshold)
}

fn level_severity(level: LevelClass) -> u8 {
    match level {
        LevelClass::Low => 0,
        LevelClass::Med => 1,
        LevelClass::High => 2,
    }
}

fn merge_reason_codes<const N: usize>(lists: [Vec<String>; N]) -> Vec<String> {
    let mut merged: Vec<String> = lists.into_iter().flatten().collect();
    merged.sort();
    merged.dedup();
    merged.truncate(ReasonSet::DEFAULT_MAX_LEN);
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reason_codes_are_deterministic() {
        let codes = merge_reason_codes([
            vec!["b".to_string(), "a".to_string()],
            vec!["a".to_string(), "c".to_string()],
        ]);
        assert_eq!(
            codes,
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
    }
}
