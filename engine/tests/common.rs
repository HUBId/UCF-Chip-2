#![cfg(test)]

use blake3::hash;
use chip4::pvgs::{Cbv, Digest32, InMemoryPvgs};
use prost::Message;
use std::sync::{Arc, Mutex};
use ucf::v1::{
    CharacterBaselineVector, ConsistencyClass, LevelClass, MacroMilestone, MacroMilestoneState,
    TraitUpdate, TraitUpdateDirection,
};

fn magnitude_to_offset(level: LevelClass) -> i32 {
    match level {
        LevelClass::Low => 1,
        LevelClass::Med => 2,
        LevelClass::High => 3,
        _ => 0,
    }
}

fn apply_trait_update(cbv: &mut CharacterBaselineVector, update: &TraitUpdate) {
    let Ok(level) = LevelClass::try_from(update.magnitude) else {
        return;
    };

    if level == LevelClass::Unknown {
        return;
    }

    let magnitude = magnitude_to_offset(level);

    if update.direction == TraitUpdateDirection::IncreaseStrictness as i32 {
        match update.trait_name.as_str() {
            "approval_strictness" => cbv.baseline_approval_strictness_offset += magnitude,
            "novelty_dampening" => cbv.baseline_novelty_dampening_offset += magnitude,
            "export_strictness" => cbv.baseline_export_strictness_offset += magnitude,
            "chain_conservatism" => cbv.baseline_chain_conservatism_offset += magnitude,
            "caution" => cbv.baseline_caution_offset += magnitude,
            "cooldown_multiplier" => cbv.baseline_cooldown_multiplier_class += magnitude,
            _ => {}
        }
    }
}

fn cbv_from_macro_milestone(milestone: &MacroMilestone) -> CharacterBaselineVector {
    let mut cbv = CharacterBaselineVector {
        baseline_caution_offset: 0,
        baseline_novelty_dampening_offset: 0,
        baseline_approval_strictness_offset: 0,
        baseline_export_strictness_offset: 0,
        baseline_chain_conservatism_offset: 0,
        baseline_cooldown_multiplier_class: 0,
    };

    for update in milestone.trait_updates.iter() {
        apply_trait_update(&mut cbv, update);
    }

    cbv
}

fn digest_for_cbv(cbv: &CharacterBaselineVector) -> [u8; 32] {
    let mut encoded = Vec::new();
    cbv.encode(&mut encoded).expect("encoding cbv");
    *hash(&encoded).as_bytes()
}

#[derive(Clone)]
pub struct PvgsHandle {
    store: InMemoryPvgs,
    next_epoch: Arc<Mutex<u64>>,
}

impl PvgsHandle {
    pub fn store(&self) -> InMemoryPvgs {
        self.store.clone()
    }

    pub fn append_macro_milestone(&self, milestone: MacroMilestone) {
        assert_eq!(milestone.state, MacroMilestoneState::Finalized as i32);
        assert_eq!(
            milestone.consistency_class,
            ConsistencyClass::ConsistencyHigh as i32
        );

        let cbv = cbv_from_macro_milestone(&milestone);
        let digest = digest_for_cbv(&cbv);

        let epoch = {
            let mut guard = self.next_epoch.lock().expect("locking epoch counter");
            *guard = guard.saturating_add(1);
            *guard
        };

        let cbv_update = Cbv {
            epoch,
            cbv_digest: Some(Digest32::from_array(digest)),
            proof_receipt_ref: Some(vec![0x01]),
            signature: Some(vec![0x02]),
            cbv: Some(cbv),
        };

        self.store.commit_cbv_update(cbv_update);
    }
}

pub fn setup_pvgs_with_keys_and_ruleset() -> PvgsHandle {
    PvgsHandle {
        store: InMemoryPvgs::new(),
        next_epoch: Arc::new(Mutex::new(0)),
    }
}
