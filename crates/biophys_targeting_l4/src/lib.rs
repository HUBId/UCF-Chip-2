#![forbid(unsafe_code)]

use biophys_core::{CompartmentId, NeuronId};
use biophys_morphology::{dendrite_compartments, soma_compartment, NeuronMorphology};
use biophys_synapses_l4::SynKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetRule {
    SomaOnly,
    ProximalDendrite,
    DistalDendrite,
    RandomDeterministic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetingPolicy {
    pub ampa_rule: TargetRule,
    pub nmda_rule: TargetRule,
    pub gaba_rule: TargetRule,
    pub seed_digest: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdgeKey {
    pub pre_neuron_id: NeuronId,
    pub post_neuron_id: NeuronId,
    pub synapse_index: u32,
}

pub fn select_post_compartment(
    neuron: &NeuronMorphology,
    kind: SynKind,
    policy: &TargetingPolicy,
    edge_key: EdgeKey,
) -> CompartmentId {
    let rule = match kind {
        SynKind::AMPA => policy.ampa_rule,
        SynKind::NMDA => policy.nmda_rule,
        SynKind::GABA => policy.gaba_rule,
    };

    let soma = soma_compartment(neuron);
    let dendrites = dendrite_compartments(neuron);
    if dendrites.is_empty() {
        return soma;
    }

    match rule {
        TargetRule::SomaOnly => soma,
        TargetRule::ProximalDendrite => pick_by_depth(neuron, &dendrites, true).unwrap_or(soma),
        TargetRule::DistalDendrite => pick_by_depth(neuron, &dendrites, false).unwrap_or(soma),
        TargetRule::RandomDeterministic => pick_random_dendrite(&dendrites, policy, edge_key),
    }
}

fn pick_by_depth(
    neuron: &NeuronMorphology,
    dendrites: &[CompartmentId],
    ascending: bool,
) -> Option<CompartmentId> {
    let mut candidates = dendrites
        .iter()
        .filter_map(|id| {
            neuron
                .compartments
                .iter()
                .find(|compartment| compartment.id == *id)
                .map(|compartment| (compartment.depth, compartment.id))
        })
        .collect::<Vec<_>>();

    candidates.sort_by_key(|(depth, id)| (*depth, id.0));
    if !ascending {
        candidates.reverse();
    }
    candidates.first().map(|(_, id)| *id)
}

fn pick_random_dendrite(
    dendrites: &[CompartmentId],
    policy: &TargetingPolicy,
    edge_key: EdgeKey,
) -> CompartmentId {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&edge_key.pre_neuron_id.0.to_le_bytes());
    hasher.update(&edge_key.post_neuron_id.0.to_le_bytes());
    hasher.update(&edge_key.synapse_index.to_le_bytes());
    hasher.update(&policy.seed_digest);
    let digest = hasher.finalize();
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest.as_bytes()[0..8]);
    let value = u64::from_le_bytes(bytes);
    let index = (value % dendrites.len() as u64) as usize;
    dendrites[index]
}

#[cfg(all(test, feature = "biophys-l4-targeting"))]
mod tests {
    use super::*;
    use biophys_core::CompartmentId;
    use biophys_morphology::{
        compute_depths, morphology_tree, Compartment, CompartmentKind, NeuronMorphology,
    };

    fn test_morphology() -> NeuronMorphology {
        NeuronMorphology {
            neuron_id: NeuronId(1),
            compartments: vec![
                Compartment {
                    id: CompartmentId(0),
                    parent: None,
                    kind: CompartmentKind::Soma,
                    depth: 0,
                    capacitance: 1.0,
                    axial_resistance: 150.0,
                },
                Compartment {
                    id: CompartmentId(1),
                    parent: Some(CompartmentId(0)),
                    kind: CompartmentKind::Dendrite,
                    depth: 1,
                    capacitance: 1.0,
                    axial_resistance: 200.0,
                },
                Compartment {
                    id: CompartmentId(2),
                    parent: Some(CompartmentId(1)),
                    kind: CompartmentKind::Dendrite,
                    depth: 2,
                    capacitance: 1.0,
                    axial_resistance: 200.0,
                },
            ],
        }
    }

    fn default_policy() -> TargetingPolicy {
        TargetingPolicy {
            ampa_rule: TargetRule::ProximalDendrite,
            nmda_rule: TargetRule::DistalDendrite,
            gaba_rule: TargetRule::SomaOnly,
            seed_digest: *blake3::hash(b"UCF:L4:TARGETING").as_bytes(),
        }
    }

    #[test]
    fn deterministic_selection_is_stable() {
        let morphology = test_morphology();
        let policy = default_policy();
        let edge_key = EdgeKey {
            pre_neuron_id: NeuronId(10),
            post_neuron_id: NeuronId(11),
            synapse_index: 3,
        };
        let first = select_post_compartment(&morphology, SynKind::AMPA, &policy, edge_key);
        let second = select_post_compartment(&morphology, SynKind::AMPA, &policy, edge_key);
        assert_eq!(first, second);
    }

    #[test]
    fn proximal_and_distal_choose_expected_depth() {
        let morphology = test_morphology();
        let policy = default_policy();
        let edge_key = EdgeKey {
            pre_neuron_id: NeuronId(1),
            post_neuron_id: NeuronId(2),
            synapse_index: 0,
        };
        let proximal = select_post_compartment(&morphology, SynKind::AMPA, &policy, edge_key);
        let distal = select_post_compartment(&morphology, SynKind::NMDA, &policy, edge_key);
        assert_eq!(proximal, CompartmentId(1));
        assert_eq!(distal, CompartmentId(2));
    }

    #[test]
    fn random_deterministic_is_stable_across_edges() {
        let morphology = test_morphology();
        let policy = TargetingPolicy {
            ampa_rule: TargetRule::RandomDeterministic,
            nmda_rule: TargetRule::RandomDeterministic,
            gaba_rule: TargetRule::RandomDeterministic,
            seed_digest: *blake3::hash(b"UCF:L4:RANDOM").as_bytes(),
        };
        let mut selections = Vec::new();
        for synapse_index in 0..6 {
            let edge_key = EdgeKey {
                pre_neuron_id: NeuronId(1),
                post_neuron_id: NeuronId(2),
                synapse_index,
            };
            selections.push(select_post_compartment(
                &morphology,
                SynKind::AMPA,
                &policy,
                edge_key,
            ));
        }
        let expected = selections.clone();
        let mut repeat = Vec::new();
        for synapse_index in 0..6 {
            let edge_key = EdgeKey {
                pre_neuron_id: NeuronId(1),
                post_neuron_id: NeuronId(2),
                synapse_index,
            };
            repeat.push(select_post_compartment(
                &morphology,
                SynKind::AMPA,
                &policy,
                edge_key,
            ));
        }
        assert_eq!(expected, repeat);
    }

    #[test]
    fn integration_smoke_for_rules() {
        let morphology = test_morphology();
        let policy = default_policy();
        let edge_key = EdgeKey {
            pre_neuron_id: NeuronId(1),
            post_neuron_id: NeuronId(2),
            synapse_index: 1,
        };
        let ampa_compartment =
            select_post_compartment(&morphology, SynKind::AMPA, &policy, edge_key);
        let nmda_compartment =
            select_post_compartment(&morphology, SynKind::NMDA, &policy, edge_key);
        let gaba_compartment =
            select_post_compartment(&morphology, SynKind::GABA, &policy, edge_key);
        assert_eq!(ampa_compartment, CompartmentId(1));
        assert_eq!(nmda_compartment, CompartmentId(2));
        assert_eq!(gaba_compartment, CompartmentId(0));
    }

    #[test]
    #[cfg(feature = "biophys-l4-morphology-multi")]
    fn multi_compartment_targeting_selects_expected_depths() {
        let morphology = morphology_tree(NeuronId(1), 15);
        let policy = default_policy();
        let edge_key = EdgeKey {
            pre_neuron_id: NeuronId(3),
            post_neuron_id: NeuronId(4),
            synapse_index: 2,
        };
        let proximal = select_post_compartment(&morphology, SynKind::AMPA, &policy, edge_key);
        let distal = select_post_compartment(&morphology, SynKind::NMDA, &policy, edge_key);

        let depths = compute_depths(&morphology);
        let prox_depth = depths[proximal.0 as usize];
        let dist_depth = depths[distal.0 as usize];
        let max_depth = *depths.iter().max().unwrap_or(&0);

        assert_eq!(prox_depth, 1);
        assert_eq!(dist_depth, max_depth);
    }

    #[test]
    #[cfg(feature = "biophys-l4-morphology-multi")]
    fn multi_compartment_random_deterministic_selects_dendrites() {
        let morphology = morphology_tree(NeuronId(1), 7);
        let policy = TargetingPolicy {
            ampa_rule: TargetRule::RandomDeterministic,
            nmda_rule: TargetRule::RandomDeterministic,
            gaba_rule: TargetRule::RandomDeterministic,
            seed_digest: *blake3::hash(b"UCF:L4:RANDOM:MULTI").as_bytes(),
        };
        let edge_key = EdgeKey {
            pre_neuron_id: NeuronId(9),
            post_neuron_id: NeuronId(10),
            synapse_index: 5,
        };
        let selected = select_post_compartment(&morphology, SynKind::AMPA, &policy, edge_key);
        let dendrites = dendrite_compartments(&morphology);
        assert!(dendrites.contains(&selected));
    }
}
