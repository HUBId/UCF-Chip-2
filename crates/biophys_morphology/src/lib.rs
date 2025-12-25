#![forbid(unsafe_code)]

use biophys_core::{CompartmentId, NeuronId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompartmentKind {
    Soma,
    Dendrite,
    Axon,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Compartment {
    pub id: CompartmentId,
    pub parent: Option<CompartmentId>,
    pub kind: CompartmentKind,
    pub depth: u32,
    pub capacitance: f32,
    pub axial_resistance: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NeuronMorphology {
    pub neuron_id: NeuronId,
    pub compartments: Vec<Compartment>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MorphologyError {
    TooManyCompartments { count: usize, max: usize },
}

pub const MAX_COMPARTMENTS: usize = 64;

impl NeuronMorphology {
    pub fn validate(&self, max_compartments: usize) -> Result<(), MorphologyError> {
        let count = self.compartments.len();
        if count > max_compartments {
            return Err(MorphologyError::TooManyCompartments {
                count,
                max: max_compartments,
            });
        }
        Ok(())
    }
}

pub fn dendrite_compartments(neuron: &NeuronMorphology) -> Vec<CompartmentId> {
    let mut ids = neuron
        .compartments
        .iter()
        .filter(|compartment| compartment.kind == CompartmentKind::Dendrite)
        .map(|compartment| compartment.id)
        .collect::<Vec<_>>();
    ids.sort_by_key(|id| id.0);
    ids
}

pub fn soma_compartment(neuron: &NeuronMorphology) -> CompartmentId {
    neuron
        .compartments
        .iter()
        .find(|compartment| compartment.kind == CompartmentKind::Soma)
        .map(|compartment| compartment.id)
        .unwrap_or_else(|| {
            neuron
                .compartments
                .first()
                .map(|compartment| compartment.id)
                .expect("neuron morphology must have at least one compartment")
        })
}

pub fn morphology_tree(neuron_id: NeuronId, compartments_per_neuron: u16) -> NeuronMorphology {
    let plan = tree_plan(compartments_per_neuron);
    let mut compartments = Vec::with_capacity(plan.total as usize);
    let mut next_id = 0u32;

    compartments.push(Compartment {
        id: CompartmentId(next_id),
        parent: None,
        kind: CompartmentKind::Soma,
        depth: 0,
        capacitance: capacitance_for_depth(0),
        axial_resistance: axial_resistance_for_depth(0),
    });
    next_id += 1;

    let mut current_level = vec![CompartmentId(0)];
    for depth in 1..=plan.depths {
        let mut next_level = Vec::new();
        let count = plan.children_per_depth[(depth - 1) as usize];
        let mut parent_index = 0usize;
        for _ in 0..count {
            let parent_id = current_level[parent_index];
            if !current_level.is_empty() {
                parent_index = (parent_index + 1) % current_level.len();
            }
            let comp_id = CompartmentId(next_id);
            next_id += 1;
            compartments.push(Compartment {
                id: comp_id,
                parent: Some(parent_id),
                kind: CompartmentKind::Dendrite,
                depth: depth as u32,
                capacitance: capacitance_for_depth(depth),
                axial_resistance: axial_resistance_for_depth(depth),
            });
            next_level.push(comp_id);
        }
        current_level = next_level;
    }

    let morphology = NeuronMorphology {
        neuron_id,
        compartments,
    };
    debug_assert!(morphology.compartments.len() <= MAX_COMPARTMENTS);
    debug_assert_eq!(morphology.compartments.len(), plan.total as usize);
    morphology
}

pub fn compute_depths(neuron: &NeuronMorphology) -> Vec<u16> {
    assert!(
        neuron.compartments.len() <= MAX_COMPARTMENTS,
        "morphology exceeds max compartments"
    );

    let mut id_to_index = std::collections::BTreeMap::new();
    for (index, compartment) in neuron.compartments.iter().enumerate() {
        id_to_index.insert(compartment.id, index);
    }

    let mut depths = vec![None; neuron.compartments.len()];
    for index in 0..neuron.compartments.len() {
        if depths[index].is_some() {
            continue;
        }

        let mut stack = Vec::new();
        let mut current = index;
        loop {
            if let Some(depth) = depths[current] {
                let mut d: u16 = depth;
                for &entry in stack.iter().rev() {
                    d = d.saturating_add(1);
                    depths[entry] = Some(d);
                }
                break;
            }

            stack.push(current);
            let parent = neuron.compartments[current].parent;
            match parent.and_then(|id| id_to_index.get(&id).copied()) {
                Some(parent_index) => {
                    current = parent_index;
                }
                None => {
                    let mut d: u16 = 0;
                    for &entry in stack.iter().rev() {
                        depths[entry] = Some(d);
                        d = d.saturating_add(1);
                    }
                    break;
                }
            }
        }
    }

    depths
        .into_iter()
        .map(|depth| depth.unwrap_or(0))
        .collect()
}

struct TreePlan {
    total: u32,
    depths: u16,
    children_per_depth: Vec<u16>,
}

fn tree_plan(compartments_per_neuron: u16) -> TreePlan {
    match compartments_per_neuron {
        7 => TreePlan {
            total: 7,
            depths: 2,
            children_per_depth: vec![2, 4],
        },
        15 => TreePlan {
            total: 15,
            depths: 3,
            children_per_depth: vec![3, 6, 5],
        },
        _ => panic!("unsupported compartment count: {compartments_per_neuron}"),
    }
}

fn capacitance_for_depth(depth: u16) -> f32 {
    let base = 1.0;
    let step = 0.15;
    let max = 2.5;
    (base + step * depth as f32).min(max)
}

fn axial_resistance_for_depth(depth: u16) -> f32 {
    let base = 150.0;
    let step = 25.0;
    let max = 300.0;
    (base + step * depth as f32).min(max)
}
