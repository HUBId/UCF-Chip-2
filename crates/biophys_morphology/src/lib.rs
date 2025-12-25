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
