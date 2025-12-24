#![forbid(unsafe_code)]

use biophys_core::{CompartmentId, NeuronId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompartmentKind {
    Soma,
    Dendrite,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Compartment {
    pub id: CompartmentId,
    pub parent: Option<CompartmentId>,
    pub kind: CompartmentKind,
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
