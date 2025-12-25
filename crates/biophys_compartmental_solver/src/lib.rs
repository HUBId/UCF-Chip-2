#![forbid(unsafe_code)]

#[cfg(feature = "biophys-l4-ca")]
use biophys_channels::{ca_current, ca_p_inf_q, CaLike};
use biophys_channels::{leak_current, nak_current, GatingState, Leak, NaK};
use biophys_core::{CompartmentId, NeuronId};
use biophys_morphology::{
    Compartment, CompartmentKind, MorphologyError, NeuronMorphology, MAX_COMPARTMENTS,
};
#[cfg(feature = "biophys-l4-synapses")]
use biophys_synapses_l4::SynapseAccumulator;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompartmentChannels {
    pub leak: Leak,
    pub nak: Option<NaK>,
    #[cfg(feature = "biophys-l4-ca")]
    pub ca: Option<CaLike>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct L4State {
    pub voltages: Vec<f32>,
    pub gates: Vec<GatingState>,
    #[cfg(feature = "biophys-l4-ca")]
    pub p_ca_q: Vec<u16>,
}

impl L4State {
    pub fn new(initial_voltage: f32, compartment_count: usize) -> Self {
        let voltages = vec![initial_voltage; compartment_count];
        let gates = voltages
            .iter()
            .map(|&v| GatingState::from_voltage(v))
            .collect();
        #[cfg(feature = "biophys-l4-ca")]
        let p_ca_q = voltages.iter().map(|&v| ca_p_inf_q(v)).collect();
        Self {
            voltages,
            gates,
            #[cfg(feature = "biophys-l4-ca")]
            p_ca_q,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L4StepOutput {
    pub ca_spike: bool,
}

#[cfg(feature = "biophys-l4-ca")]
const TAU_CA_STEPS: i32 = 50;
#[cfg(feature = "biophys-l4-ca")]
const CA_SPIKE_THRESHOLD_MV: f32 = -10.0;

#[derive(Debug, Clone)]
pub struct L4Solver {
    morphology: NeuronMorphology,
    channels: Vec<CompartmentChannels>,
    children_indices: Vec<Vec<usize>>,
    axial_currents: Vec<f32>,
    dt_ms: f32,
    clamp_min: f32,
    clamp_max: f32,
    #[cfg(feature = "biophys-l4-ca")]
    max_depth: u32,
    step_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolverError {
    Morphology(MorphologyError),
    ChannelCountMismatch {
        expected: usize,
        got: usize,
    },
    DuplicateCompartmentId(CompartmentId),
    MissingParent {
        child: CompartmentId,
        parent: CompartmentId,
    },
}

impl L4Solver {
    pub fn new(
        morphology: NeuronMorphology,
        channels: Vec<CompartmentChannels>,
        dt_ms: f32,
        clamp_min: f32,
        clamp_max: f32,
    ) -> Result<Self, SolverError> {
        morphology
            .validate(MAX_COMPARTMENTS)
            .map_err(SolverError::Morphology)?;
        if morphology.compartments.len() != channels.len() {
            return Err(SolverError::ChannelCountMismatch {
                expected: morphology.compartments.len(),
                got: channels.len(),
            });
        }

        let mut paired: Vec<(CompartmentId, Compartment, CompartmentChannels)> = morphology
            .compartments
            .into_iter()
            .zip(channels)
            .map(|(compartment, channel)| (compartment.id, compartment, channel))
            .collect();

        paired.sort_by_key(|(id, _, _)| id.0);

        for window in paired.windows(2) {
            if let [left, right] = window {
                if left.0 == right.0 {
                    return Err(SolverError::DuplicateCompartmentId(left.0));
                }
            }
        }

        let mut compartments = Vec::with_capacity(paired.len());
        let mut channels = Vec::with_capacity(paired.len());
        for (_, compartment, channel) in paired {
            compartments.push(compartment);
            channels.push(channel);
        }

        let morphology = NeuronMorphology {
            neuron_id: morphology.neuron_id,
            compartments,
        };

        let mut id_to_index = std::collections::BTreeMap::new();
        for (index, compartment) in morphology.compartments.iter().enumerate() {
            id_to_index.insert(compartment.id, index);
        }

        let mut parent_indices = Vec::with_capacity(morphology.compartments.len());
        for compartment in &morphology.compartments {
            let parent_index = match compartment.parent {
                Some(parent_id) => Some(id_to_index.get(&parent_id).copied().ok_or(
                    SolverError::MissingParent {
                        child: compartment.id,
                        parent: parent_id,
                    },
                )?),
                None => None,
            };
            parent_indices.push(parent_index);
        }

        let mut children_indices = vec![Vec::new(); morphology.compartments.len()];
        for (index, parent_index) in parent_indices.iter().enumerate() {
            if let Some(parent_index) = parent_index {
                children_indices[*parent_index].push(index);
            }
        }

        let axial_currents = vec![0.0_f32; morphology.compartments.len()];
        #[cfg(feature = "biophys-l4-ca")]
        let max_depth = morphology
            .compartments
            .iter()
            .map(|compartment| compartment.depth)
            .max()
            .unwrap_or(0);

        Ok(Self {
            morphology,
            channels,
            children_indices,
            axial_currents,
            dt_ms,
            clamp_min,
            clamp_max,
            #[cfg(feature = "biophys-l4-ca")]
            max_depth,
            step_count: 0,
        })
    }

    pub fn neuron_id(&self) -> NeuronId {
        self.morphology.neuron_id
    }

    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    pub fn step(&mut self, state: &mut L4State, input_current: &[f32]) {
        self.step_with_output(state, input_current);
    }

    pub fn step_with_output(&mut self, state: &mut L4State, input_current: &[f32]) -> L4StepOutput {
        assert_eq!(
            state.voltages.len(),
            self.morphology.compartments.len(),
            "state voltage count must match compartments"
        );
        assert_eq!(
            input_current.len(),
            self.morphology.compartments.len(),
            "input current count must match compartments"
        );

        for (v, gates) in state.voltages.iter().copied().zip(state.gates.iter_mut()) {
            gates.update(v, self.dt_ms);
        }

        #[cfg(feature = "biophys-l4-ca")]
        let mut ca_spike = false;

        for current in self.axial_currents.iter_mut() {
            *current = 0.0;
        }
        for (parent_index, children) in self.children_indices.iter().enumerate() {
            for &child_index in children {
                let compartment = &self.morphology.compartments[child_index];
                let v_child = state.voltages[child_index];
                let v_parent = state.voltages[parent_index];
                let resistance = compartment.axial_resistance.max(1e-6);
                let current = (v_parent - v_child) / resistance;
                self.axial_currents[child_index] += current;
                self.axial_currents[parent_index] -= current;
            }
        }

        for (index, compartment) in self.morphology.compartments.iter().enumerate() {
            let channels = self.channels[index];
            let v = state.voltages[index];
            let gates = state.gates[index];
            let mut ionic = leak_current(channels.leak, v);
            if let Some(nak) = channels.nak {
                ionic += nak_current(nak, gates, v);
            }
            #[cfg(feature = "biophys-l4-ca")]
            if let Some(ca) = channels.ca {
                if is_distal(compartment.depth, self.max_depth) {
                    let p_inf_q = ca_p_inf_q(v);
                    let current_q = state.p_ca_q[index] as i32;
                    let updated_q =
                        (current_q + (p_inf_q as i32 - current_q) / TAU_CA_STEPS).clamp(0, 1000);
                    state.p_ca_q[index] = updated_q as u16;
                    ionic += ca_current(ca, state.p_ca_q[index], v);
                }
            }
            let axial = self.axial_currents[index];
            let ext = input_current[index];
            let capacitance = compartment.capacitance.max(1e-6);
            let dv = self.dt_ms * (ext + axial - ionic) / capacitance;
            let updated = (v + dv).clamp(self.clamp_min, self.clamp_max);
            #[cfg(feature = "biophys-l4-ca")]
            if is_distal(compartment.depth, self.max_depth)
                && v < CA_SPIKE_THRESHOLD_MV
                && updated >= CA_SPIKE_THRESHOLD_MV
            {
                ca_spike = true;
            }
            state.voltages[index] = updated;
        }

        self.step_count = self.step_count.saturating_add(1);
        #[cfg(feature = "biophys-l4-ca")]
        {
            return L4StepOutput { ca_spike };
        }
        #[cfg(not(feature = "biophys-l4-ca"))]
        {
            L4StepOutput { ca_spike: false }
        }
    }

    #[cfg(feature = "biophys-l4-synapses")]
    pub fn step_with_synapses(
        &mut self,
        state: &mut L4State,
        input_current: &[f32],
        synaptic: &[SynapseAccumulator],
    ) {
        self.step_with_synapses_output(state, input_current, synaptic);
    }

    #[cfg(feature = "biophys-l4-synapses")]
    pub fn step_with_synapses_output(
        &mut self,
        state: &mut L4State,
        input_current: &[f32],
        synaptic: &[SynapseAccumulator],
    ) -> L4StepOutput {
        assert_eq!(
            state.voltages.len(),
            self.morphology.compartments.len(),
            "state voltage count must match compartments"
        );
        assert_eq!(
            input_current.len(),
            self.morphology.compartments.len(),
            "input current count must match compartments"
        );
        assert_eq!(
            synaptic.len(),
            self.morphology.compartments.len(),
            "synaptic count must match compartments"
        );

        for (v, gates) in state.voltages.iter().copied().zip(state.gates.iter_mut()) {
            gates.update(v, self.dt_ms);
        }

        #[cfg(feature = "biophys-l4-ca")]
        let mut ca_spike = false;

        for current in self.axial_currents.iter_mut() {
            *current = 0.0;
        }
        for (parent_index, children) in self.children_indices.iter().enumerate() {
            for &child_index in children {
                let compartment = &self.morphology.compartments[child_index];
                let v_child = state.voltages[child_index];
                let v_parent = state.voltages[parent_index];
                let resistance = compartment.axial_resistance.max(1e-6);
                let current = (v_parent - v_child) / resistance;
                self.axial_currents[child_index] += current;
                self.axial_currents[parent_index] -= current;
            }
        }

        for (index, compartment) in self.morphology.compartments.iter().enumerate() {
            let channels = self.channels[index];
            let v = state.voltages[index];
            let gates = state.gates[index];
            let mut ionic = leak_current(channels.leak, v);
            if let Some(nak) = channels.nak {
                ionic += nak_current(nak, gates, v);
            }
            #[cfg(feature = "biophys-l4-ca")]
            if let Some(ca) = channels.ca {
                if is_distal(compartment.depth, self.max_depth) {
                    let p_inf_q = ca_p_inf_q(v);
                    let current_q = state.p_ca_q[index] as i32;
                    let updated_q =
                        (current_q + (p_inf_q as i32 - current_q) / TAU_CA_STEPS).clamp(0, 1000);
                    state.p_ca_q[index] = updated_q as u16;
                    ionic += ca_current(ca, state.p_ca_q[index], v);
                }
            }
            let axial = self.axial_currents[index];
            let ext = input_current[index];
            let syn = synaptic[index].total_current(v);
            let capacitance = compartment.capacitance.max(1e-6);
            let dv = self.dt_ms * (ext + syn + axial - ionic) / capacitance;
            let updated = (v + dv).clamp(self.clamp_min, self.clamp_max);
            #[cfg(feature = "biophys-l4-ca")]
            if is_distal(compartment.depth, self.max_depth)
                && v < CA_SPIKE_THRESHOLD_MV
                && updated >= CA_SPIKE_THRESHOLD_MV
            {
                ca_spike = true;
            }
            state.voltages[index] = updated;
        }

        self.step_count = self.step_count.saturating_add(1);
        #[cfg(feature = "biophys-l4-ca")]
        {
            return L4StepOutput { ca_spike };
        }
        #[cfg(not(feature = "biophys-l4-ca"))]
        {
            L4StepOutput { ca_spike: false }
        }
    }

    pub fn config_digest(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"UCF:L4:CFG");
        update_u32(&mut hasher, self.morphology.neuron_id.0);
        update_u32(&mut hasher, self.morphology.compartments.len() as u32);
        for (compartment, channels) in self
            .morphology
            .compartments
            .iter()
            .zip(self.channels.iter())
        {
            update_u32(&mut hasher, compartment.id.0);
            update_u32(&mut hasher, compartment.parent.map_or(u32::MAX, |id| id.0));
            update_u8(&mut hasher, compartment_kind_code(compartment.kind));
            update_u32(&mut hasher, compartment.depth);
            update_f32(&mut hasher, compartment.capacitance);
            update_f32(&mut hasher, compartment.axial_resistance);
            update_f32(&mut hasher, channels.leak.g);
            update_f32(&mut hasher, channels.leak.e_rev);
            match channels.nak {
                Some(nak) => {
                    update_u8(&mut hasher, 1);
                    update_f32(&mut hasher, nak.g_na);
                    update_f32(&mut hasher, nak.g_k);
                    update_f32(&mut hasher, nak.e_na);
                    update_f32(&mut hasher, nak.e_k);
                }
                None => {
                    update_u8(&mut hasher, 0);
                }
            }
            #[cfg(feature = "biophys-l4-ca")]
            match channels.ca {
                Some(ca) => {
                    update_u8(&mut hasher, 1);
                    update_f32(&mut hasher, ca.g_ca);
                    update_f32(&mut hasher, ca.e_ca);
                }
                None => {
                    update_u8(&mut hasher, 0);
                }
            }
        }
        *hasher.finalize().as_bytes()
    }

    pub fn snapshot_digest(&self, state: &L4State) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"UCF:L4:SNAP");
        update_u64(&mut hasher, self.step_count);
        update_u32(&mut hasher, state.voltages.len() as u32);
        for (v, gates) in state.voltages.iter().zip(state.gates.iter()) {
            update_f32(&mut hasher, *v);
            update_f32(&mut hasher, gates.m);
            update_f32(&mut hasher, gates.h);
            update_f32(&mut hasher, gates.n);
        }
        #[cfg(feature = "biophys-l4-ca")]
        for p_ca_q in &state.p_ca_q {
            update_u32(&mut hasher, *p_ca_q as u32);
        }
        *hasher.finalize().as_bytes()
    }
}

fn compartment_kind_code(kind: CompartmentKind) -> u8 {
    match kind {
        CompartmentKind::Soma => 0,
        CompartmentKind::Dendrite => 1,
        CompartmentKind::Axon => 2,
    }
}

fn update_u8(hasher: &mut blake3::Hasher, value: u8) {
    hasher.update(&[value]);
}

fn update_u32(hasher: &mut blake3::Hasher, value: u32) {
    hasher.update(&value.to_le_bytes());
}

fn update_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_le_bytes());
}

fn update_f32(hasher: &mut blake3::Hasher, value: f32) {
    hasher.update(&value.to_bits().to_le_bytes());
}

#[cfg(feature = "biophys-l4-ca")]
fn is_distal(comp_depth: u32, max_depth: u32) -> bool {
    comp_depth >= max_depth.saturating_sub(1)
}
