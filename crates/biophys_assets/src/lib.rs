#![forbid(unsafe_code)]

use prost::Message;
use ucf::v1::{
    ChannelParamsPayload, ChannelParamsSetPayload, CompartmentKind as PayloadCompartmentKind,
    CompartmentPayload, ConnectivityEdgePayload, ConnectivityGraphPayload, LabelKv,
    ModChannel as PayloadModChannel, MorphNeuronPayload, MorphologySetPayload,
    SynapseParamsPayload, SynapseParamsSetPayload, SynapseType as PayloadSynapseType,
};

const MAX_COMPARTMENTS_PER_NEURON: usize = 64;
const MAX_EDGES_PER_NEURON: usize = 64;
pub const MAX_NEURONS: usize = 4096;
pub const MAX_EDGES: usize = MAX_NEURONS * MAX_EDGES_PER_NEURON;
pub const MAX_LABELS_PER_NEURON: usize = 8;
pub const MAX_LABEL_KEY_LEN: usize = 32;
pub const MAX_LABEL_VALUE_LEN: usize = 64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MorphologySet {
    pub version: u32,
    pub neurons: Vec<MorphNeuron>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MorphNeuron {
    pub neuron_id: u32,
    pub compartments: Vec<Compartment>,
    pub labels: Vec<LabelKV>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabelKV {
    pub k: String,
    pub v: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Compartment {
    pub comp_id: u32,
    pub parent: Option<u32>,
    pub kind: CompartmentKind,
    pub length_um: u16,
    pub diameter_um: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Ord, PartialOrd)]
pub enum CompartmentKind {
    Soma,
    Dendrite,
    Axon,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChannelParamsSet {
    pub version: u32,
    pub per_compartment: Vec<ChannelParams>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChannelParams {
    pub neuron_id: u32,
    pub comp_id: u32,
    pub leak_g: u16,
    pub na_g: u16,
    pub k_g: u16,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SynapseParamsSet {
    pub version: u32,
    pub params: Vec<SynapseParams>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SynapseParams {
    pub syn_type: SynType,
    pub weight_base: i32,
    pub stp_u: u16,
    pub tau_rec: u16,
    pub tau_fac: u16,
    pub mod_channel: ModChannel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Ord, PartialOrd)]
pub enum SynType {
    Exc,
    Inh,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Ord, PartialOrd)]
pub enum ModChannel {
    None,
    A,
    B,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConnectivityGraph {
    pub version: u32,
    pub edges: Vec<ConnEdge>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConnEdge {
    pub pre: u32,
    pub post: u32,
    pub syn_type: SynType,
    pub delay_steps: u16,
    pub syn_param_id: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssetManifest {
    pub manifest_version: u32,
    pub morph_digest: [u8; 32],
    pub channel_params_digest: [u8; 32],
    pub syn_params_digest: [u8; 32],
    pub connectivity_digest: [u8; 32],
    pub created_at_ms: u64,
}

pub fn to_asset_digest(
    kind: ucf::v1::AssetKind,
    version: u32,
    digest: [u8; 32],
    created_at_ms: u64,
    prev: Option<[u8; 32]>,
) -> ucf::v1::AssetDigest {
    let digest_bytes = digest.to_vec();
    debug_assert_eq!(digest_bytes.len(), 32);
    ucf::v1::AssetDigest {
        kind: kind as i32,
        version,
        digest: digest_bytes,
        created_at_ms,
        prev_digest: prev.map(|bytes| bytes.to_vec()),
    }
}

pub fn artifact_digest(domain: &str, bytes: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain.as_bytes());
    hasher.update(&[0u8]);
    hasher.update(bytes);
    *hasher.finalize().as_bytes()
}

pub fn demo_morphology_3comp(neurons: u32) -> MorphologySet {
    let mut morph_neurons = Vec::with_capacity(neurons as usize);
    for neuron_id in 0..neurons {
        let compartments = vec![
            Compartment {
                comp_id: 0,
                parent: None,
                kind: CompartmentKind::Soma,
                length_um: 10 + (neuron_id as u16 % 5),
                diameter_um: 8 + (neuron_id as u16 % 3),
            },
            Compartment {
                comp_id: 1,
                parent: Some(0),
                kind: CompartmentKind::Dendrite,
                length_um: 20 + (neuron_id as u16 % 7),
                diameter_um: 4 + (neuron_id as u16 % 2),
            },
            Compartment {
                comp_id: 2,
                parent: Some(0),
                kind: CompartmentKind::Axon,
                length_um: 30 + (neuron_id as u16 % 11),
                diameter_um: 3 + (neuron_id as u16 % 2),
            },
        ];
        debug_assert!(compartments.len() <= MAX_COMPARTMENTS_PER_NEURON);
        morph_neurons.push(MorphNeuron {
            neuron_id,
            compartments,
            labels: Vec::new(),
        });
    }
    MorphologySet {
        version: 1,
        neurons: morph_neurons,
    }
}

pub fn morphology_tree(neurons: u32, compartments_per_neuron: u16) -> MorphologySet {
    let mut morph_neurons = Vec::with_capacity(neurons as usize);
    for neuron_id in 0..neurons {
        let compartments = build_tree_compartments(compartments_per_neuron);
        debug_assert!(compartments.len() <= MAX_COMPARTMENTS_PER_NEURON);
        morph_neurons.push(MorphNeuron {
            neuron_id,
            compartments,
            labels: Vec::new(),
        });
    }
    MorphologySet {
        version: 1,
        neurons: morph_neurons,
    }
}

pub fn demo_channel_params(morph: &MorphologySet) -> ChannelParamsSet {
    let mut per_compartment = Vec::new();
    for neuron in &morph.neurons {
        for comp in &neuron.compartments {
            let has_nak = comp.kind == CompartmentKind::Soma;
            per_compartment.push(ChannelParams {
                neuron_id: neuron.neuron_id,
                comp_id: comp.comp_id,
                leak_g: 1 + (comp.comp_id as u16),
                na_g: if has_nak {
                    2 + (neuron.neuron_id as u16 % 5)
                } else {
                    0
                },
                k_g: if has_nak {
                    3 + (comp.comp_id as u16 % 4)
                } else {
                    0
                },
            });
        }
    }
    ChannelParamsSet {
        version: 1,
        per_compartment,
    }
}

pub fn demo_syn_params() -> SynapseParamsSet {
    SynapseParamsSet {
        version: 1,
        params: vec![
            SynapseParams {
                syn_type: SynType::Exc,
                weight_base: 5,
                stp_u: 100,
                tau_rec: 200,
                tau_fac: 50,
                mod_channel: ModChannel::None,
            },
            SynapseParams {
                syn_type: SynType::Inh,
                weight_base: -4,
                stp_u: 80,
                tau_rec: 180,
                tau_fac: 40,
                mod_channel: ModChannel::A,
            },
        ],
    }
}

/// Deterministic fanout: each neuron connects to the next neuron only.
/// Fanout <= 1, fanin <= 1, bounded by MAX_EDGES_PER_NEURON.
pub fn demo_connectivity(neurons: u32, syn_params: &SynapseParamsSet) -> ConnectivityGraph {
    let mut edges = Vec::new();
    if neurons == 0 {
        return ConnectivityGraph { version: 1, edges };
    }
    let syn_param_id = syn_params
        .params
        .iter()
        .position(|p| p.syn_type == SynType::Exc)
        .unwrap_or(0) as u32;

    for pre in 0..neurons {
        let post = (pre + 1) % neurons;
        edges.push(ConnEdge {
            pre,
            post,
            syn_type: SynType::Exc,
            delay_steps: 1 + (pre as u16 % 3),
            syn_param_id,
        });
    }
    debug_assert!(edges.len() <= (neurons as usize) * MAX_EDGES_PER_NEURON);
    ConnectivityGraph { version: 1, edges }
}

pub fn build_morphology_payload(morph: &MorphologySet) -> MorphologySetPayload {
    let neurons = morph
        .neurons
        .iter()
        .map(|neuron| {
            if morph.version < 2 {
                debug_assert!(
                    neuron.labels.is_empty(),
                    "labels require morphology version >= 2"
                );
            }
            debug_assert!(
                neuron.compartments.len() <= MAX_COMPARTMENTS_PER_NEURON,
                "compartment count {} exceeds max {}",
                neuron.compartments.len(),
                MAX_COMPARTMENTS_PER_NEURON
            );
            debug_assert!(
                neuron.labels.len() <= MAX_LABELS_PER_NEURON,
                "label count {} exceeds max {}",
                neuron.labels.len(),
                MAX_LABELS_PER_NEURON
            );
            let compartments = neuron
                .compartments
                .iter()
                .map(|comp| CompartmentPayload {
                    comp_id: comp.comp_id,
                    parent: comp.parent,
                    kind: match comp.kind {
                        CompartmentKind::Soma => PayloadCompartmentKind::Soma as i32,
                        CompartmentKind::Dendrite => PayloadCompartmentKind::Dendrite as i32,
                        CompartmentKind::Axon => PayloadCompartmentKind::Axon as i32,
                    },
                    length_um: comp.length_um as u32,
                    diameter_um: comp.diameter_um as u32,
                })
                .collect();
            let labels = neuron
                .labels
                .iter()
                .map(|label| {
                    debug_assert!(
                        label.k.len() <= MAX_LABEL_KEY_LEN,
                        "label key too long: {}",
                        label.k.len()
                    );
                    debug_assert!(
                        label.v.len() <= MAX_LABEL_VALUE_LEN,
                        "label value too long: {}",
                        label.v.len()
                    );
                    LabelKv {
                        k: label.k.clone(),
                        v: label.v.clone(),
                    }
                })
                .collect();
            MorphNeuronPayload {
                neuron_id: neuron.neuron_id,
                compartments,
                labels,
            }
        })
        .collect();
    let mut payload = MorphologySetPayload {
        version: morph.version,
        neurons,
    };
    normalize_payload_morphology(&mut payload);
    payload
}

pub fn build_channel_params_payload(params: &ChannelParamsSet) -> ChannelParamsSetPayload {
    let entries = params
        .per_compartment
        .iter()
        .map(|param| ChannelParamsPayload {
            neuron_id: param.neuron_id,
            comp_id: param.comp_id,
            leak_g: param.leak_g as u32,
            na_g: param.na_g as u32,
            k_g: param.k_g as u32,
        })
        .collect();
    let mut payload = ChannelParamsSetPayload {
        version: params.version,
        params: entries,
    };
    normalize_payload_channel_params(&mut payload);
    payload
}

pub fn build_synapse_params_payload(params: &SynapseParamsSet) -> SynapseParamsSetPayload {
    let entries = params
        .params
        .iter()
        .enumerate()
        .map(|(idx, param)| SynapseParamsPayload {
            syn_param_id: idx as u32,
            syn_type: match param.syn_type {
                SynType::Exc => PayloadSynapseType::Exc as i32,
                SynType::Inh => PayloadSynapseType::Inh as i32,
            },
            weight_base: param.weight_base,
            stp_u: param.stp_u as u32,
            tau_rec: param.tau_rec as u32,
            tau_fac: param.tau_fac as u32,
            mod_channel: match param.mod_channel {
                ModChannel::None => PayloadModChannel::None as i32,
                ModChannel::A => PayloadModChannel::A as i32,
                ModChannel::B => PayloadModChannel::B as i32,
            },
        })
        .collect();
    let mut payload = SynapseParamsSetPayload {
        version: params.version,
        params: entries,
    };
    normalize_payload_synapse_params(&mut payload);
    payload
}

pub fn build_connectivity_payload(graph: &ConnectivityGraph) -> ConnectivityGraphPayload {
    debug_assert!(
        graph.edges.len() <= MAX_EDGES,
        "edge count {} exceeds max {}",
        graph.edges.len(),
        MAX_EDGES
    );
    let edges = graph
        .edges
        .iter()
        .map(|edge| ConnectivityEdgePayload {
            pre: edge.pre,
            post: edge.post,
            post_compartment: 0,
            syn_param_id: edge.syn_param_id,
            delay_steps: edge.delay_steps as u32,
        })
        .collect();
    let mut payload = ConnectivityGraphPayload {
        version: graph.version,
        edges,
    };
    normalize_payload_connectivity(&mut payload);
    payload
}

pub fn normalize_payload_morphology(payload: &mut MorphologySetPayload) {
    payload.neurons.sort_by_key(|neuron| neuron.neuron_id);
    for neuron in &mut payload.neurons {
        neuron.compartments.sort_by_key(|comp| comp.comp_id);
        neuron
            .labels
            .sort_by(|a, b| (a.k.as_str(), a.v.as_str()).cmp(&(b.k.as_str(), b.v.as_str())));
    }
}

pub fn normalize_payload_channel_params(payload: &mut ChannelParamsSetPayload) {
    payload
        .params
        .sort_by_key(|param| (param.neuron_id, param.comp_id));
}

pub fn normalize_payload_synapse_params(payload: &mut SynapseParamsSetPayload) {
    payload.params.sort_by_key(|param| param.syn_param_id);
}

pub fn normalize_payload_connectivity(payload: &mut ConnectivityGraphPayload) {
    payload.edges.sort_by_key(|edge| {
        (
            edge.pre,
            edge.post,
            edge.post_compartment,
            edge.syn_param_id,
            edge.delay_steps,
        )
    });
}

pub fn morphology_payload_bytes(payload: &MorphologySetPayload) -> Vec<u8> {
    let mut normalized = payload.clone();
    normalize_payload_morphology(&mut normalized);
    normalized.encode_to_vec()
}

pub fn channel_params_payload_bytes(payload: &ChannelParamsSetPayload) -> Vec<u8> {
    let mut normalized = payload.clone();
    normalize_payload_channel_params(&mut normalized);
    normalized.encode_to_vec()
}

pub fn synapse_params_payload_bytes(payload: &SynapseParamsSetPayload) -> Vec<u8> {
    let mut normalized = payload.clone();
    normalize_payload_synapse_params(&mut normalized);
    normalized.encode_to_vec()
}

pub fn connectivity_payload_bytes(payload: &ConnectivityGraphPayload) -> Vec<u8> {
    let mut normalized = payload.clone();
    normalize_payload_connectivity(&mut normalized);
    normalized.encode_to_vec()
}

pub fn morphology_payload_digest(payload: &MorphologySetPayload) -> [u8; 32] {
    artifact_digest("UCF:ASSET:MORPH", &morphology_payload_bytes(payload))
}

pub fn channel_params_payload_digest(payload: &ChannelParamsSetPayload) -> [u8; 32] {
    artifact_digest(
        "UCF:ASSET:CHANNEL_PARAMS",
        &channel_params_payload_bytes(payload),
    )
}

pub fn synapse_params_payload_digest(payload: &SynapseParamsSetPayload) -> [u8; 32] {
    artifact_digest(
        "UCF:ASSET:SYN_PARAMS",
        &synapse_params_payload_bytes(payload),
    )
}

pub fn connectivity_payload_digest(payload: &ConnectivityGraphPayload) -> [u8; 32] {
    artifact_digest(
        "UCF:ASSET:CONNECTIVITY",
        &connectivity_payload_bytes(payload),
    )
}

pub fn morphology_from_payload(payload: &MorphologySetPayload) -> Result<MorphologySet, String> {
    if payload.neurons.len() > MAX_NEURONS {
        return Err(format!(
            "neuron count {} exceeds max {}",
            payload.neurons.len(),
            MAX_NEURONS
        ));
    }
    let mut neurons = Vec::with_capacity(payload.neurons.len());
    for neuron in &payload.neurons {
        if payload.version < 2 && !neuron.labels.is_empty() {
            return Err("labels require morphology version >= 2".to_string());
        }
        if neuron.compartments.len() > MAX_COMPARTMENTS_PER_NEURON {
            return Err(format!(
                "compartment count {} exceeds max {}",
                neuron.compartments.len(),
                MAX_COMPARTMENTS_PER_NEURON
            ));
        }
        if neuron.labels.len() > MAX_LABELS_PER_NEURON {
            return Err(format!(
                "label count {} exceeds max {}",
                neuron.labels.len(),
                MAX_LABELS_PER_NEURON
            ));
        }
        let mut compartments = Vec::with_capacity(neuron.compartments.len());
        for comp in &neuron.compartments {
            let kind = PayloadCompartmentKind::try_from(comp.kind)
                .map_err(|_| format!("invalid compartment kind {}", comp.kind))?;
            let length_um = u16_from_u32(comp.length_um, "length_um")?;
            let diameter_um = u16_from_u32(comp.diameter_um, "diameter_um")?;
            compartments.push(Compartment {
                comp_id: comp.comp_id,
                parent: comp.parent,
                kind: match kind {
                    PayloadCompartmentKind::Soma => CompartmentKind::Soma,
                    PayloadCompartmentKind::Dendrite => CompartmentKind::Dendrite,
                    PayloadCompartmentKind::Axon => CompartmentKind::Axon,
                },
                length_um,
                diameter_um,
            });
        }
        let mut labels = Vec::with_capacity(neuron.labels.len());
        for label in &neuron.labels {
            if label.k.len() > MAX_LABEL_KEY_LEN {
                return Err(format!("label key too long: {}", label.k.len()));
            }
            if label.v.len() > MAX_LABEL_VALUE_LEN {
                return Err(format!("label value too long: {}", label.v.len()));
            }
            labels.push(LabelKV {
                k: label.k.clone(),
                v: label.v.clone(),
            });
        }
        neurons.push(MorphNeuron {
            neuron_id: neuron.neuron_id,
            compartments,
            labels,
        });
    }
    Ok(MorphologySet {
        version: payload.version,
        neurons,
    })
}

pub fn channel_params_from_payload(
    payload: &ChannelParamsSetPayload,
) -> Result<ChannelParamsSet, String> {
    let mut per_compartment = Vec::with_capacity(payload.params.len());
    for param in &payload.params {
        per_compartment.push(ChannelParams {
            neuron_id: param.neuron_id,
            comp_id: param.comp_id,
            leak_g: u16_from_u32(param.leak_g, "leak_g")?,
            na_g: u16_from_u32(param.na_g, "na_g")?,
            k_g: u16_from_u32(param.k_g, "k_g")?,
        });
    }
    Ok(ChannelParamsSet {
        version: payload.version,
        per_compartment,
    })
}

pub fn synapse_params_from_payload(
    payload: &SynapseParamsSetPayload,
) -> Result<SynapseParamsSet, String> {
    let mut entries = Vec::with_capacity(payload.params.len());
    let mut seen = std::collections::BTreeSet::new();
    for param in &payload.params {
        if !seen.insert(param.syn_param_id) {
            return Err(format!("duplicate syn_param_id {}", param.syn_param_id));
        }
        let syn_type = PayloadSynapseType::try_from(param.syn_type)
            .map_err(|_| format!("invalid synapse type {}", param.syn_type))?;
        let mod_channel = PayloadModChannel::try_from(param.mod_channel)
            .map_err(|_| format!("invalid mod channel {}", param.mod_channel))?;
        entries.push((
            param.syn_param_id,
            SynapseParams {
                syn_type: match syn_type {
                    PayloadSynapseType::Exc => SynType::Exc,
                    PayloadSynapseType::Inh => SynType::Inh,
                },
                weight_base: param.weight_base,
                stp_u: u16_from_u32(param.stp_u, "stp_u")?,
                tau_rec: u16_from_u32(param.tau_rec, "tau_rec")?,
                tau_fac: u16_from_u32(param.tau_fac, "tau_fac")?,
                mod_channel: match mod_channel {
                    PayloadModChannel::None => ModChannel::None,
                    PayloadModChannel::A => ModChannel::A,
                    PayloadModChannel::B => ModChannel::B,
                },
            },
        ));
    }
    entries.sort_by_key(|(syn_param_id, _)| *syn_param_id);
    for (idx, (syn_param_id, _)) in entries.iter().enumerate() {
        if *syn_param_id != idx as u32 {
            return Err("syn_param_id entries must be contiguous starting at 0".to_string());
        }
    }
    let params = entries.into_iter().map(|(_, param)| param).collect();
    Ok(SynapseParamsSet {
        version: payload.version,
        params,
    })
}

pub fn connectivity_from_payload(
    payload: &ConnectivityGraphPayload,
    syn_payload: &SynapseParamsSetPayload,
) -> Result<ConnectivityGraph, String> {
    if payload.edges.len() > MAX_EDGES {
        return Err(format!(
            "edge count {} exceeds max {}",
            payload.edges.len(),
            MAX_EDGES
        ));
    }
    let syn_map = synapse_type_map(syn_payload)?;
    let mut edges = Vec::with_capacity(payload.edges.len());
    for edge in &payload.edges {
        if edge.post_compartment != 0 {
            return Err(format!(
                "post_compartment {} not supported",
                edge.post_compartment
            ));
        }
        let syn_type = syn_map
            .get(&edge.syn_param_id)
            .ok_or_else(|| format!("missing synapse params id {}", edge.syn_param_id))?;
        edges.push(ConnEdge {
            pre: edge.pre,
            post: edge.post,
            syn_type: *syn_type,
            delay_steps: u16_from_u32(edge.delay_steps, "delay_steps")?,
            syn_param_id: edge.syn_param_id,
        });
    }
    Ok(ConnectivityGraph {
        version: payload.version,
        edges,
    })
}

fn synapse_type_map(
    payload: &SynapseParamsSetPayload,
) -> Result<std::collections::BTreeMap<u32, SynType>, String> {
    let mut map = std::collections::BTreeMap::new();
    for param in &payload.params {
        let syn_type = PayloadSynapseType::try_from(param.syn_type)
            .map_err(|_| format!("invalid synapse type {}", param.syn_type))?;
        let entry = match syn_type {
            PayloadSynapseType::Exc => SynType::Exc,
            PayloadSynapseType::Inh => SynType::Inh,
        };
        if map.insert(param.syn_param_id, entry).is_some() {
            return Err(format!("duplicate syn_param_id {}", param.syn_param_id));
        }
    }
    Ok(map)
}

fn u16_from_u32(value: u32, label: &str) -> Result<u16, String> {
    u16::try_from(value).map_err(|_| format!("{label} value {value} exceeds u16 max"))
}

impl MorphologySet {
    pub fn to_canonical_bytes(&self) -> Vec<u8> {
        let payload = build_morphology_payload(self);
        morphology_payload_bytes(&payload)
    }

    pub fn digest(&self) -> [u8; 32] {
        artifact_digest("UCF:ASSET:MORPH", &self.to_canonical_bytes())
    }
}

impl ChannelParamsSet {
    pub fn to_canonical_bytes(&self) -> Vec<u8> {
        let payload = build_channel_params_payload(self);
        channel_params_payload_bytes(&payload)
    }

    pub fn digest(&self) -> [u8; 32] {
        artifact_digest("UCF:ASSET:CHANNEL_PARAMS", &self.to_canonical_bytes())
    }
}

impl SynapseParamsSet {
    pub fn to_canonical_bytes(&self) -> Vec<u8> {
        let payload = build_synapse_params_payload(self);
        synapse_params_payload_bytes(&payload)
    }

    pub fn digest(&self) -> [u8; 32] {
        artifact_digest("UCF:ASSET:SYN_PARAMS", &self.to_canonical_bytes())
    }
}

impl ConnectivityGraph {
    pub fn to_canonical_bytes(&self) -> Vec<u8> {
        let payload = build_connectivity_payload(self);
        connectivity_payload_bytes(&payload)
    }

    pub fn digest(&self) -> [u8; 32] {
        artifact_digest("UCF:ASSET:CONNECTIVITY", &self.to_canonical_bytes())
    }
}

impl AssetManifest {
    pub fn to_canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        push_u32(&mut bytes, self.manifest_version);
        bytes.extend_from_slice(&self.morph_digest);
        bytes.extend_from_slice(&self.channel_params_digest);
        bytes.extend_from_slice(&self.syn_params_digest);
        bytes.extend_from_slice(&self.connectivity_digest);
        push_u64(&mut bytes, self.created_at_ms);
        bytes
    }

    pub fn digest(&self) -> [u8; 32] {
        artifact_digest("UCF:ASSET:MANIFEST", &self.to_canonical_bytes())
    }
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn build_tree_compartments(compartments_per_neuron: u16) -> Vec<Compartment> {
    let plan = tree_plan(compartments_per_neuron);
    let mut compartments = Vec::with_capacity(plan.total as usize);
    let mut next_id = 0u32;

    compartments.push(Compartment {
        comp_id: next_id,
        parent: None,
        kind: CompartmentKind::Soma,
        length_um: length_for_depth(0),
        diameter_um: diameter_for_depth(0),
    });
    next_id += 1;

    let mut current_level = vec![0u32];
    for depth in 1..=plan.depths {
        let mut next_level = Vec::new();
        let count = plan.children_per_depth[(depth - 1) as usize];
        let mut parent_index = 0usize;
        for _ in 0..count {
            let parent_id = current_level[parent_index];
            if !current_level.is_empty() {
                parent_index = (parent_index + 1) % current_level.len();
            }
            let comp_id = next_id;
            next_id += 1;
            compartments.push(Compartment {
                comp_id,
                parent: Some(parent_id),
                kind: CompartmentKind::Dendrite,
                length_um: length_for_depth(depth),
                diameter_um: diameter_for_depth(depth),
            });
            next_level.push(comp_id);
        }
        current_level = next_level;
    }

    debug_assert_eq!(compartments.len(), plan.total as usize);
    debug_assert!(compartments.len() <= MAX_COMPARTMENTS_PER_NEURON);
    debug_assert!(compartments.iter().all(|comp| comp.comp_id < plan.total));
    debug_assert!(compartments
        .iter()
        .all(|comp| comp.length_um > 0 && comp.diameter_um > 0));
    debug_assert!(
        compartments[0].length_um != 0 && compartments[0].diameter_um != 0,
        "soma geometry must be set"
    );

    compartments
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

fn length_for_depth(depth: u16) -> u16 {
    let base = 10u16;
    let step = 5u16;
    let max = 60u16;
    (base + depth.saturating_mul(step)).min(max)
}

fn diameter_for_depth(depth: u16) -> u16 {
    let base = 12u16;
    let step = 2u16;
    let min = 3u16;
    base.saturating_sub(depth.saturating_mul(step)).max(min)
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    #[test]
    fn canonical_morphology_sorting() {
        let mut morph = demo_morphology_3comp(3);
        morph.neurons.reverse();
        for neuron in &mut morph.neurons {
            neuron.compartments.reverse();
        }
        let bytes_a = morph.to_canonical_bytes();
        let bytes_b = demo_morphology_3comp(3).to_canonical_bytes();
        assert_eq!(bytes_a, bytes_b);
    }

    #[test]
    fn canonical_label_sorting() {
        let mut morph = demo_morphology_3comp(1);
        morph.version = 2;
        let neuron = &mut morph.neurons[0];
        neuron.labels = vec![
            LabelKV {
                k: "role".to_string(),
                v: "E".to_string(),
            },
            LabelKV {
                k: "pool".to_string(),
                v: "SIM".to_string(),
            },
        ];
        let bytes_a = morph.to_canonical_bytes();

        let mut sorted = morph.clone();
        sorted.neurons[0]
            .labels
            .sort_by(|a, b| (a.k.as_str(), a.v.as_str()).cmp(&(b.k.as_str(), b.v.as_str())));
        let bytes_b = sorted.to_canonical_bytes();
        assert_eq!(bytes_a, bytes_b);
    }

    #[test]
    fn canonical_connectivity_sorting() {
        let syn = demo_syn_params();
        let mut graph = demo_connectivity(4, &syn);
        graph.edges.reverse();
        let bytes_a = graph.to_canonical_bytes();
        let bytes_b = demo_connectivity(4, &syn).to_canonical_bytes();
        assert_eq!(bytes_a, bytes_b);
    }

    #[test]
    fn digest_changes_on_field_update() {
        let mut morph = demo_morphology_3comp(1);
        let digest_a = morph.digest();
        morph.neurons[0].compartments[0].length_um += 1;
        let digest_b = morph.digest();
        assert_ne!(digest_a, digest_b);
    }

    #[test]
    fn digest_deterministic() {
        let morph = demo_morphology_3comp(2);
        let digest_a = morph.digest();
        let digest_b = morph.digest();
        assert_eq!(digest_a, digest_b);
    }

    #[test]
    fn manifest_digest_deterministic() {
        let morph = demo_morphology_3comp(2);
        let channels = demo_channel_params(&morph);
        let syn = demo_syn_params();
        let graph = demo_connectivity(2, &syn);
        let manifest = AssetManifest {
            manifest_version: 1,
            morph_digest: morph.digest(),
            channel_params_digest: channels.digest(),
            syn_params_digest: syn.digest(),
            connectivity_digest: graph.digest(),
            created_at_ms: 1234,
        };
        let digest_a = manifest.digest();
        let digest_b = manifest.digest();
        assert_eq!(digest_a, digest_b);
    }

    #[test]
    fn generator_bounds_hold() {
        let morph = demo_morphology_3comp(10);
        for neuron in &morph.neurons {
            assert!(neuron.compartments.len() <= MAX_COMPARTMENTS_PER_NEURON);
        }
        let syn = demo_syn_params();
        let graph = demo_connectivity(10, &syn);
        let mut counts = vec![0usize; 10];
        for edge in &graph.edges {
            counts[edge.pre as usize] += 1;
        }
        for count in counts {
            assert!(count <= MAX_EDGES_PER_NEURON);
        }
    }

    #[test]
    fn morphology_tree_deterministic_digest() {
        let morph = morphology_tree(2, 7);
        let digest_a = morph.digest();
        let digest_b = morph.digest();
        assert_eq!(digest_a, digest_b);
    }

    #[test]
    #[cfg(feature = "biophys-l4-morphology-multi")]
    fn morphology_tree_canonical_bytes_deterministic() {
        let bytes_a = morphology_tree(3, 15).to_canonical_bytes();
        let bytes_b = morphology_tree(3, 15).to_canonical_bytes();
        assert_eq!(bytes_a, bytes_b);
    }

    #[test]
    fn morphology_tree_sizes_are_bounded() {
        let morph = morphology_tree(4, 15);
        for neuron in &morph.neurons {
            assert!(neuron.compartments.len() <= MAX_COMPARTMENTS_PER_NEURON);
            assert_eq!(neuron.compartments.len(), 15);
        }
    }

    #[test]
    fn payload_canonical_roundtrip_deterministic() {
        let morph = demo_morphology_3comp(2);
        let morph_payload = build_morphology_payload(&morph);
        let bytes = morph_payload.encode_to_vec();
        let mut decoded = MorphologySetPayload::decode(bytes.as_slice()).expect("morph decode");
        normalize_payload_morphology(&mut decoded);
        let bytes_roundtrip = decoded.encode_to_vec();
        assert_eq!(bytes, bytes_roundtrip);

        let channel = demo_channel_params(&morph);
        let channel_payload = build_channel_params_payload(&channel);
        let bytes = channel_payload.encode_to_vec();
        let mut decoded =
            ChannelParamsSetPayload::decode(bytes.as_slice()).expect("channel decode");
        normalize_payload_channel_params(&mut decoded);
        let bytes_roundtrip = decoded.encode_to_vec();
        assert_eq!(bytes, bytes_roundtrip);

        let syn = demo_syn_params();
        let syn_payload = build_synapse_params_payload(&syn);
        let bytes = syn_payload.encode_to_vec();
        let mut decoded = SynapseParamsSetPayload::decode(bytes.as_slice()).expect("syn decode");
        normalize_payload_synapse_params(&mut decoded);
        let bytes_roundtrip = decoded.encode_to_vec();
        assert_eq!(bytes, bytes_roundtrip);

        let connectivity = demo_connectivity(2, &syn);
        let conn_payload = build_connectivity_payload(&connectivity);
        let bytes = conn_payload.encode_to_vec();
        let mut decoded =
            ConnectivityGraphPayload::decode(bytes.as_slice()).expect("connectivity decode");
        normalize_payload_connectivity(&mut decoded);
        let bytes_roundtrip = decoded.encode_to_vec();
        assert_eq!(bytes, bytes_roundtrip);
    }

    #[test]
    fn payload_digests_are_stable() {
        let morph = demo_morphology_3comp(2);
        let morph_payload = build_morphology_payload(&morph);
        let digest_a = morphology_payload_digest(&morph_payload);
        let digest_b = morphology_payload_digest(&morph_payload);
        assert_eq!(digest_a, digest_b);
    }
}
