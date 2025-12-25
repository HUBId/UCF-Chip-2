#![forbid(unsafe_code)]

const MAX_COMPARTMENTS_PER_NEURON: usize = 64;
const MAX_EDGES_PER_NEURON: usize = 64;
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

impl MorphologySet {
    pub fn to_canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        push_u32(&mut bytes, self.version);
        let mut neurons = self.neurons.clone();
        neurons.sort_by_key(|n| n.neuron_id);
        push_u32(&mut bytes, neurons.len() as u32);
        for neuron in neurons {
            push_u32(&mut bytes, neuron.neuron_id);
            let mut compartments = neuron.compartments.clone();
            compartments.sort_by_key(|c| c.comp_id);
            push_u32(&mut bytes, compartments.len() as u32);
            for comp in compartments {
                push_u32(&mut bytes, comp.comp_id);
                match comp.parent {
                    Some(parent) => {
                        push_u8(&mut bytes, 1);
                        push_u32(&mut bytes, parent);
                    }
                    None => push_u8(&mut bytes, 0),
                }
                push_u8(&mut bytes, comp.kind as u8);
                push_u16(&mut bytes, comp.length_um);
                push_u16(&mut bytes, comp.diameter_um);
            }
            if self.version >= 2 {
                let mut labels = neuron.labels.clone();
                labels.sort_by(|a, b| {
                    (a.k.as_str(), a.v.as_str()).cmp(&(b.k.as_str(), b.v.as_str()))
                });
                assert!(
                    labels.len() <= MAX_LABELS_PER_NEURON,
                    "label count {} exceeds max {}",
                    labels.len(),
                    MAX_LABELS_PER_NEURON
                );
                push_u32(&mut bytes, labels.len() as u32);
                for label in labels {
                    assert!(
                        label.k.len() <= MAX_LABEL_KEY_LEN,
                        "label key too long: {}",
                        label.k.len()
                    );
                    assert!(
                        label.v.len() <= MAX_LABEL_VALUE_LEN,
                        "label value too long: {}",
                        label.v.len()
                    );
                    push_u16(&mut bytes, label.k.len() as u16);
                    bytes.extend_from_slice(label.k.as_bytes());
                    push_u16(&mut bytes, label.v.len() as u16);
                    bytes.extend_from_slice(label.v.as_bytes());
                }
            } else {
                debug_assert!(
                    neuron.labels.is_empty(),
                    "labels require morphology version >= 2"
                );
            }
        }
        bytes
    }

    pub fn digest(&self) -> [u8; 32] {
        artifact_digest("UCF:ASSET:MORPH", &self.to_canonical_bytes())
    }
}

impl ChannelParamsSet {
    pub fn to_canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        push_u32(&mut bytes, self.version);
        let mut params = self.per_compartment.clone();
        params.sort_by_key(|p| (p.neuron_id, p.comp_id));
        push_u32(&mut bytes, params.len() as u32);
        for param in params {
            push_u32(&mut bytes, param.neuron_id);
            push_u32(&mut bytes, param.comp_id);
            push_u16(&mut bytes, param.leak_g);
            push_u16(&mut bytes, param.na_g);
            push_u16(&mut bytes, param.k_g);
        }
        bytes
    }

    pub fn digest(&self) -> [u8; 32] {
        artifact_digest("UCF:ASSET:CHANNEL_PARAMS", &self.to_canonical_bytes())
    }
}

impl SynapseParamsSet {
    pub fn to_canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        push_u32(&mut bytes, self.version);
        let mut params = self.params.clone();
        params.sort_by_key(|p| {
            (
                p.syn_type,
                p.weight_base,
                p.stp_u,
                p.tau_rec,
                p.tau_fac,
                p.mod_channel,
            )
        });
        push_u32(&mut bytes, params.len() as u32);
        for param in params {
            push_u8(&mut bytes, param.syn_type as u8);
            push_i32(&mut bytes, param.weight_base);
            push_u16(&mut bytes, param.stp_u);
            push_u16(&mut bytes, param.tau_rec);
            push_u16(&mut bytes, param.tau_fac);
            push_u8(&mut bytes, param.mod_channel as u8);
        }
        bytes
    }

    pub fn digest(&self) -> [u8; 32] {
        artifact_digest("UCF:ASSET:SYN_PARAMS", &self.to_canonical_bytes())
    }
}

impl ConnectivityGraph {
    pub fn to_canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        push_u32(&mut bytes, self.version);
        let mut edges = self.edges.clone();
        edges.sort_by_key(|e| (e.pre, e.post, e.syn_type, e.delay_steps, e.syn_param_id));
        push_u32(&mut bytes, edges.len() as u32);
        for edge in edges {
            push_u32(&mut bytes, edge.pre);
            push_u32(&mut bytes, edge.post);
            push_u8(&mut bytes, edge.syn_type as u8);
            push_u16(&mut bytes, edge.delay_steps);
            push_u32(&mut bytes, edge.syn_param_id);
        }
        bytes
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

fn push_u8(bytes: &mut Vec<u8>, value: u8) {
    bytes.push(value);
}

fn push_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_i32(bytes: &mut Vec<u8>, value: i32) {
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
}
