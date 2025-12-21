#![forbid(unsafe_code)]

pub mod v1 {
    use prost::Enumeration;
    use prost::Message;
    use serde::{Deserialize, Serialize};

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Enumeration)]
    #[repr(i32)]
    pub enum MacroMilestoneState {
        Unknown = 0,
        Finalized = 1,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Enumeration)]
    #[repr(i32)]
    pub enum ConsistencyClass {
        Unknown = 0,
        ConsistencyLow = 1,
        ConsistencyMed = 2,
        ConsistencyHigh = 3,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Enumeration)]
    #[repr(i32)]
    pub enum WindowKind {
        Unknown = 0,
        Short = 1,
        Medium = 2,
        Long = 3,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Enumeration)]
    #[repr(i32)]
    pub enum IntegrityStateClass {
        Unknown = 0,
        Ok = 1,
        Degraded = 2,
        Fail = 3,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Enumeration)]
    #[repr(i32)]
    pub enum LevelClass {
        Unknown = 0,
        Low = 1,
        Med = 2,
        High = 3,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Enumeration)]
    #[repr(i32)]
    pub enum ReasonCode {
        Unknown = 0,
        ReIntegrityFail = 1,
        ReIntegrityDegraded = 2,
        RcGeExecDispatchBlocked = 3,
        RcThIntegrityCompromise = 4,
        RcReReplayMismatch = 5,
        ThExfilHighConfidence = 10,
        ThPolicyProbing = 11,
        RcGvCbvUpdated = 12,
        RcGvPevUpdated = 13,
        RcCdDlpSecretPattern = 20,
        RcCdDlpObfuscation = 21,
        RcCdDlpStegano = 22,
        RcRxActionForensic = 30,
        RcGvRecoveryUnlockGranted = 31,
        RcRgProfileM1Restricted = 32,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Enumeration)]
    #[repr(i32)]
    pub enum TraitUpdateDirection {
        Unknown = 0,
        IncreaseStrictness = 1,
        DecreaseStrictness = 2,
    }

    #[derive(Clone, PartialEq, Serialize, Deserialize, Message)]
    pub struct TraitUpdate {
        #[prost(string, tag = "1")]
        pub trait_name: ::prost::alloc::string::String,
        #[prost(enumeration = "LevelClass", tag = "2")]
        pub magnitude: i32,
        #[prost(enumeration = "TraitUpdateDirection", tag = "3")]
        pub direction: i32,
    }

    #[derive(Clone, PartialEq, Serialize, Deserialize, Message)]
    pub struct MacroMilestone {
        #[prost(enumeration = "MacroMilestoneState", tag = "1")]
        pub state: i32,
        #[prost(enumeration = "ConsistencyClass", tag = "2")]
        pub consistency_class: i32,
        #[prost(message, repeated, tag = "3")]
        pub trait_updates: ::prost::alloc::vec::Vec<TraitUpdate>,
        #[prost(bytes, optional, tag = "4")]
        pub macro_digest: Option<::prost::alloc::vec::Vec<u8>>,
    }

    #[derive(Clone, PartialEq, Serialize, Deserialize, Message)]
    pub struct MacroMilestoneAppend {
        #[prost(message, optional, tag = "1")]
        pub milestone: Option<MacroMilestone>,
    }

    #[derive(Clone, PartialEq, Serialize, Deserialize, Message)]
    pub struct CharacterBaselineVector {
        #[prost(int32, tag = "1")]
        pub baseline_caution_offset: i32,
        #[prost(int32, tag = "2")]
        pub baseline_novelty_dampening_offset: i32,
        #[prost(int32, tag = "3")]
        pub baseline_approval_strictness_offset: i32,
        #[prost(int32, tag = "4")]
        pub baseline_export_strictness_offset: i32,
        #[prost(int32, tag = "5")]
        pub baseline_chain_conservatism_offset: i32,
        #[prost(int32, tag = "6")]
        pub baseline_cooldown_multiplier_class: i32,
    }

    #[derive(Clone, PartialEq, Serialize, Deserialize, Message)]
    pub struct PolicyEcologyVector {
        #[prost(int32, tag = "1")]
        pub conservatism_bias: i32,
        #[prost(int32, tag = "2")]
        pub novelty_penalty_bias: i32,
        #[prost(int32, tag = "3")]
        pub manipulation_aversion_bias: i32,
        #[prost(int32, tag = "4")]
        pub reversibility_bias: i32,
    }

    #[derive(Clone, PartialEq, Serialize, Deserialize, Message)]
    pub struct ReceiptStats {
        #[prost(uint32, tag = "1")]
        pub receipt_missing_count: u32,
        #[prost(uint32, tag = "2")]
        pub receipt_invalid_count: u32,
    }

    #[derive(Clone, PartialEq, Serialize, Deserialize, Message)]
    pub struct PolicyStats {
        #[prost(uint32, tag = "1")]
        pub deny_count: u32,
        #[prost(uint32, tag = "2")]
        pub allow_count: u32,
        #[prost(enumeration = "ReasonCode", repeated, tag = "3")]
        pub top_reason_codes: ::prost::alloc::vec::Vec<i32>,
    }

    #[derive(Clone, PartialEq, Serialize, Deserialize, Message)]
    pub struct ExecStats {
        #[prost(uint32, tag = "1")]
        pub timeout_count: u32,
        #[prost(enumeration = "ReasonCode", repeated, tag = "2")]
        pub top_reason_codes: ::prost::alloc::vec::Vec<i32>,
    }

    #[derive(Clone, PartialEq, Serialize, Deserialize, Message)]
    pub struct SignalFrame {
        #[prost(enumeration = "WindowKind", tag = "1")]
        pub window_kind: i32,
        #[prost(uint64, optional, tag = "2")]
        pub window_index: Option<u64>,
        #[prost(uint64, optional, tag = "3")]
        pub timestamp_ms: Option<u64>,
        #[prost(message, optional, tag = "4")]
        pub policy_stats: Option<PolicyStats>,
        #[prost(message, optional, tag = "5")]
        pub exec_stats: Option<ExecStats>,
        #[prost(enumeration = "IntegrityStateClass", tag = "6")]
        pub integrity_state: i32,
        #[prost(enumeration = "ReasonCode", repeated, tag = "7")]
        pub top_reason_codes: ::prost::alloc::vec::Vec<i32>,
        #[prost(bytes, optional, tag = "8")]
        pub signal_frame_digest: Option<::prost::alloc::vec::Vec<u8>>,
        #[prost(message, optional, tag = "9")]
        pub receipt_stats: Option<ReceiptStats>,
        #[prost(enumeration = "ReasonCode", repeated, tag = "10")]
        pub reason_codes: ::prost::alloc::vec::Vec<i32>,
    }

    #[derive(Clone, PartialEq, Serialize, Deserialize, Message)]
    pub struct ActiveProfile {
        #[prost(string, tag = "1")]
        pub profile: ::prost::alloc::string::String,
    }

    #[derive(Clone, PartialEq, Serialize, Deserialize, Message)]
    pub struct Overlays {
        #[prost(bool, tag = "1")]
        pub simulate_first: bool,
        #[prost(bool, tag = "2")]
        pub export_lock: bool,
        #[prost(bool, tag = "3")]
        pub novelty_lock: bool,
        #[prost(bool, tag = "4")]
        pub chain_tightening: bool,
    }

    #[derive(Clone, PartialEq, Serialize, Deserialize, Message)]
    pub struct ToolClassMask {
        #[prost(bool, tag = "1")]
        pub read: bool,
        #[prost(bool, tag = "2")]
        pub write: bool,
        #[prost(bool, tag = "3")]
        pub execute: bool,
        #[prost(bool, tag = "4")]
        pub transform: bool,
        #[prost(bool, tag = "5")]
        pub export: bool,
    }

    #[derive(Clone, PartialEq, Serialize, Deserialize, Message)]
    pub struct ControlFrame {
        #[prost(message, optional, tag = "1")]
        pub active_profile: Option<ActiveProfile>,
        #[prost(message, optional, tag = "2")]
        pub overlays: Option<Overlays>,
        #[prost(message, optional, tag = "3")]
        pub toolclass_mask: Option<ToolClassMask>,
        #[prost(enumeration = "ReasonCode", repeated, tag = "4")]
        pub profile_reason_codes: ::prost::alloc::vec::Vec<i32>,
        #[prost(bytes, optional, tag = "5")]
        pub control_frame_digest: Option<::prost::alloc::vec::Vec<u8>>,
        #[prost(bytes, optional, tag = "6")]
        pub character_epoch_digest: Option<::prost::alloc::vec::Vec<u8>>,
        #[prost(bytes, optional, tag = "7")]
        pub policy_ecology_digest: Option<::prost::alloc::vec::Vec<u8>>,
        #[prost(string, optional, tag = "8")]
        pub approval_mode: Option<::prost::alloc::string::String>,
        #[prost(bool, optional, tag = "9")]
        pub deescalation_lock: Option<bool>,
        #[prost(enumeration = "LevelClass", optional, tag = "10")]
        pub cooldown_class: Option<i32>,
    }
}
