#![forbid(unsafe_code)]

pub mod v1 {
    use prost::Enumeration;
    use prost::Message;
    use serde::{Deserialize, Serialize};

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
        ThExfilHighConfidence = 10,
        ThPolicyProbing = 11,
        RcGvCbvUpdated = 12,
        RcCdDlpSecretPattern = 20,
        RcCdDlpObfuscation = 21,
        RcCdDlpStegano = 22,
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
    }

    #[derive(Clone, PartialEq, Serialize, Deserialize, Message)]
    pub struct ExecStats {
        #[prost(uint32, tag = "1")]
        pub timeout_count: u32,
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
