#![forbid(unsafe_code)]

use std::collections::HashSet;

use ucf::v1::{Digest32, MicrocircuitConfigEvidence, Ref, ReplayRunEvidence};

const RUN_ID_DIGEST_PREFIX_LEN: usize = 8;
const MAX_MICRO_CONFIGS: usize = 8;

#[derive(Debug, Clone)]
pub struct ReplayRunEvidenceInput<'a> {
    pub session_id: &'a str,
    pub replay_id: &'a str,
    pub now_ms: u64,
    pub run_digest: [u8; 32],
    pub replay_plan_digest: [u8; 32],
    pub asset_manifest_digest: [u8; 32],
    pub steps: u32,
    pub dt_us: u32,
    pub substeps_per_tick: u32,
    pub summary_profile_seq_digest: [u8; 32],
    pub summary_dwm_seq_digest: [u8; 32],
    pub micro_config_evidence: &'a [MicrocircuitConfigEvidence],
    pub modules_used: &'a [i32],
}

pub fn build_replay_run_evidence(input: ReplayRunEvidenceInput<'_>) -> ReplayRunEvidence {
    let run_id = build_run_id(input.session_id, input.replay_id, input.run_digest);
    let replay_plan_ref = ref_from_digest("replay_plan", input.replay_plan_digest);
    let asset_manifest_ref = ref_from_digest("asset_manifest", input.asset_manifest_digest);
    let micro_configs = select_micro_configs(input.micro_config_evidence, input.modules_used);

    ReplayRunEvidence {
        run_id,
        run_digest: Some(Digest32 {
            value: input.run_digest.to_vec(),
        }),
        replay_plan_ref: Some(replay_plan_ref),
        asset_manifest_ref: Some(asset_manifest_ref),
        micro_configs,
        steps: input.steps,
        dt_us: input.dt_us,
        substeps_per_tick: input.substeps_per_tick,
        summary_profile_seq_digest: Some(Digest32 {
            value: input.summary_profile_seq_digest.to_vec(),
        }),
        summary_dwm_seq_digest: Some(Digest32 {
            value: input.summary_dwm_seq_digest.to_vec(),
        }),
        created_at_ms: input.now_ms,
        proof_receipt_ref: None,
        signatures: Vec::new(),
    }
}

fn build_run_id(session_id: &str, replay_id: &str, run_digest: [u8; 32]) -> String {
    let mut digest_hex = hex::encode(run_digest);
    if digest_hex.len() > RUN_ID_DIGEST_PREFIX_LEN {
        digest_hex.truncate(RUN_ID_DIGEST_PREFIX_LEN);
    }
    format!("replay_run:{session_id}:{replay_id}:{digest_hex}")
}

fn ref_from_digest(id: &str, digest: [u8; 32]) -> Ref {
    Ref {
        id: id.to_string(),
        digest: digest.to_vec(),
    }
}

fn select_micro_configs(
    evidence: &[MicrocircuitConfigEvidence],
    modules_used: &[i32],
) -> Vec<MicrocircuitConfigEvidence> {
    if evidence.is_empty() {
        return Vec::new();
    }

    let mut allowed = HashSet::new();
    for module in modules_used {
        allowed.insert(*module);
    }

    let mut selected: Vec<MicrocircuitConfigEvidence> = evidence
        .iter()
        .filter(|ev| allowed.contains(&ev.module))
        .cloned()
        .collect();
    selected.sort_by(|a, b| a.module.cmp(&b.module));
    if selected.len() > MAX_MICRO_CONFIGS {
        selected.truncate(MAX_MICRO_CONFIGS);
    }
    selected
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;
    use ucf::v1::MicrocircuitModule;

    fn micro_config(module: i32, created_at_ms: u64) -> MicrocircuitConfigEvidence {
        MicrocircuitConfigEvidence {
            module,
            config_version: 1,
            config_digest: vec![module as u8; 32],
            created_at_ms,
            prev_config_digest: None,
        }
    }

    #[test]
    fn replay_run_evidence_is_deterministic() {
        let input = ReplayRunEvidenceInput {
            session_id: "session-1",
            replay_id: "replay-a",
            now_ms: 55,
            run_digest: [3u8; 32],
            replay_plan_digest: [1u8; 32],
            asset_manifest_digest: [2u8; 32],
            steps: 10,
            dt_us: 250,
            substeps_per_tick: 4,
            summary_profile_seq_digest: [4u8; 32],
            summary_dwm_seq_digest: [5u8; 32],
            micro_config_evidence: &[
                micro_config(MicrocircuitModule::Hpa as i32, 1),
                micro_config(MicrocircuitModule::Sn as i32, 2),
            ],
            modules_used: &[
                MicrocircuitModule::Sn as i32,
                MicrocircuitModule::Hpa as i32,
            ],
        };

        let evidence_a = build_replay_run_evidence(input.clone());
        let evidence_b = build_replay_run_evidence(input);

        assert_eq!(evidence_a.run_id, evidence_b.run_id);
        assert_eq!(evidence_a.encode_to_vec(), evidence_b.encode_to_vec());
    }

    #[test]
    fn micro_configs_sorted_by_module() {
        let input = ReplayRunEvidenceInput {
            session_id: "session-2",
            replay_id: "replay-b",
            now_ms: 60,
            run_digest: [6u8; 32],
            replay_plan_digest: [7u8; 32],
            asset_manifest_digest: [8u8; 32],
            steps: 2,
            dt_us: 100,
            substeps_per_tick: 1,
            summary_profile_seq_digest: [9u8; 32],
            summary_dwm_seq_digest: [10u8; 32],
            micro_config_evidence: &[
                micro_config(MicrocircuitModule::Hpa as i32, 1),
                micro_config(MicrocircuitModule::Lc as i32, 2),
                micro_config(MicrocircuitModule::Sn as i32, 3),
            ],
            modules_used: &[
                MicrocircuitModule::Sn as i32,
                MicrocircuitModule::Lc as i32,
                MicrocircuitModule::Hpa as i32,
            ],
        };

        let evidence = build_replay_run_evidence(input);
        let modules: Vec<i32> = evidence.micro_configs.iter().map(|ev| ev.module).collect();

        assert_eq!(
            modules,
            vec![
                MicrocircuitModule::Lc as i32,
                MicrocircuitModule::Sn as i32,
                MicrocircuitModule::Hpa as i32
            ]
        );
    }

    #[test]
    fn micro_configs_are_bounded() {
        let modules_used: Vec<i32> = (1..=12).collect();
        let evidence_list: Vec<MicrocircuitConfigEvidence> = (1..=12)
            .map(|module| micro_config(module, module as u64))
            .collect();

        let input = ReplayRunEvidenceInput {
            session_id: "session-3",
            replay_id: "replay-c",
            now_ms: 70,
            run_digest: [11u8; 32],
            replay_plan_digest: [12u8; 32],
            asset_manifest_digest: [13u8; 32],
            steps: 1,
            dt_us: 50,
            substeps_per_tick: 2,
            summary_profile_seq_digest: [14u8; 32],
            summary_dwm_seq_digest: [15u8; 32],
            micro_config_evidence: &evidence_list,
            modules_used: &modules_used,
        };

        let evidence = build_replay_run_evidence(input);
        assert_eq!(evidence.micro_configs.len(), 8);
        let modules: Vec<i32> = evidence.micro_configs.iter().map(|ev| ev.module).collect();
        assert_eq!(modules, (1..=8).collect::<Vec<i32>>());
    }
}
