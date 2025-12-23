#![forbid(unsafe_code)]

use dbm_core::{EvidenceKind, EvidenceRef};
use engine::{RegulationEngine, RegulationSnapshot};
use hex::encode as hex_encode;
use profiles::ProfileState;
use std::env;
use ucf::v1::{
    ExecStats, IntegrityStateClass, LevelClass, PolicyStats, ReceiptStats, SignalFrame, WindowKind,
};

fn main() {
    let mut args = env::args();
    let _binary = args.next();
    match args.next().as_deref() {
        Some("regulator-dump") => run_regulator_dump(),
        _ => {
            eprintln!("usage: app regulator-dump");
            std::process::exit(1);
        }
    }
}

fn run_regulator_dump() {
    let mut engine = RegulationEngine::default();
    let now_ms = 1;
    let signal_frame = default_ok_signal_frame(now_ms);
    engine
        .enqueue_signal_frame(signal_frame)
        .expect("signal frame enqueued");
    let _ = engine.tick(now_ms);

    let snapshot = engine.snapshot();
    println!("{}", format_snapshot(&snapshot));
}

fn format_snapshot(snapshot: &RegulationSnapshot) -> String {
    let digest = snapshot
        .control_frame_digest
        .map(hex_encode)
        .unwrap_or_else(|| "NONE".to_string());
    let evidence = format_evidence_refs(&snapshot.evidence_refs);

    let lines = [
        format!("profile: {}", format_profile(snapshot.profile)),
        format!(
            "overlays: simulate_first={} export_lock={} novelty_lock={}",
            snapshot.overlays.simulate_first,
            snapshot.overlays.export_lock,
            snapshot.overlays.novelty_lock
        ),
        format!("deescalation_lock: {}", snapshot.deescalation_lock),
        format!("control_frame_digest: {}", digest),
        format!("evidence_refs: {}", evidence),
        format!(
            "rsv: integrity={} threat={} policy_pressure={} arousal={} stability={}",
            format_integrity(snapshot.rsv_summary.integrity),
            format_level(snapshot.rsv_summary.threat),
            format_level(snapshot.rsv_summary.policy_pressure),
            format_level(snapshot.rsv_summary.arousal),
            format_level(snapshot.rsv_summary.stability),
        ),
    ];

    lines.join("\n")
}

fn format_evidence_refs(refs: &[EvidenceRef]) -> String {
    if refs.is_empty() {
        return "NONE".to_string();
    }

    refs.iter()
        .map(|evidence| {
            format!(
                "{}={}",
                format_evidence_kind(evidence.kind),
                hex_encode(evidence.digest)
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_evidence_kind(kind: EvidenceKind) -> &'static str {
    match kind {
        EvidenceKind::LcMicroSnapshot => "mc:lc",
        EvidenceKind::SnMicroSnapshot => "mc:sn",
        EvidenceKind::RulesetDigest => "ruleset",
        EvidenceKind::CbvDigest => "cbv",
        EvidenceKind::PevDigest => "pev",
    }
}

fn format_profile(profile: ProfileState) -> &'static str {
    match profile {
        ProfileState::M0Research => "M0",
        ProfileState::M1Restricted => "M1",
        ProfileState::M2Quarantine => "M2",
        ProfileState::M3Forensic => "M3",
    }
}

fn format_integrity(integrity: IntegrityStateClass) -> &'static str {
    match integrity {
        IntegrityStateClass::Unknown => "UNKNOWN",
        IntegrityStateClass::Ok => "OK",
        IntegrityStateClass::Degraded => "DEGRADED",
        IntegrityStateClass::Fail => "FAIL",
    }
}

fn format_level(level: LevelClass) -> &'static str {
    match level {
        LevelClass::Unknown => "UNKNOWN",
        LevelClass::Low => "LOW",
        LevelClass::Med => "MED",
        LevelClass::High => "HIGH",
    }
}

fn default_ok_signal_frame(now_ms: u64) -> SignalFrame {
    SignalFrame {
        window_kind: WindowKind::Short as i32,
        window_index: Some(1),
        timestamp_ms: Some(now_ms),
        policy_stats: Some(PolicyStats {
            deny_count: 0,
            allow_count: 1,
            top_reason_codes: vec![],
        }),
        exec_stats: Some(ExecStats {
            timeout_count: 0,
            partial_failure_count: 0,
            tool_unavailable_count: 0,
            tool_id: None,
            dlp_block_count: 0,
            top_reason_codes: vec![],
        }),
        integrity_state: IntegrityStateClass::Ok as i32,
        top_reason_codes: vec![],
        signal_frame_digest: None,
        receipt_stats: Some(ReceiptStats {
            receipt_missing_count: 0,
            receipt_invalid_count: 0,
        }),
        reason_codes: vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::RsvSummary;
    use profiles::OverlaySet;

    #[test]
    fn formats_snapshot_output() {
        let snapshot = RegulationSnapshot {
            profile: ProfileState::M2Quarantine,
            overlays: OverlaySet {
                simulate_first: true,
                export_lock: false,
                novelty_lock: true,
                chain_tightening: false,
            },
            deescalation_lock: true,
            control_frame_digest: Some([0xAA; 32]),
            evidence_refs: Vec::new(),
            rsv_summary: RsvSummary {
                integrity: IntegrityStateClass::Degraded,
                threat: LevelClass::Med,
                policy_pressure: LevelClass::High,
                arousal: LevelClass::Low,
                stability: LevelClass::Unknown,
            },
        };

        let expected = "profile: M2\n"
            .to_string()
            + "overlays: simulate_first=true export_lock=false novelty_lock=true\n"
            + "deescalation_lock: true\n"
            + "control_frame_digest: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
            + "evidence_refs: NONE\n"
            + "rsv: integrity=DEGRADED threat=MED policy_pressure=HIGH arousal=LOW stability=UNKNOWN";

        assert_eq!(format_snapshot(&snapshot), expected);
    }
}
