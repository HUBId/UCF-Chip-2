#![cfg(not(feature = "biophys-trace"))]

use pvgs_client::MockPvgsWriter;

#[test]
fn trace_run_commit_skipped_without_feature() {
    let writer = MockPvgsWriter::default();
    assert!(writer.committed_trace_runs.is_empty());
}
