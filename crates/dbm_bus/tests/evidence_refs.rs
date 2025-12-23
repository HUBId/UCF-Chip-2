use dbm_bus::{BrainBus, BrainInput};
#[test]
fn rules_backend_has_no_evidence_refs() {
    let mut bus = BrainBus::default();
    let output = bus.tick(BrainInput::default());

    assert!(output.evidence_refs.is_empty());
}

#[cfg(all(
    feature = "microcircuit-lc-spike",
    feature = "microcircuit-sn-attractor"
))]
mod microcircuit {
    use super::*;
    use dbm_0_sn::SubstantiaNigra;
    use dbm_7_lc::Lc;
    use dbm_core::EvidenceKind;
    use dbm_core::limits::MAX_EVIDENCE_REFS;
    use microcircuit_core::CircuitConfig;

    fn micro_bus() -> BrainBus {
        let mut bus = BrainBus::default();
        let config = CircuitConfig::default();
        *bus.lc_mut() = Lc::new_micro(config);
        *bus.sn_mut() = SubstantiaNigra::new_micro(config);
        bus
    }

    #[test]
    fn evidence_refs_present_in_fixed_order() {
        let mut bus = micro_bus();
        let output = bus.tick(BrainInput::default());

        assert_eq!(output.evidence_refs.len(), 2);
        assert_eq!(output.evidence_refs[0].kind, EvidenceKind::LcMicroSnapshot);
        assert_eq!(output.evidence_refs[1].kind, EvidenceKind::SnMicroSnapshot);
        assert!(output.evidence_refs.len() <= MAX_EVIDENCE_REFS);
    }

    #[test]
    fn evidence_refs_are_deterministic() {
        let mut bus_a = micro_bus();
        let mut bus_b = micro_bus();
        let input = BrainInput::default();

        let output_a = bus_a.tick(input.clone());
        let output_b = bus_b.tick(input);

        assert_eq!(output_a.evidence_refs, output_b.evidence_refs);
    }
}
