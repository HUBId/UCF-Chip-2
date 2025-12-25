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
    use dbm_core::limits::MAX_EVIDENCE_REFS;
    use dbm_core::EvidenceKind;
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

#[cfg(feature = "biophys-l4-plasticity")]
mod plasticity_evidence {
    use super::*;
    use dbm_0_sn::SubstantiaNigra;
    use dbm_6_dopamin_nacc::DopaInput;
    use dbm_core::limits::MAX_EVIDENCE_REFS;
    use dbm_core::EvidenceKind;
    use microcircuit_core::CircuitConfig;

    fn micro_bus() -> BrainBus {
        let mut bus = BrainBus::default();
        let config = CircuitConfig::default();
        *bus.sn_mut() = SubstantiaNigra::new_micro(config);
        bus
    }

    fn tick_with_dopa(bus: &mut BrainBus, dopa_input: DopaInput) -> dbm_bus::BrainOutput {
        bus.tick(BrainInput {
            dopamin: Some(dopa_input),
            ..BrainInput::default()
        })
    }

    fn replay_high_output(bus: &mut BrainBus) -> dbm_bus::BrainOutput {
        tick_with_dopa(
            bus,
            DopaInput {
                exec_success_count_medium: 5,
                ..Default::default()
            },
        );
        for _ in 0..2 {
            tick_with_dopa(bus, DopaInput::default());
        }
        tick_with_dopa(bus, DopaInput::default())
    }

    #[test]
    fn no_plasticity_when_replay_disabled() {
        let mut bus = micro_bus();
        let output = tick_with_dopa(
            &mut bus,
            DopaInput {
                exec_success_count_medium: 5,
                ..Default::default()
            },
        );

        assert!(!output
            .evidence_refs
            .iter()
            .any(|evidence| evidence.kind == EvidenceKind::PlasticitySnapshot));
    }

    #[test]
    fn no_plasticity_when_da_low() {
        let mut bus = micro_bus();
        tick_with_dopa(&mut bus, DopaInput::default());
        tick_with_dopa(&mut bus, DopaInput::default());
        let output = tick_with_dopa(&mut bus, DopaInput::default());

        assert!(!output
            .evidence_refs
            .iter()
            .any(|evidence| evidence.kind == EvidenceKind::PlasticitySnapshot));
    }

    #[test]
    fn plasticity_emitted_when_replay_and_da_high() {
        let mut bus = micro_bus();
        let output = replay_high_output(&mut bus);
        let kinds: Vec<_> = output.evidence_refs.iter().map(|item| item.kind).collect();

        assert!(kinds.contains(&EvidenceKind::SnMicroSnapshot));
        assert!(kinds.contains(&EvidenceKind::PlasticitySnapshot));
        let sn_pos = kinds
            .iter()
            .position(|kind| *kind == EvidenceKind::SnMicroSnapshot)
            .expect("sn snapshot evidence");
        let plasticity_pos = kinds
            .iter()
            .position(|kind| *kind == EvidenceKind::PlasticitySnapshot)
            .expect("plasticity evidence");
        assert!(sn_pos < plasticity_pos);
        assert!(output.evidence_refs.len() <= MAX_EVIDENCE_REFS);
    }

    #[test]
    fn plasticity_evidence_deterministic() {
        let mut bus_a = micro_bus();
        let mut bus_b = micro_bus();

        let output_a = replay_high_output(&mut bus_a);
        let output_b = replay_high_output(&mut bus_b);

        assert_eq!(output_a.evidence_refs, output_b.evidence_refs);
    }
}
