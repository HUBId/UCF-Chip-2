#![cfg(feature = "biophys")]

use biophys_core::{StpParams, StpState, STP_SCALE};

#[test]
fn stp_updates_are_deterministic() {
    let params = StpParams {
        u: 200,
        tau_rec_steps: 2,
        tau_fac_steps: 2,
        mod_channel: None,
    };
    let mut state = StpState {
        x: STP_SCALE,
        u: params.u,
    };

    let released = state.on_spike(params);
    assert_eq!(state.u, 360);
    assert_eq!(released, 360);
    assert_eq!(state.x, 640);

    state.update_between_spikes(params);
    assert_eq!(state.x, 820);
    assert_eq!(state.u, 280);

    state.update_between_spikes(params);
    assert_eq!(state.x, 910);
    assert_eq!(state.u, 240);
}
