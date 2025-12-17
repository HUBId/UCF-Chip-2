#![forbid(unsafe_code)]

use profiles::{ProfileComposer, ProfileResolution};
use rsv::{RegulatorState, StateStore};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use wire::SignedFrame;

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct EngineInputs {
    pub tick: u64,
    pub inbound: Vec<SignedFrame>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EngineOutcome {
    pub state: RegulatorState,
    pub outbound: Vec<SignedFrame>,
    pub resolution: ProfileResolution,
}

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("engine update is not implemented")]
    NotImplemented,
}

pub trait UpdateEngine {
    type Store: StateStore;
    type Composer: ProfileComposer;

    fn store(&self) -> &Self::Store;
    fn composer(&self) -> &Self::Composer;

    fn apply(
        &mut self,
        _state: &mut RegulatorState,
        _inputs: EngineInputs,
    ) -> Result<EngineOutcome, EngineError> {
        Err(EngineError::NotImplemented)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use profiles::{PlaceholderComposer, ProfileResolutionRequest};
    use rsv::{HealthFlag, StateError};

    #[derive(Debug)]
    struct MemoryStore;

    impl StateStore for MemoryStore {
        fn load(&self) -> Result<RegulatorState, StateError> {
            Ok(RegulatorState::default())
        }

        fn persist(&self, _state: &RegulatorState) -> Result<(), StateError> {
            Ok(())
        }
    }

    struct PlaceholderEngine {
        store: MemoryStore,
        composer: PlaceholderComposer,
    }

    impl UpdateEngine for PlaceholderEngine {
        type Store = MemoryStore;
        type Composer = PlaceholderComposer;

        fn store(&self) -> &Self::Store {
            &self.store
        }

        fn composer(&self) -> &Self::Composer {
            &self.composer
        }
    }

    #[test]
    fn engine_trait_defaults_to_not_implemented() {
        let mut engine = PlaceholderEngine {
            store: MemoryStore,
            composer: PlaceholderComposer,
        };

        let mut state = RegulatorState {
            profile: "baseline".to_string(),
            active_overlays: vec!["overlay-a".to_string()],
            window_index: 0,
            health: HealthFlag::Nominal,
        };

        let result = engine.apply(
            &mut state,
            EngineInputs {
                tick: 0,
                inbound: Vec::new(),
            },
        );

        assert!(matches!(result, Err(EngineError::NotImplemented)));
        let request = ProfileResolutionRequest {
            profile: state.profile.clone(),
            overlays: state.active_overlays.clone(),
        };
        assert!(matches!(
            engine.composer.compose(request),
            Err(profiles::ProfileError::NotImplemented)
        ));
    }
}
