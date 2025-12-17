#![forbid(unsafe_code)]

use rsv::OverlayId;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct ProfileDefinition {
    pub name: String,
    pub overlays: Vec<OverlayDefinition>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct OverlayDefinition {
    pub name: OverlayId,
    pub description: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct ProfileResolutionRequest {
    pub profile: String,
    pub overlays: Vec<OverlayId>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct ProfileResolution {
    pub active_profile: String,
    pub active_overlays: Vec<OverlayId>,
}

#[derive(Debug, Error)]
pub enum ProfileError {
    #[error("profile composition is not implemented")]
    NotImplemented,
}

pub trait ProfileComposer {
    fn compose(
        &self,
        _request: ProfileResolutionRequest,
    ) -> Result<ProfileResolution, ProfileError> {
        Err(ProfileError::NotImplemented)
    }
}

#[derive(Debug, Default)]
pub struct PlaceholderComposer;

impl ProfileComposer for PlaceholderComposer {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composer_is_a_placeholder() {
        let composer = PlaceholderComposer;
        let request = ProfileResolutionRequest {
            profile: "baseline".to_string(),
            overlays: vec!["overlay".to_string()],
        };

        let result = composer.compose(request);
        assert!(matches!(result, Err(ProfileError::NotImplemented)));
    }
}
