#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct Frame {
    pub channel: String,
    pub payload: Vec<u8>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct SignedFrame {
    pub frame: Frame,
    pub signature: Option<Vec<u8>>,
}

#[derive(Debug, Error)]
pub enum WireError {
    #[error("wire IO is not implemented")]
    IoNotImplemented,
    #[error("frame validation is not implemented")]
    ValidationNotImplemented,
    #[error("frame signing is not implemented")]
    SigningNotImplemented,
}

pub trait FrameIo {
    fn send(&mut self, _frame: SignedFrame) -> Result<(), WireError> {
        Err(WireError::IoNotImplemented)
    }

    fn receive(&mut self) -> Result<Option<SignedFrame>, WireError> {
        Err(WireError::IoNotImplemented)
    }

    fn verify(&self, _frame: &SignedFrame) -> Result<bool, WireError> {
        Err(WireError::ValidationNotImplemented)
    }

    fn sign(&self, _frame: Frame) -> Result<SignedFrame, WireError> {
        Err(WireError::SigningNotImplemented)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct PlaceholderWire;

    impl FrameIo for PlaceholderWire {}

    #[test]
    fn placeholder_wire_rejects_operations() {
        let mut wire = PlaceholderWire;
        let frame = SignedFrame {
            frame: Frame {
                channel: "engine".to_string(),
                payload: vec![0],
            },
            signature: None,
        };

        assert!(matches!(
            wire.send(frame.clone()),
            Err(WireError::IoNotImplemented)
        ));
        assert!(matches!(
            wire.verify(&frame),
            Err(WireError::ValidationNotImplemented)
        ));
    }
}
