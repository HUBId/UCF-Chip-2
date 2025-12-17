#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use thiserror::Error;
use ucf::v1::WindowKind;

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
    #[error("invalid signal frame: {0}")]
    InvalidSignalFrame(String),
}

#[derive(Debug, Default)]
pub struct FrameIngestor;

impl FrameIngestor {
    pub fn new() -> Self {
        Self {}
    }

    pub fn ingest_signal_frame(&mut self, frame: ucf::v1::SignalFrame) -> Result<(), WireError> {
        let window_kind = WindowKind::try_from(frame.window_kind)
            .map_err(|_| WireError::InvalidSignalFrame("window_kind missing".to_string()))?;

        match window_kind {
            WindowKind::Short | WindowKind::Medium => {}
            _ => {
                return Err(WireError::InvalidSignalFrame(
                    "unsupported window kind".to_string(),
                ))
            }
        }

        if frame.window_index.is_none() {
            return Err(WireError::InvalidSignalFrame(
                "missing window index".to_string(),
            ));
        }

        if frame.timestamp_ms.is_none() {
            return Err(WireError::InvalidSignalFrame(
                "missing timestamp_ms".to_string(),
            ));
        }

        if frame.signal_frame_digest.is_some() {
            log::info!("TODO: verify signal_frame_digest");
        }

        Ok(())
    }
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
    use ucf::v1::{IntegrityStateClass, SignalFrame};

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

    #[test]
    fn ingestor_accepts_short_frames() {
        let mut ingestor = FrameIngestor::new();
        let frame = SignalFrame {
            window_kind: WindowKind::Short as i32,
            window_index: Some(1),
            timestamp_ms: Some(42),
            integrity_state: IntegrityStateClass::Ok as i32,
            ..SignalFrame::default()
        };

        assert!(ingestor.ingest_signal_frame(frame).is_ok());
    }

    #[test]
    fn ingestor_rejects_invalid_frames() {
        let mut ingestor = FrameIngestor::new();
        let frame = SignalFrame {
            window_kind: WindowKind::Long as i32,
            integrity_state: IntegrityStateClass::Ok as i32,
            ..SignalFrame::default()
        };

        let result = ingestor.ingest_signal_frame(frame);
        assert!(matches!(result, Err(WireError::InvalidSignalFrame(_))));
    }
}
