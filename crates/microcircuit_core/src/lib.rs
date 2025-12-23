#![forbid(unsafe_code)]

use blake3::Hasher;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CircuitConfig {
    pub version: u32,
    pub seed: u64,
    pub max_neurons: u32,
}

impl Default for CircuitConfig {
    fn default() -> Self {
        Self {
            version: 1,
            seed: 0,
            max_neurons: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CircuitStateMeta {
    pub last_step_ms: u64,
    pub step_count: u64,
}

pub trait MicrocircuitBackend<I, O> {
    fn step(&mut self, input: &I, now_ms: u64) -> O;

    fn snapshot_digest(&self) -> [u8; 32];
}

pub fn digest_meta(domain: &str, bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(domain.as_bytes());
    hasher.update(bytes);
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_depends_on_domain() {
        let a = digest_meta("alpha", b"payload");
        let b = digest_meta("beta", b"payload");
        assert_ne!(a, b);
    }

    #[test]
    fn digest_depends_on_bytes() {
        let a = digest_meta("alpha", b"payload");
        let b = digest_meta("alpha", b"payload2");
        assert_ne!(a, b);
    }
}
