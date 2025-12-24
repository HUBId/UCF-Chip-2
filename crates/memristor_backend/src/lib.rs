#![forbid(unsafe_code)]

use blake3::Hasher;

pub const MAX_CELL_VALUE: u16 = 100;

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellKey {
    AL,
    EXFIL_SENS,
    INTEGRITY_SENS,
    PROBING_SENS,
    RELIABILITY_SENS,
    RECOVERY_CONF,
}

impl CellKey {
    pub const ALL: [CellKey; 6] = [
        CellKey::AL,
        CellKey::EXFIL_SENS,
        CellKey::INTEGRITY_SENS,
        CellKey::PROBING_SENS,
        CellKey::RELIABILITY_SENS,
        CellKey::RECOVERY_CONF,
    ];

    pub fn index(self) -> usize {
        match self {
            CellKey::AL => 0,
            CellKey::EXFIL_SENS => 1,
            CellKey::INTEGRITY_SENS => 2,
            CellKey::PROBING_SENS => 3,
            CellKey::RELIABILITY_SENS => 4,
            CellKey::RECOVERY_CONF => 5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CalibrationReport {
    pub before_digest: [u8; 32],
    pub after_digest: [u8; 32],
}

pub trait MemristorBackend {
    fn read_cell(&self, key: CellKey) -> u16;

    fn write_cell(&mut self, key: CellKey, value: u16);

    fn calibrate(&mut self) -> CalibrationReport;

    fn backend_digest(&self) -> [u8; 32];
}

#[derive(Debug, Clone)]
pub struct EmulatedMemristorBackend {
    cells: [u16; 6],
    drift_bias: [i16; 6],
    seed: u64,
}

impl EmulatedMemristorBackend {
    pub fn new(seed: u64) -> Self {
        let drift_bias = derive_drift_bias(seed);
        Self {
            cells: [0; 6],
            drift_bias,
            seed,
        }
    }

    fn clamp(value: u16) -> u16 {
        value.min(MAX_CELL_VALUE)
    }

    fn digest_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend(self.seed.to_le_bytes());
        for value in self.cells.iter() {
            bytes.extend(value.to_le_bytes());
        }
        for value in self.drift_bias.iter() {
            bytes.extend(value.to_le_bytes());
        }
        bytes
    }
}

impl Default for EmulatedMemristorBackend {
    fn default() -> Self {
        Self::new(0)
    }
}

impl MemristorBackend for EmulatedMemristorBackend {
    fn read_cell(&self, key: CellKey) -> u16 {
        self.cells[key.index()]
    }

    fn write_cell(&mut self, key: CellKey, value: u16) {
        self.cells[key.index()] = Self::clamp(value);
    }

    fn calibrate(&mut self) -> CalibrationReport {
        let before_digest = self.backend_digest();
        for (idx, bias) in self.drift_bias.iter().enumerate() {
            let bias = (*bias).max(0) as u16;
            self.cells[idx] = self.cells[idx].saturating_sub(bias);
        }
        let after_digest = self.backend_digest();
        CalibrationReport {
            before_digest,
            after_digest,
        }
    }

    fn backend_digest(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(b"memristor-backend");
        hasher.update(&self.digest_bytes());
        *hasher.finalize().as_bytes()
    }
}

#[cfg(feature = "microcircuit-hpa-hw")]
#[derive(Debug, Default, Clone, Copy)]
pub struct HardwareMemristorBackend;

#[cfg(feature = "microcircuit-hpa-hw")]
impl MemristorBackend for HardwareMemristorBackend {
    fn read_cell(&self, _key: CellKey) -> u16 {
        unimplemented!("Hardware memristor backend is not implemented yet")
    }

    fn write_cell(&mut self, _key: CellKey, _value: u16) {
        unimplemented!("Hardware memristor backend is not implemented yet")
    }

    fn calibrate(&mut self) -> CalibrationReport {
        unimplemented!("Hardware memristor backend is not implemented yet")
    }

    fn backend_digest(&self) -> [u8; 32] {
        unimplemented!("Hardware memristor backend is not implemented yet")
    }
}

fn derive_drift_bias(seed: u64) -> [i16; 6] {
    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
    let mut bias = [0i16; 6];
    for slot in bias.iter_mut() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let magnitude = ((state >> 32) % 3) + 1;
        *slot = magnitude as i16;
    }
    bias
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_cell_clamps_values() {
        let mut backend = EmulatedMemristorBackend::default();
        backend.write_cell(CellKey::AL, 999);
        assert_eq!(backend.read_cell(CellKey::AL), MAX_CELL_VALUE);
    }

    #[test]
    fn calibrate_is_deterministic() {
        let mut backend = EmulatedMemristorBackend::new(42);
        backend.write_cell(CellKey::AL, 20);
        let report_a = backend.calibrate();
        let report_b = backend.calibrate();
        assert_ne!(report_a.before_digest, report_a.after_digest);
        assert_ne!(report_a.after_digest, report_b.after_digest);
    }

    #[test]
    fn backend_digest_depends_on_state() {
        let mut backend = EmulatedMemristorBackend::new(7);
        let digest_a = backend.backend_digest();
        backend.write_cell(CellKey::EXFIL_SENS, 10);
        let digest_b = backend.backend_digest();
        assert_ne!(digest_a, digest_b);
    }
}
