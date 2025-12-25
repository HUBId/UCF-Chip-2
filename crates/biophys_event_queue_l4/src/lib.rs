#![forbid(unsafe_code)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpikeEventL4 {
    pub deliver_step: u64,
    pub synapse_index: usize,
    pub release_gain_q: u16,
}

#[derive(Debug, Clone)]
pub struct SpikeEventQueueL4 {
    buckets: Vec<Vec<SpikeEventL4>>,
    max_events_per_step: usize,
    pub dropped_event_count: u64,
}

impl SpikeEventQueueL4 {
    pub fn new(max_delay_steps: u16, max_events_per_step: usize) -> Self {
        let len = max_delay_steps as usize + 1;
        Self {
            buckets: vec![Vec::new(); len.max(1)],
            max_events_per_step,
            dropped_event_count: 0,
        }
    }

    pub fn schedule_spike<F, G>(
        &mut self,
        current_step: u64,
        synapse_indices: &[usize],
        delay_steps_for: F,
        mut release_gain_for: G,
    ) where
        F: Fn(usize) -> u16,
        G: FnMut(usize) -> u16,
    {
        if synapse_indices.is_empty() {
            return;
        }
        let mut sorted = synapse_indices.to_vec();
        sorted.sort_unstable();
        for synapse_index in sorted {
            let deliver_step = current_step.saturating_add(delay_steps_for(synapse_index) as u64);
            let bucket = (deliver_step as usize) % self.buckets.len();
            if self.buckets[bucket].len() >= self.max_events_per_step {
                self.dropped_event_count = self.dropped_event_count.saturating_add(1);
                continue;
            }
            self.buckets[bucket].push(SpikeEventL4 {
                deliver_step,
                synapse_index,
                release_gain_q: release_gain_for(synapse_index),
            });
        }
    }

    pub fn drain_current(&mut self, current_step: u64) -> Vec<SpikeEventL4> {
        let bucket = (current_step as usize) % self.buckets.len();
        let mut events = std::mem::take(&mut self.buckets[bucket]);
        let mut current = Vec::new();
        for event in events.drain(..) {
            if event.deliver_step == current_step {
                current.push(event);
            } else {
                self.buckets[bucket].push(event);
            }
        }
        current
    }
}
