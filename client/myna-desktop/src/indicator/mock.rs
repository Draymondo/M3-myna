//! `MockIndicator` — the hermetic activity-indicator fixture (T008).
//!
//! Records the `IndicatorState` sequence so controller tests can
//! assert the state timeline (Recording→Transcribing→Finalizing→Hidden, error
//! states) with no GTK and no display.
//! (contract indicator.md N-mapping tests).

use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use super::{Indicator, IndicatorState, InputLevels};

/// A hermetic [`Indicator`] that records every state it is shown. Clone the log
/// handle (`.log()`) before moving the mock into the controller.
#[derive(Default)]
pub struct MockIndicator {
    log: Arc<Mutex<Vec<IndicatorState>>>,
    last_errors: Arc<Mutex<Vec<(String, String)>>>,
    input_levels: Arc<Mutex<Vec<InputLevels>>>,
}

impl MockIndicator {
    pub fn new() -> Self {
        Self::default()
    }

    /// A shared handle to the recorded state sequence.
    pub fn log(&self) -> Arc<Mutex<Vec<IndicatorState>>> {
        self.log.clone()
    }

    /// A shared handle to every `(headline, detail)` recorded as the last
    /// error, in order.
    pub fn last_errors(&self) -> Arc<Mutex<Vec<(String, String)>>> {
        self.last_errors.clone()
    }

    /// A shared handle to every session's published input levels, in order.
    pub fn input_levels(&self) -> Arc<Mutex<Vec<InputLevels>>> {
        self.input_levels.clone()
    }
}

#[async_trait]
impl Indicator for MockIndicator {
    async fn set_state(&mut self, state: IndicatorState) {
        self.log.lock().unwrap().push(state);
    }

    async fn set_last_error(&mut self, headline: &str, detail: &str) {
        self.last_errors
            .lock()
            .unwrap()
            .push((headline.to_string(), detail.to_string()));
    }

    async fn set_input_levels(&mut self, levels: InputLevels) {
        self.input_levels.lock().unwrap().push(levels);
    }
}
