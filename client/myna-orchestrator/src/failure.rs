//! What a failed utterance tells the user, split from what it tells a log.
//!
//! A [`Failure`] carries a short translated `headline` for the HUD, the
//! screen reader and the notification, and an untranslated `detail` (the raw
//! cause, which may name paths or commands) for logs and Settings
//! Diagnostics only.

use myna_core::CaptureError;

use crate::i18n::tr;

/// A failure as the user is told it (`headline`) and as a log records it
/// (`detail`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Failure {
    pub headline: String,
    pub detail: String,
}

impl Failure {
    pub fn new(headline: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            headline: headline.into(),
            detail: detail.into(),
        }
    }

    /// A capture fault, headlined by what it means for the microphone.
    pub fn capture(fault: &CaptureError) -> Self {
        Self::new(capture_headline(fault), fault.to_string())
    }

    /// A capture fault an utterance was salvaged from: the audio after it is
    /// gone, which is the news, whatever the device did.
    pub fn audio_lost(fault: &CaptureError) -> Self {
        Self::new(
            tr("Some audio lost"),
            format!("some audio was lost: {fault}"),
        )
    }
}

/// The headline for a capture fault.
pub fn capture_headline(fault: &CaptureError) -> String {
    match fault {
        CaptureError::DeviceUnavailable(_) => tr("Microphone unavailable"),
        CaptureError::ServiceUnresponsive(_) => tr("Audio system not responding"),
        CaptureError::NoSource(_) => tr("No microphone found"),
        CaptureError::NoFlow(_) => tr("Microphone silent"),
        CaptureError::UnsupportedFormat(_) => tr("Microphone format unsupported"),
        CaptureError::Backend(_) => tr("Audio system error"),
        CaptureError::Overloaded(_) => tr("System overloaded"),
    }
}

/// The headline for a backend `error` event; its message is the server's own.
pub fn transcription_failed() -> String {
    tr("Transcription failed")
}

/// The headline for a backend that made no progress within the deadline.
pub fn model_not_responding() -> String {
    tr("Model not responding")
}

/// The headline for a session whose connection dropped before it completed.
pub fn model_connection_lost() -> String {
    tr("Model connection lost")
}

/// The headline for a backend warning: worded here for the conditions the wire
/// names, the server's own text for any other.
pub fn warning_headline(code: Option<&str>, text: &str) -> String {
    match code {
        Some(myna_core::WARNING_GPU_UNAVAILABLE) => tr("Using the CPU: GPU unavailable"),
        Some(myna_core::WARNING_MEMORY_PRESSURE) => tr("Dictation is slow: low memory"),
        _ => text.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use myna_core::AudioFormat;

    #[test]
    fn every_capture_fault_has_its_headline() {
        let cases = [
            (
                CaptureError::DeviceUnavailable("gone".into()),
                "Microphone unavailable",
            ),
            (
                CaptureError::UnsupportedFormat(AudioFormat::default()),
                "Microphone format unsupported",
            ),
            (
                CaptureError::ServiceUnresponsive("x".into()),
                "Audio system not responding",
            ),
            (CaptureError::NoSource("x".into()), "No microphone found"),
            (CaptureError::NoFlow("x".into()), "Microphone silent"),
            (CaptureError::Backend("x".into()), "Audio system error"),
            (CaptureError::Overloaded(2.5), "System overloaded"),
        ];
        for (fault, headline) in cases {
            let failure = Failure::capture(&fault);
            assert_eq!(failure.headline, headline);
            assert_eq!(failure.detail, fault.to_string());
        }
    }

    #[test]
    fn a_salvage_is_headlined_as_lost_audio_and_keeps_the_cause() {
        let failure = Failure::audio_lost(&CaptureError::Overloaded(1.5));
        assert_eq!(failure.headline, "Some audio lost");
        assert!(
            failure
                .detail
                .starts_with("some audio was lost: audio buffer overflow after 1.5s"),
            "{}",
            failure.detail
        );
    }

    /// Headlines are for people, not terminals: no commands, no code
    /// formatting, no internal vocabulary, no em dash.
    #[test]
    fn no_headline_speaks_cli() {
        use crate::backend::share::{ResolveError, Unusable};
        use crate::backend::BackendError;
        let unusable = |no_unix_socket: &[&str], not_serving: &[&str], malformed| {
            ResolveError::NotConnected(Unusable {
                no_unix_socket: no_unix_socket.iter().map(|s| s.to_string()).collect(),
                not_serving: not_serving.iter().map(|s| s.to_string()).collect(),
                malformed,
                unmounted: 0,
            })
        };
        let mut headlines = vec![
            transcription_failed(),
            model_not_responding(),
            model_connection_lost(),
            warning_headline(Some(myna_core::WARNING_GPU_UNAVAILABLE), ""),
            warning_headline(Some(myna_core::WARNING_MEMORY_PRESSURE), ""),
            Failure::audio_lost(&CaptureError::Overloaded(1.0)).headline,
        ];
        for fault in [
            CaptureError::DeviceUnavailable("x".into()),
            CaptureError::UnsupportedFormat(AudioFormat::default()),
            CaptureError::ServiceUnresponsive("x".into()),
            CaptureError::NoSource("x".into()),
            CaptureError::NoFlow("x".into()),
            CaptureError::Backend("x".into()),
            CaptureError::Overloaded(1.0),
        ] {
            headlines.push(capture_headline(&fault));
        }
        for resolve in [
            unusable(&[], &[], 0),
            unusable(&["a"], &[], 0),
            unusable(&[], &["b"], 0),
            unusable(&[], &[], 1),
            ResolveError::NotConnected(Unusable {
                unmounted: 1,
                ..Unusable::default()
            }),
            ResolveError::Ambiguous(vec!["a".into(), "b".into()]),
        ] {
            headlines.push(resolve.headline());
            headlines.push(BackendError::Resolve(resolve).headline());
        }
        for error in [
            BackendError::Connect("x".into()),
            BackendError::Handshake("x".into()),
            BackendError::Rejected {
                code: "unsupported_protocol_version".into(),
                message: "x".into(),
            },
            BackendError::Rejected {
                code: "x".into(),
                message: "x".into(),
            },
            BackendError::Wire(myna_core::WireError::NotAnEvent),
            BackendError::Closed,
            BackendError::Transport("x".into()),
        ] {
            headlines.push(error.headline());
        }
        for headline in headlines {
            assert!(!headline.is_empty());
            for banned in ["snap ", "`", "backend", "Backend", "\u{2014}"] {
                assert!(!headline.contains(banned), "{headline:?} has {banned:?}");
            }
        }
    }
}
