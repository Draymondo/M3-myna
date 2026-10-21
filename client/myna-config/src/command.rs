use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::future::poll_fn;
use std::io;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::task::{Poll, Waker};
use std::time::Duration;

use async_trait::async_trait;
use gio::{glib, SubprocessFlags, SubprocessLauncher};
use glib::translate::*;
use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommandRequest {
    executable: String,
    arguments: Vec<String>,
    environment: BTreeMap<String, String>,
    removed: BTreeSet<String>,
    timeout: Option<Duration>,
}

impl CommandRequest {
    pub fn new(executable: String, arguments: Vec<String>) -> Self {
        Self {
            executable,
            arguments,
            environment: BTreeMap::new(),
            removed: BTreeSet::new(),
            timeout: Some(Duration::from_secs(30)),
        }
    }

    pub fn with_environment(mut self, environment: BTreeMap<String, String>) -> Self {
        self.environment = environment;
        self
    }

    /// Keep `name` out of the environment the process inherits from this one.
    pub fn without_inherited(mut self, name: &str) -> Self {
        self.removed.insert(name.to_owned());
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Run until the process exits or is cancelled, however long that takes.
    pub fn without_timeout(mut self) -> Self {
        self.timeout = None;
        self
    }

    pub fn timeout(&self) -> Option<Duration> {
        self.timeout
    }

    pub fn executable(&self) -> &str {
        &self.executable
    }

    pub fn arguments(&self) -> &[String] {
        &self.arguments
    }

    pub fn environment(&self) -> &BTreeMap<String, String> {
        &self.environment
    }

    pub fn removed_environment(&self) -> &BTreeSet<String> {
        &self.removed
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommandOutput {
    exit_status: Option<i32>,
    stdout: String,
    stderr: String,
}

impl CommandOutput {
    pub fn new(
        exit_status: Option<i32>,
        stdout: impl Into<String>,
        stderr: impl Into<String>,
    ) -> Self {
        Self {
            exit_status,
            stdout: stdout.into(),
            stderr: stderr.into(),
        }
    }

    pub fn exit_status(&self) -> Option<i32> {
        self.exit_status
    }

    pub fn stdout(&self) -> &str {
        &self.stdout
    }

    pub fn stderr(&self) -> &str {
        &self.stderr
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputStream {
    Stdout,
    Stderr,
}

impl std::fmt::Display for OutputStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Stdout => "stdout",
            Self::Stderr => "stderr",
        })
    }
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum CommandError {
    #[error("command not found: {executable}")]
    NotFound { executable: String },
    #[error("could not spawn {executable}: {message}")]
    Spawn {
        executable: String,
        kind: io::ErrorKind,
        message: String,
    },
    #[error("command timed out after {}", duration_text(*timeout))]
    Timeout { timeout: Duration },
    #[error("command was cancelled")]
    Cancelled,
    #[error("{}", non_zero_text(*exit_status, stdout, stderr))]
    NonZero {
        exit_status: Option<i32>,
        stdout: String,
        stderr: String,
    },
    #[error("{stream} was not valid UTF-8: {message}")]
    InvalidUtf8 {
        stream: OutputStream,
        message: String,
    },
    #[error("fake command runner has no scripted outcome")]
    FakeScriptExhausted,
}

/// Longest stretch of a failed command's output a one-line message quotes.
const QUOTED_OUTPUT_CHARS: usize = 300;

/// How the command ended, then the last line it printed, which names the
/// cause: a traceback or a log ends in it.
fn non_zero_text(exit_status: Option<i32>, stdout: &str, stderr: &str) -> String {
    let mut text = match exit_status {
        Some(code) => format!("command exited with status {code}"),
        None => "command was killed by a signal".to_owned(),
    };
    let last_line = |output: &str| {
        output
            .lines()
            .map(str::trim)
            .rfind(|line| !line.is_empty())
            .map(str::to_owned)
    };
    if let Some(line) = last_line(stderr).or_else(|| last_line(stdout)) {
        text.push_str(": ");
        match line.char_indices().nth(QUOTED_OUTPUT_CHARS) {
            Some((cut, _)) => {
                text.push_str(&line[..cut]);
                text.push('…');
            }
            None => text.push_str(&line),
        }
    }
    text
}

/// A duration as a person reads it: "30 s", "250 ms", "1.5 s".
pub fn duration_text(duration: Duration) -> String {
    if duration.subsec_nanos() == 0 {
        format!("{} s", duration.as_secs())
    } else if duration < Duration::from_secs(1) {
        format!("{} ms", duration.as_millis())
    } else {
        format!("{:.1} s", duration.as_secs_f64())
    }
}

#[derive(Clone, Debug, Default)]
pub struct CancellationToken {
    inner: Arc<CancellationState>,
}

#[derive(Debug, Default)]
struct CancellationState {
    cancelled: AtomicBool,
    wakers: Mutex<Vec<Waker>>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        if !self.inner.cancelled.swap(true, Ordering::SeqCst) {
            let wakers = std::mem::take(
                &mut *self
                    .inner
                    .wakers
                    .lock()
                    .expect("cancellation token lock poisoned"),
            );
            for waker in wakers {
                waker.wake();
            }
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.inner.cancelled.load(Ordering::SeqCst)
    }

    /// Withdraw a cancellation the operation cannot honour, such as a process
    /// that now runs as root. The operation carries on to its real outcome,
    /// so the token reads as not cancelled again.
    pub fn refuse(&self) {
        self.inner.cancelled.store(false, Ordering::SeqCst);
    }

    fn poll_cancelled(&self, waker: &Waker) -> bool {
        if self.is_cancelled() {
            return true;
        }

        let mut wakers = self
            .inner
            .wakers
            .lock()
            .expect("cancellation token lock poisoned");
        if self.is_cancelled() {
            return true;
        }
        if !wakers.iter().any(|registered| registered.will_wake(waker)) {
            wakers.push(waker.clone());
        }
        false
    }
}

#[async_trait(?Send)]
pub trait CommandRunner: Send + Sync {
    async fn run(
        &self,
        request: CommandRequest,
        cancellation: CancellationToken,
    ) -> Result<CommandOutput, CommandError>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GioCommandRunner;

#[async_trait(?Send)]
impl CommandRunner for GioCommandRunner {
    async fn run(
        &self,
        request: CommandRequest,
        cancellation: CancellationToken,
    ) -> Result<CommandOutput, CommandError> {
        run_process(request, cancellation, kill_process).await
    }
}

/// What sending a process SIGKILL achieved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stop {
    Stopped,
    /// The process runs as another user, as `pkexec` does once authorized.
    Refused,
}

fn kill_process(subprocess: &gio::Subprocess) -> Stop {
    // No identifier once the process has exited and been reaped.
    let Some(pid) = subprocess
        .identifier()
        .and_then(|pid| pid.parse::<libc::pid_t>().ok())
    else {
        return Stop::Stopped;
    };
    // SAFETY: kill(2) on our own unreaped child has no memory effects.
    if unsafe { libc::kill(pid, libc::SIGKILL) } == 0 {
        return Stop::Stopped;
    }
    if io::Error::last_os_error().raw_os_error() == Some(libc::EPERM) {
        Stop::Refused
    } else {
        Stop::Stopped
    }
}

/// A cancellation the process refuses is withdrawn from the token and the
/// process is awaited to its real outcome, since abandoning it would report a
/// failure while it goes on changing the system.
async fn run_process(
    request: CommandRequest,
    cancellation: CancellationToken,
    stop: fn(&gio::Subprocess) -> Stop,
) -> Result<CommandOutput, CommandError> {
    if cancellation.is_cancelled() {
        return Err(CommandError::Cancelled);
    }

    let launcher =
        SubprocessLauncher::new(SubprocessFlags::STDOUT_PIPE | SubprocessFlags::STDERR_PIPE);
    for name in &request.removed {
        launcher.unsetenv(name);
    }
    for (name, value) in &request.environment {
        launcher.setenv(name, value, true);
    }
    let argv = std::iter::once(request.executable.as_str())
        .chain(request.arguments.iter().map(String::as_str))
        .map(std::ffi::OsStr::new)
        .collect::<Vec<_>>();
    let subprocess = launcher
        .spawn(&argv)
        .map_err(|error| spawn_error(&request.executable, error))?;

    let mut communication = subprocess.communicate_future(None);
    let mut timeout = request.timeout.map(glib::timeout_future);
    let mut stoppable = true;
    let completed = loop {
        let next = poll_fn(|context| {
            if stoppable && cancellation.poll_cancelled(context.waker()) {
                return Poll::Ready(ProcessResult::Cancelled);
            }
            if let Poll::Ready(result) = communication.as_mut().poll(context) {
                return Poll::Ready(ProcessResult::Completed(result));
            }
            if let Some(timeout) = timeout.as_mut() {
                if timeout.as_mut().poll(context).is_ready() {
                    return Poll::Ready(ProcessResult::TimedOut);
                }
            }
            Poll::Pending
        })
        .await;
        match next {
            ProcessResult::Cancelled => match stop(&subprocess) {
                Stop::Stopped => return Err(CommandError::Cancelled),
                Stop::Refused => {
                    cancellation.refuse();
                    stoppable = false;
                }
            },
            ProcessResult::TimedOut => {
                stop(&subprocess);
                return Err(CommandError::Timeout {
                    timeout: request.timeout.unwrap_or_default(),
                });
            }
            ProcessResult::Completed(result) => break result,
        }
    };

    let (stdout, stderr) = completed.map_err(|error| spawn_error(&request.executable, error))?;
    let stdout = String::from_utf8(stdout.map_or_else(Vec::new, |bytes| bytes.to_vec())).map_err(
        |error| CommandError::InvalidUtf8 {
            stream: OutputStream::Stdout,
            message: error.to_string(),
        },
    )?;
    let stderr = String::from_utf8(stderr.map_or_else(Vec::new, |bytes| bytes.to_vec())).map_err(
        |error| CommandError::InvalidUtf8 {
            stream: OutputStream::Stderr,
            message: error.to_string(),
        },
    )?;
    let exit_status = subprocess.has_exited().then(|| subprocess.exit_status());

    if !subprocess.is_successful() {
        return Err(CommandError::NonZero {
            exit_status,
            stdout,
            stderr,
        });
    }

    Ok(CommandOutput::new(exit_status, stdout, stderr))
}

enum ProcessResult {
    Completed(Result<(Option<glib::Bytes>, Option<glib::Bytes>), glib::Error>),
    TimedOut,
    Cancelled,
}

fn spawn_error(executable: &str, error: glib::Error) -> CommandError {
    let kind = glib_error_kind(&error);
    if kind == io::ErrorKind::NotFound {
        CommandError::NotFound {
            executable: executable.to_owned(),
        }
    } else {
        CommandError::Spawn {
            executable: executable.to_owned(),
            kind,
            message: error.to_string(),
        }
    }
}

fn glib_error_kind(error: &glib::Error) -> io::ErrorKind {
    if let Some(kind) = error.kind::<gio::IOErrorEnum>() {
        return kind.into();
    }

    let spawn_error_domain = unsafe { from_glib(glib::ffi::g_spawn_error_quark()) };
    if error.domain() != spawn_error_domain {
        return io::ErrorKind::Other;
    }
    match error.code() {
        glib::ffi::G_SPAWN_ERROR_NOENT => io::ErrorKind::NotFound,
        glib::ffi::G_SPAWN_ERROR_ACCES | glib::ffi::G_SPAWN_ERROR_PERM => {
            io::ErrorKind::PermissionDenied
        }
        glib::ffi::G_SPAWN_ERROR_INVAL => io::ErrorKind::InvalidInput,
        _ => io::ErrorKind::Other,
    }
}

#[derive(Clone, Default)]
pub struct FakeCommandRunner {
    inner: Arc<Mutex<FakeState>>,
}

#[derive(Default)]
struct FakeState {
    calls: Vec<CommandRequest>,
    outcomes: VecDeque<Result<CommandOutput, CommandError>>,
}

impl FakeCommandRunner {
    pub fn scripted(
        outcomes: impl IntoIterator<Item = Result<CommandOutput, CommandError>>,
    ) -> Self {
        Self {
            inner: Arc::new(Mutex::new(FakeState {
                calls: Vec::new(),
                outcomes: outcomes.into_iter().collect(),
            })),
        }
    }

    pub fn calls(&self) -> Vec<CommandRequest> {
        self.inner
            .lock()
            .expect("fake runner lock poisoned")
            .calls
            .clone()
    }
}

#[async_trait(?Send)]
impl CommandRunner for FakeCommandRunner {
    async fn run(
        &self,
        request: CommandRequest,
        cancellation: CancellationToken,
    ) -> Result<CommandOutput, CommandError> {
        let mut state = self.inner.lock().expect("fake runner lock poisoned");
        state.calls.push(request);
        if cancellation.is_cancelled() {
            return Err(CommandError::Cancelled);
        }
        state
            .outcomes
            .pop_front()
            .unwrap_or(Err(CommandError::FakeScriptExhausted))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sleep(seconds: &str) -> CommandRequest {
        CommandRequest::new("sleep".to_owned(), vec![seconds.to_owned()])
    }

    fn cancel_soon(token: &CancellationToken) {
        let token = token.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            token.cancel();
        });
    }

    #[test]
    fn a_request_without_timeout_runs_to_completion() {
        let request = sleep("0.2")
            .with_timeout(Duration::from_millis(10))
            .without_timeout();
        assert_eq!(request.timeout(), None);

        let output = glib::MainContext::new()
            .block_on(GioCommandRunner.run(request, CancellationToken::new()))
            .unwrap();

        assert_eq!(output.exit_status(), Some(0));
    }

    #[test]
    fn a_refused_cancellation_is_withdrawn_and_the_process_awaited() {
        let token = CancellationToken::new();
        cancel_soon(&token);

        let output = glib::MainContext::new()
            .block_on(run_process(sleep("0.3"), token.clone(), |_| Stop::Refused))
            .unwrap();

        assert_eq!(output.exit_status(), Some(0));
        assert!(!token.is_cancelled());
    }

    #[test]
    fn a_stopped_process_is_cancelled() {
        let token = CancellationToken::new();
        cancel_soon(&token);

        let error = glib::MainContext::new()
            .block_on(run_process(sleep("5"), token.clone(), kill_process))
            .unwrap_err();

        assert_eq!(error, CommandError::Cancelled);
        assert!(token.is_cancelled());
    }

    fn non_zero(exit_status: Option<i32>, stdout: &str, stderr: &str) -> String {
        CommandError::NonZero {
            exit_status,
            stdout: stdout.to_owned(),
            stderr: stderr.to_owned(),
        }
        .to_string()
    }

    /// modelctl list-engines on a backend without hardware-observe: the
    /// message must carry the cause, not `Some(1)`.
    #[test]
    fn a_failed_command_says_how_it_ended_and_its_last_stderr_line() {
        let text = non_zero(
            Some(1),
            "partial listing\n",
            "Traceback (most recent call last):\n  ...\n\
             PermissionError: [Errno 13] Permission denied: '/sys/bus/usb/devices'\n\n",
        );
        assert_eq!(
            text,
            "command exited with status 1: PermissionError: [Errno 13] \
             Permission denied: '/sys/bus/usb/devices'"
        );
        assert!(!text.contains("Some("), "{text}");
    }

    #[test]
    fn a_signalled_command_says_so_and_falls_back_to_stdout() {
        assert_eq!(non_zero(None, "", ""), "command was killed by a signal");
        assert_eq!(
            non_zero(None, "out of memory\n", "  \n"),
            "command was killed by a signal: out of memory"
        );
    }

    #[test]
    fn a_long_output_line_is_cut() {
        let text = non_zero(Some(2), "", &"é".repeat(QUOTED_OUTPUT_CHARS + 5));
        assert!(text.ends_with('…'), "{text}");
        assert_eq!(
            text.chars().filter(|&c| c == 'é').count(),
            QUOTED_OUTPUT_CHARS
        );
    }

    #[test]
    fn timeouts_and_streams_read_as_words() {
        let timeout = |duration| CommandError::Timeout { timeout: duration }.to_string();
        assert_eq!(
            timeout(Duration::from_secs(10)),
            "command timed out after 10 s"
        );
        assert_eq!(
            timeout(Duration::from_millis(250)),
            "command timed out after 250 ms"
        );
        assert_eq!(
            timeout(Duration::from_millis(1500)),
            "command timed out after 1.5 s"
        );
        let invalid = CommandError::InvalidUtf8 {
            stream: OutputStream::Stderr,
            message: "bad byte".to_owned(),
        };
        assert_eq!(invalid.to_string(), "stderr was not valid UTF-8: bad byte");
    }
}
