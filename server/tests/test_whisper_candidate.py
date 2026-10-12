"""Candidate metadata for the faster-whisper adapter (no model/extra needed).

Constructing the adapter and reading ``.candidate`` does not import
faster_whisper, so this runs in the default offline suite. It guards the
label normalisation that keeps result records readable when the snap loads
weights from a local CTranslate2 model-component directory (T15).
"""

import asyncio
import sys

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from myna.core import AudioFormat, SessionConfig, TranscriptionError
from myna.testbed.whisper import WHISPER_RATE, FasterWhisperAdapter, _iso639_1

CANONICAL = (WHISPER_RATE, 1, 2)  # 16 kHz mono S16LE — the only accepted format


def test_iso639_1_drops_region_subtag():
    # faster-whisper rejects BCP-47 region tags; the corpus uses them.
    assert _iso639_1("en-GB") == "en"
    assert _iso639_1("de") == "de"
    assert _iso639_1(None) is None  # auto-detect


@pytest.mark.parametrize(
    ("tag", "code"), [("nb", "no"), ("nb-NO", "no"), ("jv", "jw"), ("fil", "tl")]
)
def test_iso639_1_uses_whispers_code_where_it_differs_from_bcp47(tag, code):
    # FLEURS nb_no, jv_id and fil_ph arrive as their BCP-47 tags; faster-whisper
    # rejects those and knows the languages as no, jw and tl.
    assert _iso639_1(tag) == code


def test_candidate_labels_a_bare_size():
    cand = FasterWhisperAdapter("small").candidate
    assert cand.model == "whisper-small"
    assert cand.engine == "faster-whisper-cpu"
    assert cand.streaming_strategy == "commit-on-finalize"


async def test_unload_drops_the_model():
    # idle-unload (T27); no real model needed — just the reference handling
    adapter = FasterWhisperAdapter("tiny")
    adapter._model = object()
    await adapter.unload()
    assert adapter._model is None
    await adapter.unload()  # idempotent


def test_candidate_labels_a_component_directory_by_leaf():
    # snap passes --model $SNAP_COMPONENTS/model-small
    cand = FasterWhisperAdapter("/snap/whisper/components/42/model-small/", device="cuda").candidate
    assert cand.model == "whisper-model-small"  # leaf, not the absolute path
    assert cand.engine == "faster-whisper-cuda"


@given(
    rate=st.integers(min_value=1, max_value=192_000),
    channels=st.integers(min_value=1, max_value=8),
    width=st.integers(min_value=1, max_value=4),
)
def test_rejects_any_noncanonical_format(rate, channels, width):
    # Rejection happens before the model loads (audio-push: client owns
    # capture + conversion), so this needs neither faster-whisper nor a model.
    assume((rate, channels, width) != CANONICAL)
    fmt = AudioFormat(sample_rate_hz=rate, channels=channels, sample_width_bytes=width)

    async def drive():
        events = []

        async def emit(event):
            events.append(event)

        async def no_audio():
            for chunk in ():  # rejected before any chunk is read
                yield chunk

        await FasterWhisperAdapter("tiny").run_session(
            SessionConfig(audio_format=fmt), no_audio(), emit
        )
        return events

    events = asyncio.run(drive())
    assert len(events) == 1  # rejected outright, nothing else emitted
    assert isinstance(events[0], TranscriptionError)
    assert events[0].code == "unsupported_audio_format"


# ─── compute type ────────────────────────────────────────────────────────────


def _fake_ctranslate2(monkeypatch, supported):
    """CTranslate2's own answer, stubbed - the real one needs the whisper extra."""
    module = type(
        "M", (), {"get_supported_compute_types": staticmethod(lambda device: set(supported))}
    )
    monkeypatch.setitem(sys.modules, "ctranslate2", module)


def test_a_compute_type_the_device_cannot_do_is_refused(monkeypatch):
    """CTranslate2 rejects float16-on-CPU too, but only from inside the model
    constructor and without naming what it would have taken - which is a sweep
    cell dying on its first clip, or a daemon that starts and then fails on
    first dictation."""
    _fake_ctranslate2(monkeypatch, {"float32", "int8", "int8_float32"})
    adapter = FasterWhisperAdapter("tiny", device="cpu", compute_type="float16")
    with pytest.raises(ValueError, match="not available on device 'cpu'"):
        adapter._check_compute_type()


def test_a_supported_compute_type_passes(monkeypatch):
    _fake_ctranslate2(monkeypatch, {"float32", "int8", "int8_float32"})
    FasterWhisperAdapter("tiny", device="cpu", compute_type="int8")._check_compute_type()


@pytest.mark.parametrize("deferral", ["default", "auto"])
def test_a_deferral_is_not_a_request_and_is_left_to_ctranslate2(monkeypatch, deferral):
    def explode(device):
        raise AssertionError("a deferral must not be checked against the supported set")

    monkeypatch.setitem(
        sys.modules,
        "ctranslate2",
        type("M", (), {"get_supported_compute_types": staticmethod(explode)}),
    )
    FasterWhisperAdapter("tiny", compute_type=deferral)._check_compute_type()


# ─── GPU unavailable ─────────────────────────────────────────────────────────


class _RecordingWhisperModel:
    """faster_whisper.WhisperModel, recording where it was asked to load."""

    loads: list[tuple[str, str]] = []

    def __init__(self, model_size, *, device, compute_type, download_root):
        self.loads.append((device, compute_type))
        self.model = type("CT2", (), {"compute_type": compute_type})()

    def transcribe(self, samples, **kwargs):
        return iter(()), None


def _fake_whisper_stack(monkeypatch, cuda_devices):
    module = type(
        "M",
        (),
        {
            "get_cuda_device_count": staticmethod(lambda: cuda_devices),
            "get_supported_compute_types": staticmethod(
                lambda device: {"float16", "float32", "int8"} if device == "cuda" else {"float32"}
            ),
        },
    )
    monkeypatch.setitem(sys.modules, "ctranslate2", module)
    _RecordingWhisperModel.loads = []
    monkeypatch.setitem(
        sys.modules, "faster_whisper", type("F", (), {"WhisperModel": _RecordingWhisperModel})
    )


def test_a_cuda_engine_without_a_usable_gpu_loads_on_the_cpu(monkeypatch):
    """The reporter's whisper engine failed every utterance: \"CUDA driver
    version is insufficient for CUDA runtime version\", then a float16 refusal.
    CTranslate2 sees no CUDA device then; load on the CPU in its own type."""
    _fake_whisper_stack(monkeypatch, cuda_devices=0)
    adapter = FasterWhisperAdapter("small", device="cuda", compute_type="float16")

    asyncio.run(adapter._load_model())

    assert _RecordingWhisperModel.loads == [("cpu", "default")]
    assert adapter.capabilities().runtime["device"] == "cpu"


def test_a_usable_gpu_loads_as_asked(monkeypatch):
    _fake_whisper_stack(monkeypatch, cuda_devices=1)
    adapter = FasterWhisperAdapter("small", device="cuda", compute_type="float16")

    asyncio.run(adapter._load_model())

    assert _RecordingWhisperModel.loads == [("cuda", "float16")]
    assert adapter.capabilities().runtime["device"] == "cuda"


def test_a_session_on_a_cpu_fallback_says_so_when_ready(monkeypatch):
    from myna.core import PHASE_READY, WARNING_GPU_UNAVAILABLE
    from myna.server.lifecycle import GPU_UNAVAILABLE_MESSAGE

    _fake_whisper_stack(monkeypatch, cuda_devices=0)
    adapter = FasterWhisperAdapter("small", device="cuda", compute_type="float16")

    async def drive():
        events = []

        async def emit(event):
            events.append(event)

        async def no_audio():
            for chunk in ():
                yield chunk

        fmt = AudioFormat(sample_rate_hz=WHISPER_RATE, channels=1, sample_width_bytes=2)
        await adapter.run_session(SessionConfig(audio_format=fmt), no_audio(), emit)
        return events

    warned = [e for e in asyncio.run(drive()) if getattr(e, "warning", None)]
    assert [(e.phase, e.warning, e.warning_code) for e in warned] == [
        (PHASE_READY, GPU_UNAVAILABLE_MESSAGE, WARNING_GPU_UNAVAILABLE)
    ]
