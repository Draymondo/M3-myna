# Preface

Read this document before evaluating a model or a change to one (new
backend, quantisation, export, runtime or execution provider), treating
Parakeet's window-collapse rate as a port bug, tuning ONNX Runtime CUDA
execution, re-attempting an fp16 Parakeet export, or reviving the Nemotron
3.5 ASR evaluation.

Read the top-level `.kb/agents.md` file before continuing below.

# Overview

Findings from the 2026-09-17 Parakeet GPU workstream (fp32 CUDA export, NeMo
parity, fp16 attempt) and the parked Nemotron 3.5 ASR evaluation. Numbers are
from zephyrus (RTX 4080 Laptop 12 GB, driver 595) on the 82-clip
`corpus/english/manifest-balanced.json` (+ 5-minute long-form clip) or FLEURS
(40 clips/language), unless noted.

# Important

## Evaluating a change

Decide first which question is being asked; they need different evidence.
Every number comes from `myna.benchmarker` runs that pass `make bench-check`
(`server/.kb/benchmarking.md`).

### Same model, different numerics

Quantisation, export, kernel, ONNX Runtime version or execution provider:
"did it break?". Against the shipped artifact:

1. Controls first: an identical rerun, and if possible a trivially
   perturbed one, give the churn floor everything else is read against.
2. Accuracy: `myna-bench compare` on the balanced corpus; the WER delta
   must not be worse beyond its interval. Read the per-clip diffs: noise is
   symmetric and sits on short or marginal clips, damage is directional
   (clean clips regress with nothing offsetting).
3. Failure modes: each known pathology's rate is not worse, and its churn
   is symmetric. A skew is a signal even in the good direction: the
   rejected 2026-08 T07 encoder collapsed less (8 windows regressed, 29
   fixed) while its WER got systematically worse.
4. Speed and resources: distributions over repeats, never one number.
5. Runtime nets such as `_transcribe_guarded` stay regardless.

Never gate a knife-edge metric on zero regressions. Collapse flips on
ulp-scale differences (a metadata-only kernel swap moved ~10 of 557
windows both ways; a rerun moved none), so a zero-regression bar admits only
bit-identical changes. A probe that only counts failures cannot localise
them; per-clip WER on sentinel clips with bisection over the change's parts
can (it found the one damaging FFN node of eleven).

### New model or backend

"Is it better for users?":

1. Validate the pipeline: a full publication preset must land within
   interval of the model's published numbers; if not, suspect our
   adapter, frontend or scoring before the model.
2. Product numbers: WER/CER per language and `--by-category`, finalize
   latency at `pace: realtime`, cold load, RAM/VRAM and J per audio-second,
   on each hardware tier the engine targets. `USABILITY_FAIL` is a result.
3. Find the model's own pathology before trusting its averages, then probe
   it: Parakeet goes blank over windows (below), Fun-ASR-Nano's int8
   export truncates past ~24 s and hallucinates on noise. Long-form and noise categories catch some
   of this; the balanced corpus has no speech-free clip, so hallucination on
   silence needs its own clips.

Collapse is probed by `dev/parakeet/collapse_probe.py`, outside the
benchmarker (no guard, provenance or `compare`); fold failure-mode probes
into the benchmarker when the next evaluation needs one.

## Parakeet v3 window collapse is the model's

A window collapses when it yields under 0.5 tokens per second of audio
(`_COLLAPSE_WORDS_PER_SECOND`, shared by the probe and the runtime retry).
It is a window-boundary effect, not a per-clip one.

557 sliding windows (3-16 s) over concatenated real speech: NeMo PyTorch
collapses 58/557 (10.4%), our ONNX fp32 60/557 (10.8%), 51 windows collapse
in both. The port is not the cause; keep the adapter's retry
(`_transcribe_guarded` in `myna/testbed/parakeet.py`) rather than chasing
this as an export or runtime bug. Reproduce with `dev/parakeet/collapse_probe.py`
run once per backend and diff the collapsed-window sets.

## fp32 ONNX parity with NeMo PyTorch

- fp32 ONNX transcripts equal NeMo PyTorch: 0.00% WER between them, 82 clips.
- Our `encoder-model.onnx.data` is byte-identical to istupakov's HF upload
  (also a NeMo export).
- onnx-asr's `nemo128.onnx` mel frontend matches NeMo's own frontend except
  at the edge frames.

## CUDA execution

- Each new input shape pays ONNX Runtime kernel-setup cost: streaming
  (mostly novel window lengths) measured ~150x this cost versus ~240x for
  repeated lengths. Not mitigated in this beta; would need shape bucketing.
- Arena fragmentation, the IOBinding-on-host pitfall, and GPU engine
  matching are covered in `server/.kb/inference-packaging.md`'s GPU
  engines section; not repeated here.

## fp16 dropped (2026-09-17)

- Naive `onnxruntime.transformers.float16` conversion (`keep_io_types=True`)
  measured slower than fp32 on fresh shapes (Cast nodes around every blocked
  op): streaming 5.6x vs fp32 streaming 9.4x real time.
- Long-window accuracy regresses: 5-minute clip 5.9% WER (fp16) vs 0.9%
  (fp32).
- NeMo 3.0's `autocast` fp16 export yields an invalid graph (mixed LayerNorm
  types).
- ORT's conformer fusion mis-fuses NeMo's relative-position attention.
- A real mixed-precision export (attention scores, Softmax, LayerNorm kept
  fp32) or the TensorRT EP could recover fp16; neither attempted.

## Nemotron 3.5 ASR (parked, not built)

`nvidia/nemotron-3.5-asr-streaming-0.6b`, rev `ea30d66`, OpenMDW-1.1.
Evaluated 2026-09-17, never packaged as a snap engine.

Export recipe (NeMo 3.0.0, inference is torch-free):

- `model.set_export_config({"cache_support": "True"})` before
  `model.export()` yields a cache-aware encoder (cache I/O ports) and
  decoder_joint.
- NeMo's exporter drops the language-prompt projection
  (`PromptStreamingMixin._apply_prompt_to_encoded`); export it separately as
  its own small `torch.onnx.export` graph (`prompt-model.onnx`).
- torch's ONNX exporter cannot export NeMo's STFT, so the mel frontend
  (`normalize` NA) runs as hand-written numpy at inference instead of an
  exported graph.
- The exported encoder bakes in `drop_extra_pre_encoded=2` and
  `valid_out_len`; the inference loop follows NeMo's
  `pad_and_drop_preencoded` semantics and flushes with one chunk of silence
  appended after the real audio.
- Recipe script: deleted 2026-10-11 with the evaluation parked; recover it
  with `git log --diff-filter=D -- dev/eval/export_nemotron35_onnx.py`.

Accuracy, 2026-09-17:

| corpus | Parakeet v3 (fp32 CUDA) | Nemotron 3.5 |
|---|---|---|
| English, 82 clips | - | NeMo 2.77% (offline 2.67), ONNX fp32 2.57%, int8 2.93% |
| FLEURS de | 5.2 | 10.6 |
| FLEURS es | 3.1 | 5.3 |
| FLEURS fr | 5.7 | 11.4 |
| FLEURS ar | not covered | 13.7 (CER 3.9) |
| FLEURS ja | not covered | CER 15.8 |
| FLEURS zh | not covered | CER 25.9 |

WER unless marked CER, 40-clip subset, scored with the current normalizer
(typographic apostrophes folded, see `server/.kb/benchmarking.md`). Before
that fold, fr read 8.1/13.7: FLEURS French references use U+2019 for
elisions, so an ASCII hypothesis like "L'accident" split into two words
against them. Nemotron's de/es/fr still run 26-28% above NVIDIA's published
figures (8.31/4.11/9.03); not root-caused. Parakeet's full-FLEURS numbers
(862/908/676 clips) land within bootstrap CI of NVIDIA's published figures -
see `server/.kb/benchmarking.md`.

- CPU int8 real-time headroom: 14.9x on 4 threads (fp32 CPU 8.4x), fixed
  1.12 s chunk shapes so no CUDA-style per-shape cost applies.
- Why parked: ~2x worse WER than Parakeet where Parakeet has the language.
  Nemotron's advantages are native streaming (~1.1 s chunk-to-commit, no
  VAD-driven windowing) and language coverage (ar/ja/zh/hi/ko, which
  Parakeet v3 lacks).
- If revived: restore the recipe script, lock it (`uv lock --script`) with
  the fsspec override `dev/parakeet/export_parakeet_onnx.py` carries, and
  bring the scratch scripts (streaming inference loop, int8 quantization,
  FLEURS harness, NeMo reference runner) beside it, each with its own PEP 723
  metadata; never add nemo/torch to `server/uv.lock`.
