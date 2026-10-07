# Incremental codec streaming

`inference_live.py` is an experimental, bounded-memory streaming runtime for
Mage-VL. It accepts a local file or RTSP source, keeps one FFmpeg process alive,
and emits a gate decision as soon as each codec segment is finalized and processed.

This differs from `inference_streaming.py`, which is a useful **offline causal
replay**: it preprocesses every segment of a complete file, stores all segment
tensors, runs the gate once, and prints only after preprocessing finishes.

## What “live” means in this implementation

`codec-video-prep==0.2.5` accepts file paths, not packets or an open decoder. The
runtime therefore uses causal, keyframe-aligned micro-batches:

```text
file / RTSP
    -> persistent FFmpeg process
    -> finalized H.264 segment
    -> cv-preinfer canvas + patch selection
    -> Mage-ViT segment features
    -> persistent Mamba gate state
    -> immediate silence/response event
```

Only the current segment and an approximately 200 KiB recurrent gate cache are kept
on the GPU. Segment files and codec caches are deleted after use by default. The
minimum gate latency is one `--segment-sec` interval plus preprocessing and model
time; this is not frame-by-frame packet ingestion.

## Known-compatible environment

The live path requires Linux and an NVIDIA GPU. `mamba-ssm==2.3.2.post1` currently
pulls a NumPy-2 Mamba3 dependency stack, while `codec-video-prep==0.2.5` declares
NumPy below 2. Keep the model and `cv-preinfer` tool in two small environments; the
model invokes the codec tool as a subprocess. This also uses SDPA instead of requiring
Flash Attention.

Model environment:

```bash
python3.12 -m venv .venv-live
source .venv-live/bin/activate
python -m pip install --upgrade pip setuptools wheel ninja packaging

python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.9.1 torchvision==0.24.1

python -m pip install \
  numpy==2.4.2 transformers==5.7.0 accelerate safetensors \
  'huggingface_hub>=1.5,<2' pillow 'scipy>=1.11,<1.16' \
  decord==0.6.0 'moviepy<2' opencv-python-headless==4.11.0.86

MAX_JOBS=8 python -m pip install mamba-ssm==2.3.2.post1 \
  --no-build-isolation
```

Codec-tool environment:

```bash
python3.12 -m venv .venv-codec
.venv-codec/bin/python -m pip install --upgrade pip setuptools wheel
.venv-codec/bin/python -m pip install \
  numpy==1.26.4 opencv-python-headless==4.11.0.86 pillow==12.3.0 \
  codec-video-prep==0.2.5

export CV_PREINFER_BIN="$PWD/.venv-codec/bin/cv-preinfer"
```

System `ffmpeg` and `ffprobe` are also required. Verify the complete preprocessing
stack before loading the model:

```bash
nvidia-smi
ffmpeg -version
ffprobe -version
"$PWD/.venv-codec/bin/codec-video-prep-doctor"
"$CV_PREINFER_BIN" --help
```

`codec-video-prep==0.2.5` has a known doctor inconsistency: it expects
`thread_type=auto` while its own config defaults to `slice`. A lone
`thread_type: FAIL (slice)` is a false negative when the import/library checks and
the standalone `cv-preinfer` smoke test succeed.

## Smoke test with the bundled video

The runner pins the released checkpoint revision by default and downloads a complete
snapshot, including the 1.07 GB StreamMind gate. It auto-confirms only the redundant
nested Transformers prompt for that content-addressed local snapshot:

```bash
CUDA_VISIBLE_DEVICES=0 python mage_vl/inference_live.py \
  --source mage_vl/assets/examples/soccer-broadcast.mp4 \
  --video-backend codec \
  --codec-engine hevc \
  --segment-sec 8 \
  --max-segments 4 \
  --gate-threshold 0.5 \
  --attn-impl sdpa \
  --output mage_live_events.jsonl
```

Local files are rate-limited with FFmpeg `-re` by default, so decisions arrive at
stream speed. Use `--no-realtime` for a faster causal functional test.

Each event is printed and flushed to JSONL immediately:

```text
[t=0.0-8.0s] gate=silence (p=0.19) prep=...s gate=...s
[t=8.0-16.0s] gate=response (p=0.55) prep=...s gate=...s -> ...
```

The JSONL record also includes EPFE step count, recurrent-state bytes, whether FFmpeg
was still ingesting when the event was emitted, CUDA allocated/reserved bytes, source
lag, and preprocessing/gate/generation latency.

### Validated behavior

The pinned stack above was validated on one NVIDIA H800 with the bundled 30-second
football clip:

- incremental segment-boundary gate logits matched the released one-shot gate within
  `atol=rtol=2e-3` in BF16;
- the first generated response was flushed for `[t=0.0-4.0s]` while the 30-second
  FFmpeg source process was still running;
- codec preprocessing took 0.68–0.96 seconds per four-second segment;
- the recurrent state remained exactly 204,800 bytes while EPFE steps increased;
- CUDA allocated memory stayed within 4.2 MB across all eight segments; and
- processed and unprocessed segment files were removed, with GPU usage returning to
  zero after process exit.

## RTSP

```bash
CUDA_VISIBLE_DEVICES=0 python mage_vl/inference_live.py \
  --source 'rtsp://user:password@camera.example/live' \
  --rtsp-transport tcp \
  --segment-sec 4 \
  --max-backlog-segments 4 \
  --video-backend codec \
  --attn-impl sdpa \
  --output camera_events.jsonl
```

By default FFmpeg normalizes input to H.264 with forced segment-boundary keyframes,
which also handles VP6/MPEG-4/other inputs that `cv-preinfer` cannot consume directly.
Use `--copy-codec` only when the source is already H.264/HEVC with sufficiently
frequent keyframes; its boundaries follow source keyframes rather than exact wall
clock intervals.

## State and position handling

- `cv-preinfer` writes `src_patch_position.npy` for every completed segment.
- Mage's processor converts those values to `patch_positions` consumed by Mage-ViT's
  3D RoPE. No manual patch-position construction is needed.
- Patch positions intentionally remain local to each codec segment, matching the
  released offline segmented baseline.
- Cross-segment temporal state is carried by one Mamba `InferenceParams` cache per
  stream. The cache advances by EPFE/canvas time tokens, not raw frames.
- `IncrementalStreamMindSession.reset()` creates fresh state for a discontinuity or
  a new stream.

## Current limitations

- The codec preprocessor is segment-granular; `codec-video-prep` does not expose an
  incremental packet/GOP API.
- Response generation uses the current segment only. The gate retains stream state,
  but the language-model prompt does not yet include a recent-segment visual buffer.
- If processing is slower than the source, finalized segments queue on disk. The
  event record's `source_lag_s` makes this visible. Processing is lossless by
  default; set `--max-backlog-segments N` to drop a stale prefix above `N` segments
  and reset recurrent state for low-latency deployments.
- Traditional H.264/HEVC is the tested first target. `--codec-engine dcvc-rt` is
  exposed but requires the complete local checkpoint snapshot and more setup.
- One process owns one stream session. Do not share its Mamba cache between streams.

## Tests

```bash
python -m unittest discover -s mage_vl/tests -v
```

The GPU acceptance test additionally compares incremental segment-boundary logits
against `streammind_gate_forward_segments` on identical preprocessed segments and
checks that recurrent-state memory remains constant as stream length grows.
