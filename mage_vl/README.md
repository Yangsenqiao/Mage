<h1 align="center">Mage-VL<br><span style="font-size: 0.55em; font-weight: normal;">An Efficient Codec-Native Streaming Multimodal Foundation Model</span></h1>

<p align="center">
  <a href="https://microsoft.github.io/Mage"><img alt="Project Page" src="https://img.shields.io/badge/%F0%9F%8C%90-Project%20Page-blue" height="22" /></a>
  <a href="https://arxiv.org/abs/2607.24904"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-Mage--VL-b31b1b" height="22" /></a>
  <a href="https://github.com/microsoft/Mage"><img src="https://img.shields.io/badge/Code-GitHub-181717?logo=github" alt="GitHub" height="22"></a>
  <a href="https://huggingface.co/microsoft/Mage-VL"><img alt="Mage-VL" src="https://img.shields.io/badge/%F0%9F%A4%97-Mage--VL-yellow" height="22" /></a>
  <a href="https://huggingface.co/microsoft/Mage-ViT"><img alt="Mage-ViT" src="https://img.shields.io/badge/%F0%9F%A4%97-Mage--ViT-yellow" height="22" /></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="License: Apache 2.0" height="22"></a>
</p>

<div align="center">
<img src="assets/mage-vl-cover.png" width="100%" alt="Mage-VL">
</div>

---

**Mage-VL** is a **codec-native, proactive-streaming multimodal foundation model** for image and video understanding, whose visual perception stack is trained **entirely from scratch** at a compact **4B** scale. It targets a modern *Moravec's paradox* of VLMs — strong at complex offline reasoning, yet slow and compute-heavy on simple real-time streaming perception. Instead of decoding video into uniformly-sampled frames and pushing a dense grid of patch tokens through a frozen web-pretrained ViT, Mage-VL follows the structure of modern video codecs: it separates a stream into **anchor (I) frames** and **predicted (P) frames**, keeps every anchor patch, and retains only the predicted-frame patches where the codec spends bits — the regions carrying real motion and new detail. This codec-aligned sparsity cuts visual tokens by **over 75%** while preserving spatio-temporal context, yielding **up to 3.5× wall-clock inference speedup** over uniform frame sampling.

The system pairs **two components**:

- **Mage-ViT** — a from-scratch *Codec-ViT* visual encoder that allocates tokens by codec-derived spatio-temporal importance, on a shared `16×16` patch grid with 3D rotary position encoding. It is **codec-agnostic**: the same interface accepts a traditional codec (H.264/AVC, HEVC/H.265) via motion vectors + residual energy, or a neural codec (DCVC-RT) via its learned rate map — no architecture or retraining change.
- **Qwen3-4B causal decoder** — a Qwen3-4B-Instruct-2507 language backbone that consumes Mage-ViT's variable-length token stream through a lightweight two-layer MLP projector, with a unified interface for images, short/long/ultra-long video, and streaming.

On top of this pair, a **System 1 & System 2 dual-process design** adds proactive streaming inside a single model: a lightweight **cognition gate** (System 1) watches each rolling codec window and stays silent on routine content, invoking the full VLM (System 2) only when a response-worthy event completes — no multi-agent pipeline required.

## ✨ Highlights

- **Codec-native & from scratch.** The entire visual stack is trained from scratch — no billion-scale image-text ViT initialization. The bio-inspired predictive-patch mechanism (I/P frames at `16×16`) cuts visual-token consumption by **over 75%** (**~1/8 or less** of dense frame sampling), letting the model train on videos **8× longer** under the same budget.
- **Codec-native speedup.** Codec tokenization sets a superior accuracy–efficiency frontier — **up to 3.5× wall-clock inference speedup** over uniform frame sampling at matched accuracy, and the fastest of all compared models on most video benchmarks.
- **Native-resolution scaling.** Variable-resolution pretraining lets Mage-ViT improve *monotonically* with the token budget (peaking **>96.1% Food-101 / >86.3% ImageNet** at 676 tokens) where fixed-resolution encoders saturate or degrade.
- **Strong for its size.** On par with Qwen3-VL-4B on static images, and clearly ahead on video understanding and spatial intelligence (**+11.0** VSI-Bench, **+53.1** CrossPoint, **+5.2** EmbSpatial, **+22.5** QVHighlight).
- **Proactive streaming, single model.** A frozen-backbone cognition gate delivers low-latency, event-gated commentary; it tops **TimVal / F1 / ROC-AUC / PR-AUC** on SoccerNet streaming and generalizes to real 2026 World Cup broadcasts.

## 📥 Model

A **single checkpoint**, `microsoft/Mage-VL`, is one unified model that **simultaneously** provides image & video understanding **and** the proactive streaming gate — the same weights answer offline image/video questions and drive event-gated commentary. It covers every Mage-VL capability: image understanding, frame-sampled video, traditional H.264/HEVC codec video, neural DCVC-RT codec video, and event-gated streaming. The repository bundles the codec processor, the neural codec package, and the proactive gate weights — no separate understanding, NVC, or streaming checkpoint is required.

We additionally release **`microsoft/Mage-ViT`** — the standalone visual encoder from the two-stage, from-scratch ViT pre-training. This is the **ViT-pre-trained checkpoint only**: it has **not** gone through the joint VLM training with the language model. Use it as a data-efficient, codec-native visual encoder or as a drop-in ViT for your own multimodal training.

| Model | Task | Backbone | Hugging Face |
| :--- | :--- | :--- | :--- |
| `Mage-VL` | image & video understanding **+** proactive streaming gate | Mage-ViT + Qwen3-4B-Instruct-2507 | [🤗 microsoft/Mage-VL](https://huggingface.co/microsoft/Mage-VL) |
| `Mage-ViT` | codec-native visual encoder — ViT pre-training only, no VLM joint training | Codec-ViT (from scratch) | [🤗 microsoft/Mage-ViT](https://huggingface.co/microsoft/Mage-ViT) |

## 🏗️ Architecture

<div align="center">
<img src="assets/mage-vl-framework.png" width="100%" alt="Mage-VL proactive streaming framework"><br>
<em>Proactive streaming framework — Mage-ViT incrementally encodes the continuous stream into codec-native visual features shared by the event gate and the causal decoder. The gate scores each rolling window and stays silent on routine content; when it opens, the decoder emits an event-conditioned response.</em>
</div>

**Mage-ViT** — a from-scratch Codec-ViT visual encoder. On a `16×16` patch grid it keeps all anchor (I) frame patches and only the motion-salient predicted (P) frame patches, cutting visual tokens by over 75% while a shared 3D RoPE preserves spatio-temporal positions.

**Mage-VL** — a unified model where projected visual tokens and text tokens share one causal Qwen3 decoder. Still images become a single spatial block; videos become temporally-ordered codec windows. In **streaming** mode, a lightweight **cognition gate** predicts `p_speak = g(h_t)` per rolling window (over a recurrent streaming memory kept by an event-preserving feature extractor) and triggers generation when `p_speak ≥ τ`; the response is decoded by the frozen base model from a local sliding window of the most recent codec segments, and a text query can be injected at any time.

**Training** — a progressive **five-stage** supervised curriculum (no preference/RL post-training) that produces one unified model:

1. **Multimodal alignment via captions** — ~350M dense image captions + 4.2M short-video captions.
2. **Instruction tuning + short temporal grounding** — ~54M image-instruction samples + 3.4M 30–180s video captions.
3. **Temporal-horizon expansion** — medium/long video (LLaVA-Video, TimeLens, VideoChat-Flash, Molmo2) with retained image SFT.
4. **Codec-native long-context adaptation** — 350K long videos as rolling codec windows (up to 384/768 frames).
5. **Proactive streaming alignment** — a cognition gate fine-tuned on ~3.3M streaming samples with the visual encoder and LLM kept frozen (only the gate is trained).

The five stages together produce a **single unified model**, `Mage-VL`, that handles image understanding, offline video reasoning, and proactive streaming — no separate variants are shipped.

Two parts of the pipeline apply an **AI4AI** (AI-for-AI) paradigm: (1) dense recaptioning runs through an agentic closed loop where a GPT-5 rubric scorer grades captions and a Copilot coding agent co-designs the prompt *and* harness code (e.g. rendering timestamp overlays) under a human validation gate — improving every downstream OCR/doc/chart/perception benchmark and inspiring SkillOpt-Lite; and (2) Stage-3 uses AI-based diagnostics to decide which video categories, resolutions, and frame counts to train on.

## 📊 Performance

<details>
<summary><b>Image understanding & spatial intelligence — click to expand</b></summary>

Performance comparison across models. Mage-VL-4B and Qwen3-VL-4B use the same 4B Qwen3 LLM backbone; Phi-4-Multimodal-Instruct (Phi-4-MM, 5.6B) and Phi-4-Reasoning-Vision (Phi-4-R-V, 15B) are reported for reference. `–` = not run. **Bold** = best in row.

| Benchmark | Mage-VL-4B | Qwen3-VL-4B | Phi-4-MM-5.6B | Phi-4-R-V-15B |
| :--- | :---: | :---: | :---: | :---: |
| *Document understanding* | | | | |
| DocVQA-val | **95.14** | 94.69 | 92.79 | 76.20 |
| InfoVQA-val | **80.33** | 79.50 | 71.84 | 55.41 |
| AI2D w/ Mask | **83.16** | 81.54 | 81.83 | 82.87 |
| ChartQA | **84.88** | 83.96 | 83.76 | 83.40 |
| OCRBench | **81.80** | 81.60 | 81.70 | 73.90 |
| MultiDocVQA-val | **87.46** | 87.21 | 46.84 | 58.35 |
| ChartQAPro | **32.57** | 26.79 | 0.13 | 25.38 |
| TextVQA-val | 77.28 | **80.55** | 39.93 | 76.06 |
| CC-OCR Doc | 32.25 | **39.69** | 4.99 | 17.65 |
| *General VQA* | | | | |
| MMBench-EN-dev | 84.02 | 83.25 | 65.81 | **84.19** |
| MMBench-CN-dev | **82.04** | 80.58 | 75.17 | 79.47 |
| MMStar | **67.32** | 62.04 | 61.24 | 59.63 |
| MME-Perception | **1709.54** | 1703.50 | 1409.66 | 1590.21 |
| SeedBench (All) | **76.78** | 75.65 | 68.28 | 73.70 |
| CV-Bench | **87.79** | 85.37 | 57.09 | 81.31 |
| MME-RealWorld | **66.52** | 63.20 | 32.45 | 57.80 |
| *Spatial intelligence* | | | | |
| CV-Bench-2D | **82.13** | 81.00 | 56.12 | 80.11 |
| CV-Bench-3D | **94.75** | 92.30 | 56.92 | 82.50 |
| BLINK | **65.11** | 65.10 | 35.24 | 57.80 |
| EmbSpatial | **82.67** | 77.50 | 41.51 | 72.67 |
| CrossPoint | **80.00** | 26.90 | 12.20 | 47.73 |
| CRPE-Relation | 76.12 | **77.70** | 34.60 | 74.46 |
| SAT | 67.33 | **69.30** | 55.33 | 66.67 |

</details>

<details>
<summary><b>Video understanding & temporal grounding — click to expand</b></summary>

**Bold** = best in row.

| Benchmark | Mage-VL-4B | Qwen3-VL-4B | Phi-4-MM-5.6B | Phi-4-R-V-15B |
| :--- | :---: | :---: | :---: | :---: |
| *Video QA* | | | | |
| MV-Bench | 65.1 | **66.7** | 44.9 | 49.2 |
| NextQA | **83.1** | 79.8 | 54.1 | 69.0 |
| VideoMME | **64.0** | 59.7 | 44.7 | 55.3 |
| LongVideoBench | **61.3** | 57.7 | 41.14 | 51.2 |
| LVBench | **41.8** | 39.2 | 25.31 | 34.4 |
| MLVU-dev | **68.7** | 61.5 | 44.18 | 51.8 |
| VideoEval-Pro | **45.2** | 20.7 | 14.35 | 16.8 |
| *Temporal grounding* | | | | |
| Timelens-Charades | **50.7** | 43.1 | 4.09 | 20.6 |
| Timelens-ActivityNet | **45.4** | 28.4 | 2.03 | 23.0 |
| Timelens-QVHighlight | **57.4** | 34.9 | 2.47 | 11.6 |
| *Spatial reasoning* | | | | |
| VSI-Bench | **64.3** | 53.3 | 24.09 | 25.5 |
| *Tracking (J&F)* | | | | |
| Ref-DAVIS17 | **25.83** | 7.48 | 3.14 | 2.15 |
| MeViS-ValidU | **22.55** | 3.16 | 10.28 | 1.53 |
| ReasonVOS | **17.76** | 9.66 | 9.50 | 9.77 |
| Ref-YT-VOS | **25.57** | 5.28 | 8.64 | 3.85 |

</details>

<details>
<summary><b>Proactive streaming (SoccerNet) & online video (OVO-Bench) — click to expand</b></summary>

**SoccerNet — response timing** (StreamMind protocol, codec-native inputs, zero-tolerance canvas matching). **Bold** = best in column.

| Method | TriggerAcc | TimVal | F1 | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| StreamMind | 52.18 | 47.36 | – | – | – |
| JoyAI-VL-Interaction-9B | **97.98** | 19.25 | 3.55 | 56.26 | 1.68 |
| **Mage-VL-4B** | 79.21 | **55.54** | **16.35** | **83.14** | **9.30** |

JoyAI's high TriggerAcc comes from predicting silence almost everywhere under SoccerNet's heavy class imbalance, so it collapses on the precision-sensitive metrics; StreamMind is trained *in-distribution* on SoccerNet, whereas Mage-VL is not.

**OVO-Bench** — online video understanding (SimpleStream recent-window protocol, 4 frames @ 1 fps; no streaming-specific fine-tuning). Mage-VL sets a new state-of-the-art overall score **among streaming architectures**. RT-Avg / BT-Avg are the Real-Time Visual Perception / Backward Tracing sub-task averages; Overall is their mean. **Bold** = best model per column (Human is the reference upper bound).

| Model | #Frames | RT-Avg | BT-Avg | Overall |
| :--- | :---: | :---: | :---: | :---: |
| Human | – | 93.2 | 92.3 | 92.77 |
| *Offline video LLMs* | | | | |
| Qwen2.5-VL-7B | 1 fps | 59.9 | 44.7 | 52.28 |
| LLaVA-Video-7B | 64 | 63.5 | 40.4 | 51.95 |
| Qwen3-VL-4B | 64 | 72.8 | **53.1** | 63.00 |
| *Online / streaming video LLMs* | | | | |
| VideoLLM-online-8B | 2 fps | 20.8 | 17.7 | 19.26 |
| Flash-VStream-7B | 1 fps | 28.4 | 27.4 | 27.90 |
| Dispider-7B | 1 fps | 54.6 | 36.1 | 45.35 |
| TimeChat-Online-7B | 1 fps | 61.9 | 41.7 | 51.80 |
| StreamForest-7B | 1 fps | 61.2 | 52.0 | 56.60 |
| Streamo-7B | 1 fps | 66.0 | 46.1 | 56.05 |
| HERMES-7B<sup>†</sup> | 1 fps | 69.0 | 49.4 | 59.20 |
| JoyAI-VL-Interaction-9B | 1 fps | 68.4 | 48.6 | 58.50 |
| **Mage-VL-4B** | 1 fps | **79.84** | 48.15 | **64.00** |

<sub><sup>†</sup> HERMES = Qwen2.5-VL-7B + HERMES (4K tokens). Baseline results and table structure follow SimpleStream.</sub>

</details>

## 🔬 Key Findings

Beyond the model, the report distills **seven empirical findings** for efficient multimodal training:

1. **Data-efficient tokenizer.** Mage-ViT trains on only **560M unlabeled images + 100M video frames**, yet matches SigLIP2 pretrained on billions of image-text pairs.
2. **Variable-resolution pretraining scales monotonically.** Quality keeps improving with the visual-token budget instead of saturating/degrading like fixed-resolution encoders.
3. **Codec-native tokenization sets a better accuracy–efficiency frontier** — up to **3.5× wall-clock inference speedup** over uniform frame sampling.
4. **Explicit VideoQA SFT is redundant.** Dense video *captions* + standard image SFT are sufficient for strong zero-shot VideoQA.
5. **Motion–spatial synergy.** Dynamic video training substantially improves static 2D/3D spatial reasoning.
6. **AI4AI data pipeline.** Agentic closed-loop feedback + prompt/code co-design systematically lift caption quality and downstream scores (inspired SkillOpt-Lite).
7. **Zero-Vision SFT for multimodal RL.** Bypassing visual SFT in favor of pure-text reasoning SFT unlocks stronger multimodal RL — a compute-efficient path.

## 🚀 Quick Start

A single checkpoint, `microsoft/Mage-VL`, covers every capability below.

| Capability | Script | How to run |
|---|---|---|
| Image understanding | `inference_base.py` | `--mode offline --image` |
| Frame-sampled video | `inference_base.py` | `--mode offline --video --video-backend frames` |
| Traditional H.264/HEVC codec video | `inference_base.py` | `--mode offline --video --video-backend codec --codec-engine traditional` |
| Neural DCVC-RT codec video | `inference_base.py` | `--mode offline --video --video-backend codec --codec-engine neural` |
| Online image / video (SGLang) | `inference_base.py` | `--mode online … --base-url <server>` |
| Offline causal replay | `inference_streaming.py` | preprocesses a complete file, then scores causal segments |
| Incremental file / RTSP streaming | `inference_live.py` | emits per finalized segment with persistent bounded gate state |

### Installation

```bash
pip install -r mage_vl/requirements.txt
```

Codec-based video inference requires `ffmpeg` and `ffprobe` on `PATH`. The traditional codec path uses the `cv-preinfer` command supplied by `codec-video-prep`; the streaming frames backend uses Decord directly.

> **torch / CUDA:** install a PyTorch build matching your CUDA toolkit first when the default wheel is unsuitable, since `flash-attn` and `mamba-ssm` compile CUDA extensions against your installed torch.

### Examples

Two sample inputs ship with the repository:

| Input | Question | Content |
|---|---|---|
| `mage_vl/assets/examples/dog.jpg` | Describe this image in detail. | Photo of a dog sitting in front of a patterned rug |
| `mage_vl/assets/examples/soccer-broadcast.mp4` | Describe this video. | 30s, 960×540 football broadcast clip |

### Offline inference

Offline mode loads `AutoModelForCausalLM.from_pretrained` directly and supports images, frame sampling, and both codec engines:

```bash
# image
python mage_vl/inference_base.py \
  --mode offline \
  --image mage_vl/assets/examples/dog.jpg \
  --question "Describe this image in detail."
```

> The image depicts a dog sitting on a patterned rug. The dog appears to be a
> medium-sized breed with a thick, fluffy coat. Its fur is primarily white with
> patches of black and brown. The dog's ears are perked up, and it has a calm and
> attentive expression. [...]

```bash
# video — uniform frame sampling
python mage_vl/inference_base.py \
  --mode offline \
  --video mage_vl/assets/examples/soccer-broadcast.mp4 \
  --video-backend frames \
  --num-frames 32 \
  --question "Describe this video."
```

> The video opens with a man in a black polo shirt, sporting a short haircut,
> standing in a stadium. He is holding a yellow microphone with the BBC Sport
> logo on it. The background reveals a large crowd of spectators. [...]

```bash
# video — traditional codec (HEVC/H.264)
python mage_vl/inference_base.py \
  --mode offline \
  --video mage_vl/assets/examples/soccer-broadcast.mp4 \
  --video-backend codec \
  --codec-engine traditional \
  --num-frames 32 \
  --question "Describe this video."
```

> The video opens with a BBC Sport broadcast, featuring a presenter in a black
> shirt holding a yellow microphone. The background reveals a packed stadium,
> with the scoreboard displaying "ENG 1 ARG 2 FT", indicating the final score of
> the match. [...]

```bash
# video — neural codec (DCVC-RT)
python mage_vl/inference_base.py \
  --mode offline \
  --video mage_vl/assets/examples/soccer-broadcast.mp4 \
  --video-backend codec \
  --codec-engine neural \
  --num-frames 32 \
  --question "Describe this video."
```

> The video opens with a BBC Sport broadcast, featuring a presenter standing in a
> stadium filled with spectators. The presenter, dressed in a black shirt, holds
> a yellow BBC Sport microphone and wears a black earpiece. [...]

### Online inference

Online mode talks to an OpenAI-compatible SGLang server. **First** build and launch the server with the Mage-VL SGLang branch (building it needs `protobuf-compiler` and a Rust toolchain):

```bash
sudo apt-get update && sudo apt-get install -y protobuf-compiler
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
  | sh -s -- -y --profile minimal --default-toolchain 1.90.0
source "$HOME/.cargo/env"

git clone -b feat/mage-vl https://github.com/kcz358/sglang
cd sglang
pip install -e 'python[all]'
python -m sglang.launch_server \
  --model-path microsoft/Mage-VL \
  --trust-remote-code
```

**Then** send an image or sampled video frames to the running server:

```bash
pip install openai

python mage_vl/inference_base.py \
  --mode online \
  --image mage_vl/assets/examples/dog.jpg \
  --question "Describe this image in detail." \
  --base-url http://localhost:30000/v1

python mage_vl/inference_base.py \
  --mode online \
  --video mage_vl/assets/examples/soccer-broadcast.mp4 \
  --num-frames 32 \
  --question "Describe this video." \
  --base-url http://localhost:30000/v1
```

Use `--model`, `--max-new-tokens`, and `--api-key` to override their defaults.

### Offline causal replay

`inference_streaming.py` processes a prerecorded video causally in non-overlapping
segments. It prepares the complete file before scoring, so it is a replay/evaluation
tool rather than a live camera loop. The gate stays silent on routine content and
generates a caption only when a response-worthy event is detected:

```bash
python mage_vl/inference_streaming.py \
  --video mage_vl/assets/examples/soccer-broadcast.mp4 \
  --video_backend codec \
  --segment_sec 8
```

```text
[t=0.0-8.0s] gate=silence (p=0.19)
[t=8.0-16.0s] gate=response (p=0.55) -> The video features a live sports broadcast from BBC Sport, set in a large stadium filled with spectators. The broadcast focuses on a football match between England and Argentina, with the score displayed as England 1, Argentina 2. [...]
[t=16.0-24.0s] gate=response (p=0.73) -> The video features a sports broadcast set in a large stadium filled with spectators. Four commentators are gathered around a table with a 'BBC Sport' logo, each holding a yellow microphone. [...]
[t=24.0-30.0s] gate=silence (p=0.31)
```

The gate is trained on codec inputs, so `--video_backend codec` is the intended setting. Use `--video_backend frames` for direct frame sampling. Additional controls include `--num_frames`, `--cur_fps`, `--max_segments`, `--max_new_tokens`, `--gate_threshold`, and `--attn_impl`.

For bounded-memory incremental file or RTSP ingestion, persistent cross-segment gate
state, pinned dependencies, and codec troubleshooting, see
[`LIVE_STREAMING.md`](LIVE_STREAMING.md).

## 📝 Citation

```bibtex
@article{yang2026mage,
  title={Mage-VL: An Efficient Codec-Native Streaming Multimodal Foundation Model},
  author={Yang, Senqiao and Zhang, Kaichen and Jia, Zhaoyang and Guo, Jinghao and Shen, Yifei and Zhang, Xinjie and Zhang, Xiaoyi and Wang, Haoqing and Li, Xiao and Zhang, Peng and others},
  journal={arXiv preprint arXiv:2607.24904},
  year={2026}
}
```

## 📄 License

Mage-VL is released under the [Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
