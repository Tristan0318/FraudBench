<h1 align="center">
  FraudBench: A Multimodal Benchmark for Detecting AI-Generated Fraudulent Refund Evidence
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2605.08820">
    <img src="https://img.shields.io/badge/arXiv-2605.08820-b31b1b.svg?style=for-the-badge&logo=arxiv&logoWidth=20" alt="arXiv"></a>
  &nbsp;&nbsp;
  <a href="https://tristan0318.github.io/FraudBench/">
    <img src="https://img.shields.io/badge/Project-Page-779977?style=for-the-badge&logoWidth=20" alt="Project Page"></a>
  &nbsp;&nbsp;
  <a href="https://huggingface.co/datasets/TristanYan/FraudBench">
    <img src="https://img.shields.io/badge/🤗-Dataset-FF6B00?style=for-the-badge&logoWidth=20" alt="HuggingFace Dataset"></a>
</p>

<p align="center">
  <a href="https://scholar.google.com/citations?user=QXkrWQoAAAAJ">Xinyu Yan</a><sup>1,2</sup>,
  <a href="https://openreview.net/profile?id=~Boyang_Chen6">Boyang Chen</a><sup>1</sup>,
  <a href="https://jiamingzhang94.github.io/">Jiaming Zhang</a><sup>1</sup>,
  <a href="https://scholar.google.com/citations?user=7YsN6lMAAAAJ">Tiantong Wu</a><sup>1,2</sup>,
  <a href="https://bryanhx.github.io/">Hong Xi Tae</a><sup>1</sup>,
  <a href="https://openreview.net/profile?id=~Yichen_He5">Yichen He</a><sup>1</sup>,<br>
  <a href="https://openreview.net/profile?id=~Tiantong_Wang1">Tiantong Wang</a><sup>1,2</sup>,
  <a href="https://scholar.google.com/citations?user=hqLGERIAAAAJ">Yachun Mi</a><sup>1</sup>,
  <a href="https://yuronghaoa.github.io/yuronghaoA/">Yurong Hao</a><sup>1</sup>,
  <a href="https://elainezhao92.github.io/">Yilei Zhao</a><sup>1</sup>,
  <a href="https://openreview.net/profile?id=~Lei_Xiao7">Lei Xiao</a><sup>3</sup>,
  <a href="https://scholar.google.com/citations?user=EQDfV9cAAAAJ">Longtao Huang</a><sup>3</sup>,<br>
  <a href="https://scholar.google.com/citations?user=K2CHjf0AAAAJ">Pengjun Xie</a><sup>3</sup>,
  <a href="https://scholar.google.com/citations?user=yXeVDeoAAAAJ">Wei Liu</a><sup>3</sup>,
  <a href="https://sites.google.com/view/wyb/people">Wei Yang Bryan Lim</a><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>College of Computing and Data Science, Nanyang Technological University<br>
  <sup>2</sup>Alibaba-NTU Global e-Sustainability CorpLab (ANGEL)<br>
  <sup>3</sup>Alibaba Group
</p>

---

This repo is the official implementation of **FraudBench**, proposed in our paper ["FraudBench: A Multimodal Benchmark for Detecting AI-Generated Fraudulent Refund Evidence"](https://arxiv.org/abs/2605.08820).

**FraudBench** is a multimodal benchmark for evaluating AI-generated fraudulent refund evidence detection under realistic transaction settings. It contains **822** real-world review samples and **7,928** images across **29** product/service categories, covering e-commerce, food delivery, and travel services. Fake-damaged evidence is synthesized from real undamaged references using **6** state-of-the-art image editing and generation models. The benchmark evaluates **11 MLLMs**, **4 specialized detectors**, and human participants across **5** evaluation dimensions.

> ⚠️ **Note:** This benchmark is intended solely for academic research purposes. It studies the risk of AI-generated fraudulent refund evidence to support the development of more reliable detection methods, platform safeguards, and responsible evaluation. The goal is not to facilitate refund fraud or provide actionable guidance for misuse.

---

## 🗺️ Navigation

- [📖 Overview](#-overview)
  - [Experiment Conditions](#11-experiment-conditions)
  - [Ablation Studies](#12-ablation-studies)
- [⚡ Requirements](#-requirements)
- [🔑 API Keys](#-api-keys)
- [📂 Script Reference](#-script-reference)
- [🧪 Usage](#-usage)
  - [Main Experiments](#51-main-experiments)
  - [Ablation Studies](#52-ablation-studies)
  - [Human Evaluation Interface](#53-human-evaluation-interface)
- [📊 Output Structure](#-output-structure)
- [📈 Computing Metrics](#-computing-metrics)
- [🤖 Models](#-models)
- [🔎 Reproducibility Notes](#-reproducibility-notes)
- [📄 Citation](#-citation)

---

## 📖 Overview

The pipeline evaluates 11 multimodal large language models (MLLMs) across six experiment conditions and two ablation studies. All inference is performed via vendor APIs; no local GPU is required.

### 1.1 Experiment Conditions

| Mode | Script flag(s) | Description |
|---|---|---|
| SingleImage-NoReview | *(default)* | One image per call, no review text |
| SingleImage-withReview | `--with-review` | One image per call, review injected |
| MultiImage-NoReview | `--review-mode --single-turn` | All images packed into one user message |
| MultiImage-withReview | `--review-mode --single-turn --with-review` | Single-turn delivery with review injected |
| MultiStep-NoReview | `--review-mode` | Images delivered one per turn with continuation prompts |
| MultiStep-withReview | `--review-mode --with-review` | Multi-step delivery with review injected |

### 1.2 Ablation Studies

| Study | Description |
|---|---|
| Prompt Sensitivity | Five prompt variants (Base / Merged / NoChk. / Gen. / Min.) on a 10% stratified sample |
| Mismatch Review | Same images paired with a review from a **different** product category |

---

## ⚡ Requirements

```
python >= 3.10
pillow        # required for xAI HTTP-413 image downscale retry
pandas
openpyxl      # required for Excel export
flask         # required for the human evaluation interface
```

Install dependencies:

```bash
pip install pillow pandas openpyxl flask
```

---

## 🔑 API Keys

Set the following environment variables before running any script. Models whose key is absent are automatically skipped.

```bash
export DASHSCOPE_API_KEY_1="..."   # qvq-max-latest, qwen3-vl-plus
export DASHSCOPE_API_KEY_2="..."   # qwen3.6-plus, kimi-k2.6, qwen3-vl-flash
export DASHSCOPE_API_KEY_3="..."   # qwen3.6-flash, qwen3.5-omni-plus
export XAI_API_KEY="..."           # grok-4-1-fast-reasoning, grok-4.20-reasoning-latest
export GEMINI_API_KEY="..."        # gemini-3-flash
export OPENAI_API_KEY="..."        # gpt-5.4-mini
```

---

## 📂 Script Reference

**`scripts/`** — Shell runners (entry points for reviewers)

| Script | Role |
|---|---|
| `scripts/run_detect.sh` | **Unified runner** for all six main experiment conditions |
| `scripts/run_ablation.sh` | **Unified runner** for both ablation studies (incl. index generation) |

**`tools/`** — Python inference and analysis scripts

| Script | Role |
|---|---|
| `tools/detect.py` | Unified detection script for all six experiment conditions |
| `tools/detect_prompt_ablation.py` | Prompt ablation detection (reads from sample index) |
| `tools/detect_mismatch.py` | Mismatch-review detection (reads from mismatch index) |
| `tools/generate_sample_index.py` | Generates the stratified 10% sample index for ablation |
| `tools/generate_mismatch_index.py` | Generates the cross-category mismatch index |
| `tools/compute_accuracy.py` | Computes macro-averaged metrics from main experiment results |
| `tools/compute_ablation_accuracy.py` | Computes macro-averaged metrics for prompt ablation variants |

**`human_eval_interface/`** — Local web app for blind human evaluation

| File | Role |
|---|---|
| `human_eval_interface/app.py` | Flask server — serves images and records judgements |
| `human_eval_interface/catalog.py` | Scans dataset directories and builds the image registry |
| `human_eval_interface/sampler.py` | Balanced real/fake sampler with anti-recency logic |
| `human_eval_interface/store.py` | Atomic per-evaluator result storage with undo support |

---

## 🧪 Usage

### 5.1 Main Experiments

```bash
# SingleImage-NoReview (default)
bash scripts/run_detect.sh

# SingleImage-withReview
bash scripts/run_detect.sh --with-review

# MultiImage-NoReview
bash scripts/run_detect.sh --review-mode --single-turn

# MultiImage-withReview
bash scripts/run_detect.sh --review-mode --single-turn --with-review

# MultiStep-NoReview
bash scripts/run_detect.sh --review-mode

# MultiStep-withReview
bash scripts/run_detect.sh --review-mode --with-review

# Override concurrency (default 4)
bash scripts/run_detect.sh --concurrency 6
```

Results are written to:

```
{category}/Results/{MODE}/summary.json
```

### 5.2 Ablation Studies

```bash
# Run both ablation studies (default)
bash scripts/run_ablation.sh

# Prompt sensitivity only
bash scripts/run_ablation.sh --prompt

# Mismatch review only
bash scripts/run_ablation.sh --mismatch

# Force regeneration of indices
bash scripts/run_ablation.sh --regen-sample
bash scripts/run_ablation.sh --regen-mismatch

# Custom options
bash scripts/run_ablation.sh --concurrency 6 --sample-ratio 0.1 --sample-seed 42
```

Both studies are **resumable**: re-running skips already-completed `(image, model)` pairs. Indices are generated automatically if they do not exist.

Results are written to:

```
PromptAblation/{variant}/{category}/summary.json
MismatchReview/{category}/summary.json
```

### 5.3 Human Evaluation Interface

A lightweight Flask web app for blind image-level human evaluation (Real vs. DeepFake).

```bash
# Start the server (default: dataset root inferred from script location)
python human_eval_interface/app.py

# Specify dataset root and/or custom results directory explicitly
python human_eval_interface/app.py \
    --root /path/to/dataset \
    --results-root human_eval_interface/Results

# Custom host/port (default: 127.0.0.1:5050)
python human_eval_interface/app.py --host 0.0.0.0 --port 8080
```

Open `http://127.0.0.1:5050/` in a browser. Enter an evaluator name and select a category scope (a specific category or "All"). Images are served one at a time in a balanced real/fake order; press **F / ←** for Real, **J / →** for DeepFake, **U** to undo the last judgement.

Each session is **resumable**: restarting the server and re-entering the same name resumes from where it left off.

Results are written atomically to:

```
human_eval_interface/Results/{evaluator}/{scope}/
├── Negative.json
├── DeepFake/{gen_model}.json
├── summary.json          ← written on completion
└── _progress.json        ← incremental progress checkpoint
```

---

## 📊 Output Structure

```
{category}/
└── Results/
    ├── SingleImage-NoReview/
    │   ├── Negative/<model>.json
    │   ├── DeepFake/<generator>/<model>.json
    │   └── summary.json               ← cross-model roll-up, one row per image
    ├── SingleImage-withReview/        ...
    ├── MultiImage-NoReview/           ...
    ├── MultiImage-withReview/         ...
    ├── MultiStep-NoReview/            ...
    └── MultiStep-withReview/          ...

PromptAblation/
├── sample_index.json               ← fixed 10% sample index, generated once
├── v1_baseline/{category}/summary.json
├── v2_merged_role/...
├── v3_no_artifacts/...
├── v4_generic_role/...
└── v5_minimal/...

MismatchReview/
├── mismatch_index.json             ← cross-category pairings index, generated once
├── {category}/summary.json
└── logs/{category}.log

human_eval_interface/Results/
└── {evaluator}/
    └── {scope}/
        ├── Negative.json
        ├── DeepFake/{gen_model}.json
        ├── summary.json            ← written on session completion
        └── _progress.json          ← incremental checkpoint (resumable)
```

Each MLLM `summary.json` contains a `rows` list with one entry per image/review, each row holding a `verdicts` dict keyed by model name with fields: `status`, `is_ai_modified`, `confidence`, `reason`, `error`.

---

## 📈 Computing Metrics

```bash
# Main experiment results → accuracy_results.xlsx
python tools/compute_accuracy.py

# Prompt ablation results → ablation_results.xlsx
python tools/compute_ablation_accuracy.py
```

Both scripts compute **macro-averaged** metrics over product categories:

- **TNR** — True Negative Rate on real damaged images
- **TPR** — per-generator True Positive Rate on AI-modified images
- **F1** — binary F1 on the fake class (macro-averaged over categories)
- **Bal.Acc** — balanced accuracy = (TPR + TNR) / 2, pooled across generators
- **Conf.** — mean confidence score on **correctly classified** images only

---

## 🤖 Models

| Model | Provider | API Key Env Var |
|---|---|---|
| `qvq-max-latest` | Alibaba DashScope | `DASHSCOPE_API_KEY_1` |
| `qwen3-vl-plus` | Alibaba DashScope | `DASHSCOPE_API_KEY_1` |
| `qwen3.6-plus` | Alibaba DashScope | `DASHSCOPE_API_KEY_2` |
| `kimi-k2.6` | Moonshot / DashScope (used) | `DASHSCOPE_API_KEY_2` |
| `qwen3-vl-flash` | Alibaba DashScope | `DASHSCOPE_API_KEY_2` |
| `qwen3.6-flash` | Alibaba DashScope | `DASHSCOPE_API_KEY_3` |
| `qwen3.5-omni-plus` | Alibaba DashScope | `DASHSCOPE_API_KEY_3` |
| `grok-4-1-fast-reasoning` | xAI | `XAI_API_KEY` |
| `grok-4.20-reasoning-latest` | xAI | `XAI_API_KEY` |
| `gemini-3-flash` | Google | `GEMINI_API_KEY` |
| `gpt-5.4-mini` | OpenAI | `OPENAI_API_KEY` |

All reasoning/thinking parameters are left at vendor defaults (no overrides). For xAI models, HTTP 413 (payload too large) triggers automatic image downscaling through up to 10 progressive resolution tiers before the request is abandoned.

---

## 🔎 Reproducibility Notes

- The stratified sample for ablation studies uses **ratio = 0.1, seed = 42**.
- The cross-category review assignment for mismatch review uses **seed = 42**.
- Passing `--regen-sample` or `--regen-mismatch` rebuilds indices from scratch; omitting these flags reuses existing indices so every run tests the exact same image set.
- All results are written atomically (temp-file + rename) and are safe to interrupt and resume.

---

## 📄 Citation

If you find this work useful, please kindly consider citing our paper:

```bibtex
@misc{yan2026fraudbenchmultimodalbenchmarkdetecting,
      title={FraudBench: A Multimodal Benchmark for Detecting AI-Generated Fraudulent Refund Evidence},
      author={Xinyu Yan and Boyang Chen and Jiaming Zhang and Tiantong Wu and Hong Xi Tae and Yichen He and Tiantong Wang and Yachun Mi and Yurong Hao and Yilei Zhao and Lei Xiao and Longtao Huang and Pengjun Xie and Wei Liu and Wei Yang Bryan Lim},
      year={2026},
      eprint={2605.08820},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.08820},
}
```
