# [NeurIPS 2026] Masked Diffusion Language Agents for Tool-Integrated Chemical Reasoning

This is the official repository for:

> **[NeurIPS 2026] Masked Diffusion Language Agents for Tool-Integrated Chemical Reasoning**

Accepted to **NeurIPS 2026**.

## 1. Repository Structure

The repository contains the training and inference code used by ChemDiffAgent.

```text
ChemDiffAgent/
├── inference/          # ChemDiffAgent inference
├── scripts/            # Training and inference launchers
└── training/
    ├── sft/            # Agentic SFT
    └── vrpo/           # Agentic VRPO
```

## 2. Environment Setup

Create a Python environment and install the dependencies required for training and inference.

```bash
git clone https://github.com/bilemond/ChemDiffAgent.git
cd ChemDiffAgent

conda create -n chemdiffagent python=3.10 -y
conda activate chemdiffagent

pip install -r inference/requirements.txt
pip install -r training/sft/requirements.txt
pip install -r training/vrpo/requirements.txt
```

Optional chemistry backends can be installed with:

```bash
bash scripts/setup_tools.sh
```

## 3. Dataset

Training data and inference benchmarks are available at [Serendipity001/ChemDiffAgent](https://huggingface.co/datasets/Serendipity001/ChemDiffAgent).

```bash
pip install -U huggingface_hub
hf download Serendipity001/ChemDiffAgent \
  --repo-type dataset \
  --local-dir data/ChemDiffAgent
```

## 4. Training

Train ChemDiffAgent using Agentic SFT followed by Agentic VRPO.

Run Agentic SFT with the downloaded `training/train.json`:

```bash
export BASE_MODEL=JetLM/SDAR-8B-Chat
export SFT_DATA=$PWD/data/ChemDiffAgent/training/train.json
export OUTPUT_DIR=$PWD/outputs/chemdiffagent-sft

bash scripts/train_sft.sh
```

Run Agentic VRPO with execution-verified preference data prepared from `training/rl_tasks.jsonl`:

```bash
export SFT_MODEL=$PWD/outputs/chemdiffagent-sft
export PREFERENCE_DATA=/path/to/preferences.jsonl
export OUTPUT_DIR=$PWD/outputs/chemdiffagent-vrpo

bash scripts/train_vrpo.sh
```

## 5. Inference

Run ChemDiffAgent on the downloaded single-turn and multi-turn benchmarks.

```bash
export MODEL_PATH=/path/to/your/checkpoint
export DATA_DIR=$PWD/data/ChemDiffAgent/benchmark
export OUTPUT_DIR=$PWD/outputs/inference

bash scripts/run_inference.sh
```
