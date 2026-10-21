# FinRCR

Retrieval-Augmented Counterfactual Reasoning for financial risk prediction on [FinBench](https://huggingface.co/datasets/yuweiyin/FinBench).

FinRCR combines a tabular risk model with risk evidence cards, contrastive retrieval of risky and safe cases, counterfactual reasoning, uncertainty-aware fusion, and optional local LLM inference. This repository contains source code and runnable configuration only. Generated predictions, metrics, tables, figures, model checkpoints, paper PDFs, and other experimental results are intentionally excluded.

## Repository layout

```text
finrcr-github/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── src/
│   ├── finrcr_experiment.py
│   ├── finbench_common.py
│   └── finbench_tabular_baseline.py
├── scripts/
│   └── run_finrcr_all_tasks.py
└── docs/
    └── method.md
```

## Environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset

The code can download the required split files from the FinBench Hugging Face dataset on first use. To use a local copy instead, place the dataset under `datasets/finbench_raw/data/<task>` or pass `--data-root` to the runner. The expected task names are `cd1`, `cd2`, `ld1`, `ld2`, `ld3`, `cf1`, `cf2`, `cc1`, `cc2`, and `cc3`.

The expected local task directory contains the NumPy splits, labels, metadata, and profile JSONL files distributed with FinBench. Do not commit the dataset or generated outputs to this repository.

## Run one task

Run the surrogate reasoning channel without an external LLM:

```bash
python src/finrcr_experiment.py \
  --ds-name cd1 \
  --reasoner surrogate \
  --top-k-evidence 3 \
  --risk-cases 1 \
  --safe-cases 1 \
  --max-memory-cases 500 \
  --output-dir outputs/finrcr_cd1
```

The command writes predictions and run metadata to the selected output directory. These generated files are ignored by Git.

## Use a local OpenAI-compatible LLM server

Start an OpenAI-compatible local server, then run:

```bash
python src/finrcr_experiment.py \
  --ds-name cd1 \
  --reasoner api \
  --api-base http://127.0.0.1:1234 \
  --api-model qwen/qwen3.5-9b \
  --top-k-evidence 3 \
  --risk-cases 1 \
  --safe-cases 1 \
  --max-memory-cases 500 \
  --llm-max-tokens 3072 \
  --output-dir outputs/finrcr_cd1_qwen
```

The API channel expects `/v1/chat/completions` and returns a JSON-compatible response. Use `--llm-policy uncertain` or `--llm-budget` to limit LLM calls during development.

## Run all tasks

```bash
python scripts/run_finrcr_all_tasks.py \
  --reasoner surrogate \
  --output-root outputs/finrcr_all
```

For a local LLM server, add `--api-base` and `--api-model`. Use `--max-eval` for a smoke test before launching a full run.



Supported baseline modes are `logistic_regression`, `random_forest`, and `hist_gradient_boosting`.



## Citation

If you use this code, please contaxt us. Dataset access is provided through the Hugging Face repository linked above.
