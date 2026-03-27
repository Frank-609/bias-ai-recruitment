# Bias in Generative AI — Recruitment Study

Master 1 AI, JUNIA ISEN Lille, 2025–2026

## Overview

This project studies bias in large language models when they are used to evaluate job applicants in recruitment-like settings.

The pipeline generates synthetic CV templates, creates controlled matched variants, builds evaluation sessions, asks LLMs to compare candidates, and analyzes whether model decisions systematically differ across bias-relevant attributes such as gender signals, name origin, career gaps, age signals, or hobbies.

The core experimental principle is controlled comparison: within each matched pair, only one target attribute should change while the rest of the candidate profile remains as similar as possible. This makes it easier to interpret whether differences in model judgments are linked to the manipulated variable rather than to unrelated differences in qualifications.

## Pipeline Structure

The project follows five main stages:

1. Generate synthetic base CV templates
2. Create controlled variants from those templates
3. Build evaluation sessions from matched candidate pairs
4. Evaluate sessions with an LLM
5. Aggregate and visualize the results

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file at the project root and add the API keys you need:
OPENAI_API_KEY=your_openai_key_here
MISTRAL_API_KEY=your_mistral_key_here


You only need the key for the provider you plan to use.

## Full Pipeline

Run the full pipeline for one job:

```bash
python -m src.run_pipeline \
  --job "Data Analyst" \
  --languages English \
  --template-count 10 \
  --provider openai \
  --model gpt-4o-mini \
  --scenario single_attribute \
  --prompts neutral_v1 strict_merit_v1 role_specific_v1 \
  --replicates 3 \
  --concurrency 5 \
  --reset
```

Run the full pipeline for multiple jobs:

```bash
python -m src.run_pipeline \
  --jobs "Data Analyst" "Software Engineer" \
  --languages English \
  --template-count 10 \
  --provider openai \
  --model gpt-4o-mini \
  --scenario single_attribute \
  --prompts neutral_v1 strict_merit_v1 role_specific_v1 \
  --replicates 3 \
  --concurrency 5 \
  --reset
```

## Generation-Only Workflow

Generate templates and variants without evaluating yet:

```bash
python -m src.run_pipeline \
  --job "Data Analyst" \
  --languages English \
  --template-count 10 \
  --provider openai \
  --model gpt-4o-mini \
  --scenario single_attribute \
  --prompts neutral_v1 strict_merit_v1 role_specific_v1 \
  --replicates 3 \
  --concurrency 5 \
  --reset \
  --skip-evaluation \
  --skip-analysis
```

This is useful when you want to inspect generated CVs before spending API calls on evaluation.

## Stage-by-Stage Commands

### 1) Generate templates

```bash
python -m src.generator templates \
  --job "Data Analyst" \
  --language English \
  --count 10 \
  --provider openai \
  --model gpt-4o-mini \
  --reset
```

### 2) Generate variants

```bash
python -m src.generator variants \
  --provider openai \
  --model gpt-4o-mini \
  --scenario single_attribute \
  --reset
```

### 3) Build sessions

```bash
python -m src.session_builder \
  --prompts neutral_v1 strict_merit_v1 role_specific_v1 \
  --replicates 3
```

### 4) Evaluate sessions

```bash
python -m src.evaluator \
  --provider openai \
  --model gpt-4o-mini \
  --concurrency 5
```

Blind evaluation mode hides name and age:

```bash
python -m src.evaluator \
  --provider openai \
  --model gpt-4o-mini \
  --concurrency 5 \
  --blind
```

### 5) Run analysis

```bash
python -m src.analysis
```

## Preview Utilities

There are two different preview tools:

**CV preview** — renders a single candidate CV as a PDF and saves it to `outputs/preview_pdfs/`:

```bash
python -m src.preview --cv-id cv_xxxxxxxxxxxx
python -m src.preview --random
```

**Session result preview** — renders a compact session report PDF from an evaluated result and saves it to `outputs/session_previews/`:

```bash
python -m src.session_preview --session-id sess_xxxxxxxxxxxx
python -m src.session_preview --random
```

> `src.session_preview` reads from `outputs/results.jsonl`, so it is meant for previewing evaluated session results, not raw unevaluated sessions.

## Reset Behavior

Using `--reset` in the main pipeline clears the experimental state before starting again. That includes:

- `data/templates.jsonl`
- `data/variants.jsonl`
- `data/sessions.jsonl`
- `outputs/results.jsonl`
- `outputs/raw/`
- `outputs/parsed/`
- `outputs/figures/`
- `outputs/preview_pdfs/`
- `outputs/session_previews/`
- `outputs/metrics.json`

This prevents stale outputs from old runs from contaminating the current experiment.

## Output Files

| File | Description |
|---|---|
| `data/templates.jsonl` | Generated base CV templates |
| `data/variants.jsonl` | Controlled CV variants |
| `data/sessions.jsonl` | Evaluation sessions built from matched pairs |
| `outputs/results.jsonl` | Evaluation results |
| `outputs/raw/` | Raw model outputs saved per session |
| `outputs/parsed/` | Parsed JSON outputs saved per session |
| `outputs/figures/` | Charts generated by the analysis step |
| `outputs/preview_pdfs/` | PDFs for single-CV previews |
| `outputs/session_previews/` | PDFs for evaluated session result previews |
| `outputs/metrics.json` | Aggregated metrics |

## Evaluation Prompts

The repository includes several evaluation prompt variants:

- `neutral_v1`
- `strict_merit_v1`
- `role_specific_v1`

These are not redundant. They are experimental conditions. The goal is to test whether model behavior changes depending on instruction framing, rubric strictness, or role-specific emphasis.

## Notes

- All candidate data is synthetic
- The project is intended for research and analysis, not real hiring
- Prompt wording can materially affect evaluation outcomes
- A clean reset is important for reproducible experiments
- CV preview and session result preview are intentionally separate tools with separate output folders