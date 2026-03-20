# Bias in Generative AI — Recruitment Study

Master 1 AI, JUNIA ISEN Lille, 2025–2026.

This repository implements a reproducible pipeline for studying bias in LLM-based recruitment decisions with controlled synthetic CV comparisons. The code now supports end-to-end execution, environment loading, multiple evaluation prompts, multilingual template generation, optional intersectional experiments, duplicate-safe session building, result skipping on reruns, aggregated metrics, and a one-command orchestration script.

## What the pipeline does

1. Generates synthetic base CV templates for a target job.
2. Produces matched CV variants where only one controlled variable changes, or optionally intersectional pairs where two variables are combined.
3. Builds evaluation sessions with shuffled candidate order and multiple prompt formulations.
4. Sends those sessions to an LLM evaluator such as OpenAI or Mistral.
5. Saves raw responses, parsed responses, flat results, figures, and aggregate metrics.
6. Optionally renders any generated CV as a PDF preview.

## Implemented experiment coverage

- Single-attribute bias comparisons
- Intersectional bias comparisons
- Multi-prompt evaluation using separate prompt files
- English and French or other multi-language template generation
- Multi-provider evaluation
- Blind and non-blind prompt modes during evaluation

## Project structure

```text
bias-ai-recruitment/
├─ data/
│  ├─ templates.jsonl
│  ├─ variants.jsonl
│  └─ sessions.jsonl
├─ outputs/
│  ├─ figures/
│  ├─ parsed/
│  ├─ preview_pdfs/
│  ├─ raw/
│  ├─ metrics.json
│  └─ results.jsonl
├─ prompts/
│  ├─ evaluate_neutral_v1.txt
│  ├─ evaluate_role_specific_v1.txt
│  ├─ evaluate_strict_merit_v1.txt
│  ├─ generate_template.txt
│  └─ generate_variant.txt
├─ src/
│  ├─ analysis.py
│  ├─ common.py
│  ├─ evaluator.py
│  ├─ generator.py
│  ├─ llm_client.py
│  ├─ preview.py
│  ├─ run_pipeline.py
│  └─ session_builder.py
├─ .env.example
├─ requirements.txt
└─ README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows PowerShell
pip install -r requirements.txt
cp .env.example .env
```

Then edit `.env` and add at least one provider key:

```env
OPENAI_API_KEY=your_key_here
MISTRAL_API_KEY=your_key_here
```

## Recommended quick start

Run the whole experiment in one command:

```bash
python -m src.run_pipeline \
  --job "Data Analyst" \
  --languages English French \
  --template-count 5 \
  --provider openai \
  --model gpt-4o-mini \
  --scenario single_attribute \
  --prompts neutral_v1 strict_merit_v1 role_specific_v1 \
  --replicates 3 \
  --concurrency 5 \
  --reset
```

This will generate templates, build variants, create sessions, evaluate them, and run the analysis.

## Manual step-by-step commands

Generate templates:

```bash
python -m src.generator templates \
  --job "Data Analyst" \
  --language English French \
  --count 5 \
  --provider openai \
  --model gpt-4o-mini \
  --reset
```

Generate single-attribute variants:

```bash
python -m src.generator variants \
  --provider openai \
  --model gpt-4o-mini \
  --scenario single_attribute \
  --reset
```

Generate intersectional variants instead:

```bash
python -m src.generator variants \
  --provider openai \
  --model gpt-4o-mini \
  --scenario intersectional
```

Build sessions:

```bash
python -m src.session_builder \
  --prompts neutral_v1 strict_merit_v1 role_specific_v1 \
  --replicates 3 \
  --reset
```

Evaluate sessions:

```bash
python -m src.evaluator \
  --provider openai \
  --model gpt-4o-mini \
  --concurrency 5
```

Run a blind ablation that hides name and age from the prompt:

```bash
python -m src.evaluator \
  --provider openai \
  --model gpt-4o-mini \
  --concurrency 5 \
  --blind
```

Analyze results:

```bash
python -m src.analysis
```

Preview one generated CV as PDF:

```bash
python -m src.preview --random
python -m src.preview --cv-id cv_xxxxxxxxxxxx
python -m src.preview --random --anonymized
```

## Methodological note

By default, evaluation includes name and age because the project is designed to test whether those visible signals affect the model's choice. For an ablation that removes those signals from the prompt, use `--blind` in `src.evaluator`.

## Outputs

- `data/templates.jsonl`: generated base profiles
- `data/variants.jsonl`: controlled candidate variants
- `data/sessions.jsonl`: pairwise evaluation sessions
- `outputs/raw/`: raw model responses per session
- `outputs/parsed/`: parsed model responses per session
- `outputs/results.jsonl`: flat evaluation results
- `outputs/metrics.json`: aggregated metrics, including dominant variant rate and a simple exact binomial p-value
- `outputs/figures/`: bar charts for counts and win rates
- `outputs/preview_pdfs/`: PDF previews of generated CVs

## Notes

- All candidate data is synthetic.
- Re-running the evaluator skips sessions already evaluated for the same provider and model, unless `--no-skip-existing` is used.
- Re-running session building with the same inputs avoids duplicate session definitions.
- Environment variables are loaded automatically from `.env`.
