# Bias in Generative AI — Recruitment Study

Master 1 AI, JUNIA ISEN Lille, 2025–2026

## Overview
This project builds a controlled experimental pipeline to study bias in large language models (LLMs) when they perform recruitment tasks.

We generate realistic synthetic CVs and create matched variants where only one attribute (such as gender, name origin, age, career gap, or hobbies) is modified while all other qualifications remain unchanged. These CVs are then evaluated by LLMs such as OpenAI and Mistral to observe how decisions vary under controlled conditions.

The objective is to measure and analyze systematic differences in model behavior and identify potential bias signals.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate        # Linux / Mac
.venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

Create environment variables:
```bash
cp .env.example .env
```

Then add your API keys:
```
OPENAI_API_KEY=your_key_here
MISTRAL_API_KEY=your_key_here
```

## Quick Start
Run the full pipeline:
```bash
python run_pipeline.py
```

## Pipeline Steps
Generate base CV templates:
```bash
python -m src.generator templates --job "Data Analyst" --count 10
```

Generate controlled variants:
```bash
python -m src.generator variants
```

Build evaluation sessions:
```bash
python -m src.session_builder
```

Evaluate with LLMs:
```bash
python -m src.evaluator --provider openai
python -m src.evaluator --provider mistral
```

Analyze results:
```bash
python -m src.analysis
```

## Outputs
- outputs/results.jsonl → raw evaluation results  
- outputs/parsed/ → structured model outputs  
- outputs/figures/ → visualizations  
- outputs/metrics.json → aggregated metrics  

## Project Structure
```
data/       generated datasets (templates, variants, sessions)
outputs/    evaluation outputs and figures
prompts/    LLM prompts
src/        core pipeline code
```

## Method
The pipeline follows a controlled comparison approach:
- Start from a base CV  
- Create variants where only one variable changes  
- Evaluate candidates with LLMs  
- Compare outcomes across conditions  

This design allows differences in model decisions to be attributed to specific variables rather than unrelated factors.

## Notes
- All data is synthetic (no real personal data)  
- Results depend on model behavior and prompts  
- This project is for research purposes only  