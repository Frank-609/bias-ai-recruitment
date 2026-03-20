from __future__ import annotations

import json
import logging
import math
from collections import defaultdict

import matplotlib.pyplot as plt

from src.common import OUTPUTS_DIR, bootstrap, read_jsonl

bootstrap()

RESULTS_PATH = OUTPUTS_DIR / 'results.jsonl'
METRICS_PATH = OUTPUTS_DIR / 'metrics.json'
FIGURES_DIR = OUTPUTS_DIR / 'figures'

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def extract_pair_metrics(row: dict) -> dict | None:
    if not row.get('parse_ok'):
        return None
    parsed = row.get('parsed_response')
    if not isinstance(parsed, dict):
        return None
    scores = parsed.get('scores')
    if not isinstance(scores, dict):
        return None
    variant_values = row.get('variant_values', [])
    if len(variant_values) != 2:
        return None
    v1, v2 = variant_values
    s1, s2 = safe_float(scores.get('1')), safe_float(scores.get('2'))
    winner = None
    if str(parsed.get('top_pick')) == '1':
        winner = v1
    elif str(parsed.get('top_pick')) == '2':
        winner = v2
    score_delta = None if s1 is None or s2 is None else s1 - s2
    return {
        'provider': row.get('provider'), 'model': row.get('model'), 'prompt_id': row.get('prompt_id'), 'scenario': row.get('scenario', 'single_attribute'),
        'bias_variable': row.get('bias_variable'), 'variant_1': v1, 'variant_2': v2, 'winner': winner, 'score_delta_v1_minus_v2': score_delta,
    }


def binomial_two_sided_pvalue(k: int, n: int) -> float | None:
    if n <= 0:
        return None
    # exact two-sided p-value around p=0.5 without scipy
    probs = []
    observed = math.comb(n, k) * (0.5 ** n)
    for i in range(n + 1):
        p = math.comb(n, i) * (0.5 ** n)
        if p <= observed + 1e-12:
            probs.append(p)
    return min(1.0, sum(probs))


def aggregate_metrics(results: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in results:
        pair = extract_pair_metrics(row)
        if pair is None:
            continue
        key = (pair['provider'], pair['model'], pair['prompt_id'], pair['scenario'], pair['bias_variable'])
        grouped[key].append(pair)

    summaries = []
    for (provider, model, prompt_id, scenario, bias_variable), rows in grouped.items():
        win_counts = defaultdict(int)
        deltas = []
        ordered_pair_counts = defaultdict(int)
        for row in rows:
            if row['winner'] is not None:
                win_counts[row['winner']] += 1
            if row['score_delta_v1_minus_v2'] is not None:
                deltas.append(row['score_delta_v1_minus_v2'])
            ordered_pair_counts[f"{row['variant_1']}__vs__{row['variant_2']}"] += 1
        total_decisions = sum(win_counts.values())
        win_rates = {k: v / total_decisions for k, v in win_counts.items()} if total_decisions else {}
        dominant_variant = max(win_counts, key=win_counts.get) if win_counts else None
        dominant_rate = (win_counts[dominant_variant] / total_decisions) if dominant_variant and total_decisions else None
        summaries.append({
            'provider': provider,
            'model': model,
            'prompt_id': prompt_id,
            'scenario': scenario,
            'bias_variable': bias_variable,
            'n_sessions': len(rows),
            'win_counts': dict(win_counts),
            'win_rates': win_rates,
            'mean_score_delta_v1_minus_v2': (sum(deltas) / len(deltas)) if deltas else None,
            'mean_abs_score_delta': (sum(abs(x) for x in deltas) / len(deltas)) if deltas else None,
            'pair_comparisons': dict(ordered_pair_counts),
            'dominant_variant': dominant_variant,
            'dominant_rate': dominant_rate,
            'dominant_variant_binomial_pvalue': binomial_two_sided_pvalue(win_counts.get(dominant_variant, 0), total_decisions) if dominant_variant else None,
        })
    return summaries


def plot_bar(metric: dict, values: dict, ylabel: str, title_prefix: str, filename_prefix: str, ylim: tuple[float, float] | None = None) -> None:
    if not values:
        return
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    labels = list(values.keys())
    heights = list(values.values())
    plt.figure(figsize=(8, 5))
    plt.bar(labels, heights)
    plt.title(f"{title_prefix}\n{metric['provider']} | {metric['scenario']} | {metric['bias_variable']} | {metric['prompt_id']}")
    plt.xlabel('Variant value')
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xticks(rotation=20)
    plt.tight_layout()
    out_path = FIGURES_DIR / f"{filename_prefix}_{metric['provider']}_{metric['scenario']}_{metric['bias_variable']}_{metric['prompt_id']}.png"
    plt.savefig(out_path)
    plt.close()


def print_summary(metrics: list[dict]) -> None:
    for metric in metrics:
        logging.info('provider=%s | model=%s | scenario=%s | prompt=%s | bias=%s | n=%d | dominant=%s | p=%s | mean_abs_delta=%s', metric['provider'], metric['model'], metric['scenario'], metric['prompt_id'], metric['bias_variable'], metric['n_sessions'], metric['dominant_variant'], metric['dominant_variant_binomial_pvalue'], metric['mean_abs_score_delta'])


def main() -> None:
    results = read_jsonl(RESULTS_PATH)
    if not results:
        raise ValueError('No results found in outputs/results.jsonl')
    metrics = aggregate_metrics(results)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    for metric in metrics:
        plot_bar(metric, metric.get('win_counts', {}), 'Top-pick count', 'Top-pick counts', 'wins')
        plot_bar(metric, metric.get('win_rates', {}), 'Win rate', 'Top-pick rates', 'winrates', ylim=(0, 1))
    print_summary(metrics)
    logging.info('Analysis complete. Saved metrics to %s', METRICS_PATH)
    logging.info('Figures saved to %s', FIGURES_DIR)


if __name__ == '__main__':
    main()
