from __future__ import annotations

import argparse
import logging
import random
import uuid
from collections import defaultdict
from datetime import UTC, datetime

from src.common import DATA_DIR, PROMPTS_DIR, bootstrap, normalize_cli_list, read_jsonl, write_jsonl

bootstrap()

VARIANTS_PATH = DATA_DIR / 'variants.jsonl'
SESSIONS_PATH = DATA_DIR / 'sessions.jsonl'

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def existing_prompt_ids() -> set[str]:
    ids = set()
    for path in PROMPTS_DIR.glob('evaluate_*.txt'):
        ids.add(path.stem.replace('evaluate_', '', 1))
    return ids


def validate_prompt_ids(prompt_ids: list[str]) -> None:
    available = existing_prompt_ids()
    missing = [p for p in prompt_ids if p not in available]
    if missing:
        raise ValueError(f'Missing prompt files for ids {missing}. Available: {sorted(available)}')


def dedupe_sessions(rows: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    deduped: list[dict] = []

    for row in rows:
        key = (
            row['matched_group_id'],
            row['prompt_id'],
            row['replicate_index'],
            row['scenario'],
            tuple(sorted(row['cv_ids'])),
            tuple(row['display_order']),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(row)

    return deduped


def build_sessions(prompt_ids: list[str], replicates: int = 3) -> list[dict]:
    random.seed(42)

    variants = read_jsonl(VARIANTS_PATH)
    if not variants:
        raise ValueError('No variants found. Generate variants first.')

    validate_prompt_ids(prompt_ids)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in variants:
        grouped[row['matched_group_id']].append(row)

    sessions: list[dict] = []
    for matched_group_id, members in grouped.items():
        if len(members) != 2:
            logging.warning(
                'Skipping matched_group_id=%s because it has %d members instead of 2',
                matched_group_id,
                len(members),
            )
            continue

        first = members[0]

        for prompt_id in prompt_ids:
            for replicate_index in range(1, replicates + 1):
                shown = members[:]
                random.shuffle(shown)

                sessions.append(
                    {
                        'session_id': f"sess_{uuid.uuid4().hex[:12]}",
                        'matched_group_id': matched_group_id,
                        'template_id': first['template_id'],
                        'scenario': first.get('scenario', 'single_attribute'),
                        'job_title_target': first['job_title_target'],
                        'pool_size': 2,
                        'cv_ids': [shown[0]['cv_id'], shown[1]['cv_id']],
                        'display_order': [shown[0]['cv_id'], shown[1]['cv_id']],
                        'bias_variable': first['bias_variable'],
                        'variant_values': [shown[0]['variant_value'], shown[1]['variant_value']],
                        'language': first['language'],
                        'prompt_id': prompt_id,
                        'replicate_index': replicate_index,
                        'created_at': utc_now_iso(),
                    }
                )

    return dedupe_sessions(sessions)


def main() -> None:
    parser = argparse.ArgumentParser(description='Build evaluation sessions from generated variants')
    parser.add_argument('--prompts', nargs='+', default=['neutral_v1', 'strict_merit_v1', 'role_specific_v1'])
    parser.add_argument('--replicates', type=int, default=3)
    args = parser.parse_args()

    prompt_ids = normalize_cli_list(args.prompts, default=['neutral_v1', 'strict_merit_v1', 'role_specific_v1'])
    sessions = build_sessions(prompt_ids=prompt_ids, replicates=args.replicates)
    write_jsonl(SESSIONS_PATH, sessions)
    logging.info('Built %d sessions', len(sessions))


if __name__ == '__main__':
    main()