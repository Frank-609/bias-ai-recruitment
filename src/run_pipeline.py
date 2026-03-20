from __future__ import annotations

import argparse
import asyncio
import logging

from src.analysis import main as analysis_main
from src.generator import run_generate_templates, run_generate_variants
from src.session_builder import build_sessions, SESSIONS_PATH
from src.common import append_jsonl, bootstrap, normalize_cli_list, reset_file
from src.evaluator import run_evaluation

bootstrap()
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the full recruitment-bias experiment pipeline')
    parser.add_argument('--job', required=True)
    parser.add_argument('--languages', nargs='+', default=['English'])
    parser.add_argument('--template-count', type=int, default=5)
    parser.add_argument('--provider', choices=['openai', 'mistral'], default='openai')
    parser.add_argument('--model', default=None)
    parser.add_argument('--scenario', choices=['single_attribute', 'intersectional'], default='single_attribute')
    parser.add_argument('--prompts', nargs='+', default=['neutral_v1', 'strict_merit_v1', 'role_specific_v1'])
    parser.add_argument('--replicates', type=int, default=3)
    parser.add_argument('--concurrency', type=int, default=5)
    parser.add_argument('--reset', action='store_true')
    parser.add_argument('--blind', action='store_true')
    args = parser.parse_args()

    languages = normalize_cli_list(args.languages, default=['English'])
    prompt_ids = normalize_cli_list(args.prompts, default=['neutral_v1', 'strict_merit_v1', 'role_specific_v1'])

    asyncio.run(run_generate_templates(job=args.job, languages=languages, count=args.template_count, provider=args.provider, model=args.model, reset=args.reset))
    asyncio.run(run_generate_variants(provider=args.provider, model=args.model, scenario=args.scenario, reset=args.reset))

    sessions = build_sessions(prompt_ids=prompt_ids, replicates=args.replicates)
    if args.reset:
        reset_file(SESSIONS_PATH)
    append_jsonl(SESSIONS_PATH, sessions)
    logging.info('Built %d sessions', len(sessions))

    asyncio.run(run_evaluation(provider=args.provider, model=args.model, concurrency=args.concurrency, include_personal_signals=not args.blind, skip_existing=not args.reset))
    analysis_main()


if __name__ == '__main__':
    main()
