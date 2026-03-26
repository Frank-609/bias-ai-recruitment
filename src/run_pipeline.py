from __future__ import annotations

import argparse
import asyncio
import logging

from src.analysis import main as analysis_main
from src.common import bootstrap, normalize_cli_list, reset_pipeline_state, write_jsonl
from src.evaluator import run_evaluation
from src.generator import run_generate_templates, run_generate_variants
from src.session_builder import SESSIONS_PATH, build_sessions

bootstrap()
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')


def parse_jobs(job: str | None, jobs: list[str] | None) -> list[str]:
    values: list[str] = []
    if job:
        values.append(job)
    if jobs:
        values.extend(normalize_cli_list(jobs, default=[]))

    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


async def run_pipeline(
    jobs: list[str],
    languages: list[str],
    template_count: int,
    provider: str,
    model: str | None,
    scenario: str,
    prompt_ids: list[str],
    replicates: int,
    concurrency: int,
    reset: bool,
    blind: bool,
    skip_session_build: bool,
    skip_evaluation: bool,
    skip_analysis: bool,
) -> None:
    if reset:
        logging.info('Reset requested: clearing data and outputs')
        reset_pipeline_state()

    for index, job in enumerate(jobs):
        logging.info('Generating templates for job=%s (%d/%d)', job, index + 1, len(jobs))
        await run_generate_templates(
            job=job,
            languages=languages,
            count=template_count,
            provider=provider,
            model=model,
            reset=False,
        )

    logging.info('Generating variants for all currently available templates')
    await run_generate_variants(
        provider=provider,
        model=model,
        scenario=scenario,
        reset=False,
    )

    if not skip_session_build:
        logging.info('Building sessions')
        sessions = build_sessions(prompt_ids=prompt_ids, replicates=replicates)
        write_jsonl(SESSIONS_PATH, sessions)
        logging.info('Built %d sessions', len(sessions))
    else:
        logging.info('Skipping session building')

    if not skip_evaluation:
        logging.info('Running evaluation')
        await run_evaluation(
            provider=provider,
            model=model,
            concurrency=concurrency,
            include_personal_signals=not blind,
            skip_existing=not reset,
        )
    else:
        logging.info('Skipping evaluation')

    if not skip_analysis:
        logging.info('Running analysis')
        analysis_main()
    else:
        logging.info('Skipping analysis')

    logging.info('Pipeline complete')


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the recruitment-bias experiment pipeline')
    parser.add_argument('--job', default=None)
    parser.add_argument('--jobs', nargs='+', default=None, help='Multiple jobs, space-separated or comma-separated')
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
    parser.add_argument('--skip-session-build', action='store_true')
    parser.add_argument('--skip-evaluation', action='store_true')
    parser.add_argument('--skip-analysis', action='store_true')
    args = parser.parse_args()

    job_list = parse_jobs(args.job, args.jobs)
    if not job_list:
        raise ValueError('Provide --job or --jobs')

    languages = normalize_cli_list(args.languages, default=['English'])
    prompt_ids = normalize_cli_list(args.prompts, default=['neutral_v1', 'strict_merit_v1', 'role_specific_v1'])

    asyncio.run(
        run_pipeline(
            jobs=job_list,
            languages=languages,
            template_count=args.template_count,
            provider=args.provider,
            model=args.model,
            scenario=args.scenario,
            prompt_ids=prompt_ids,
            replicates=args.replicates,
            concurrency=args.concurrency,
            reset=args.reset,
            blind=args.blind,
            skip_session_build=args.skip_session_build,
            skip_evaluation=args.skip_evaluation,
            skip_analysis=args.skip_analysis,
        )
    )


if __name__ == '__main__':
    main()