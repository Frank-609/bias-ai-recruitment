from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any

import aiohttp

from src.common import DATA_DIR, OUTPUTS_DIR, PROMPTS_DIR, append_jsonl, bootstrap, read_jsonl
from src.llm_client import call_llm

bootstrap()

VARIANTS_PATH = DATA_DIR / 'variants.jsonl'
SESSIONS_PATH = DATA_DIR / 'sessions.jsonl'
RAW_DIR = OUTPUTS_DIR / 'raw'
PARSED_DIR = OUTPUTS_DIR / 'parsed'
RESULTS_JSONL = OUTPUTS_DIR / 'results.jsonl'

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def load_variant_index() -> dict[str, dict]:
    return {row['cv_id']: row for row in read_jsonl(VARIANTS_PATH)}


def load_sessions() -> list[dict]:
    rows = read_jsonl(SESSIONS_PATH)
    if not rows:
        raise ValueError('No sessions found. Build sessions first.')
    return rows


def load_eval_prompt(prompt_id: str, job: str) -> str:
    path = PROMPTS_DIR / f'evaluate_{prompt_id}.txt'
    if not path.exists():
        raise FileNotFoundError(f'Missing prompt file: {path}')
    return path.read_text(encoding='utf-8').format(job=job)


def format_cv_for_prompt(cv: dict, idx: int, include_personal_signals: bool) -> str:
    lines = [f'Candidate {idx}']

    if include_personal_signals:
        lines += [
            f"Name: {cv.get('full_name', 'N/A')}",
            f"Age: {cv.get('age', 'N/A')}",
        ]

    lines.append(f"Target role: {cv.get('job_title_target', 'N/A')}")

    if cv.get('professional_summary'):
        lines.append(f"Professional summary: {cv['professional_summary']}")

    for edu in cv.get('education', []):
        lines.append(
            f"Education: {edu.get('degree', 'Degree')} at {edu.get('school', 'Institution')} ({edu.get('year', 'N/A')})"
        )

    for exp in cv.get('experience', []):
        lines.append(
            f"Experience: {exp.get('title', 'Role')} at {exp.get('company', 'Company')} ({exp.get('duration_months', 'N/A')} months)"
        )
        bullets = exp.get('achievements') or exp.get('missions') or exp.get('responsibilities') or []
        if isinstance(bullets, str):
            bullets = [bullets]
        for bullet in bullets[:3]:
            lines.append(f'- {bullet}')

    if cv.get('skills'):
        lines.append(f"Technical skills: {', '.join(cv['skills'])}")

    if cv.get('soft_skills'):
        lines.append(f"Soft skills: {', '.join(cv['soft_skills'])}")

    if cv.get('languages'):
        lines.append(f"Languages: {', '.join(cv['languages'])}")

    if cv.get('certifications'):
        lines.append(f"Certifications: {', '.join(cv['certifications'])}")

    if int(cv.get('career_gap_months', 0) or 0) > 0:
        lines.append(f"Career gap: {cv.get('career_gap_months')} months ({cv.get('gap_reason', 'Unspecified')})")

    if cv.get('hobbies'):
        lines.append(f"Hobbies: {', '.join(cv['hobbies'])}")

    if cv.get('cover_letter'):
        lines.append('Cover letter excerpt:')
        lines.append(cv['cover_letter'])

    return '\n'.join(lines)


def extract_json_object(raw: str) -> str:
    text = raw.strip()
    if text.startswith('```'):
        parts = text.split('```')
        for part in parts:
            candidate = part.strip()
            if candidate.startswith('json'):
                candidate = candidate[4:].strip()
            if candidate.startswith('{') and candidate.endswith('}'):
                return candidate
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def parse_response(raw: str) -> tuple[bool, dict[str, Any] | None, str | None]:
    try:
        parsed = json.loads(extract_json_object(raw))
        if not isinstance(parsed, dict):
            return False, None, 'Parsed response is not an object'
        return True, parsed, None
    except Exception as exc:
        return False, None, str(exc)


async def evaluate_one_session(
    session: dict,
    cv_index: dict[str, dict],
    provider: str,
    model: str | None,
    http_session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    include_personal_signals: bool,
    skip_existing: bool,
    existing_done: set[tuple[str | None, str | None, str | None]],
) -> None:
    async with semaphore:
        done_key = (session.get('session_id'), provider, model)
        if skip_existing and done_key in existing_done:
            logging.info(
                'Skipping already evaluated session=%s provider=%s model=%s',
                session['session_id'],
                provider,
                model,
            )
            return

        cv1 = cv_index.get(session['cv_ids'][0])
        cv2 = cv_index.get(session['cv_ids'][1])

        if not cv1 or not cv2:
            logging.warning(
                'Skipping session=%s because one or more cv_ids are missing from variants.jsonl',
                session['session_id'],
            )
            return

        prompt = load_eval_prompt(session['prompt_id'], session['job_title_target'])
        prompt += '\n\n' + format_cv_for_prompt(cv1, 1, include_personal_signals)
        prompt += '\n\n' + format_cv_for_prompt(cv2, 2, include_personal_signals)

        raw_response = await call_llm(
            prompt=prompt,
            provider=provider,
            session=http_session,
            model=model,
            temperature=0.2,
            max_tokens=1800,
            response_format_json=True,
            random_seed=abs(hash(session['session_id'])) % (2**31),
        )

        parse_ok, parsed_response, parse_error = parse_response(raw_response)

        raw_record = {
            'session_id': session['session_id'],
            'provider': provider,
            'model': model,
            'prompt_id': session['prompt_id'],
            'raw_response': raw_response,
            'created_at': utc_now_iso(),
        }

        parsed_record = {
            'session_id': session['session_id'],
            'provider': provider,
            'model': model,
            'prompt_id': session['prompt_id'],
            'matched_group_id': session['matched_group_id'],
            'bias_variable': session['bias_variable'],
            'variant_values': session['variant_values'],
            'cv_ids': session['cv_ids'],
            'scenario': session.get('scenario', 'single_attribute'),
            'parse_ok': parse_ok,
            'parse_error': parse_error,
            'parsed_response': parsed_response,
            'created_at': utc_now_iso(),
            'include_personal_signals': include_personal_signals,
        }

        RAW_DIR.mkdir(parents=True, exist_ok=True)
        PARSED_DIR.mkdir(parents=True, exist_ok=True)

        (RAW_DIR / f"{session['session_id']}_{provider}_{model or 'default'}.json").write_text(
            json.dumps(raw_record, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        (PARSED_DIR / f"{session['session_id']}_{provider}_{model or 'default'}.json").write_text(
            json.dumps(parsed_record, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

        append_jsonl(RESULTS_JSONL, [{
            'session_id': session['session_id'],
            'matched_group_id': session['matched_group_id'],
            'provider': provider,
            'model': model,
            'prompt_id': session['prompt_id'],
            'bias_variable': session['bias_variable'],
            'variant_values': session['variant_values'],
            'cv_ids': session['cv_ids'],
            'scenario': session.get('scenario', 'single_attribute'),
            'parse_ok': parse_ok,
            'parse_error': parse_error,
            'parsed_response': parsed_response,
            'include_personal_signals': include_personal_signals,
            'created_at': utc_now_iso(),
        }])

        existing_done.add(done_key)
        logging.info('Evaluated %s with %s', session['session_id'], provider)


async def run_evaluation(
    provider: str,
    model: str | None,
    concurrency: int,
    include_personal_signals: bool,
    skip_existing: bool,
) -> None:
    existing_done: set[tuple[str | None, str | None, str | None]] = {
        (row.get('session_id'), row.get('provider'), row.get('model'))
        for row in read_jsonl(RESULTS_JSONL)
    }

    sessions = load_sessions()
    cv_index = load_variant_index()
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency)

    async with aiohttp.ClientSession(connector=connector) as http_session:
        await asyncio.gather(*[
            evaluate_one_session(
                session=s,
                cv_index=cv_index,
                provider=provider,
                model=model,
                http_session=http_session,
                semaphore=semaphore,
                include_personal_signals=include_personal_signals,
                skip_existing=skip_existing,
                existing_done=existing_done,
            )
            for s in sessions
        ])


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate generated sessions with an LLM provider')
    parser.add_argument('--provider', type=str, choices=['openai', 'mistral'], required=True)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--concurrency', type=int, default=5)
    parser.add_argument('--blind', action='store_true', help='Hide name and age from the prompt for ablation experiments')
    parser.add_argument('--no-skip-existing', action='store_true', help='Re-run sessions even if results already exist')
    args = parser.parse_args()

    asyncio.run(
        run_evaluation(
            provider=args.provider,
            model=args.model,
            concurrency=args.concurrency,
            include_personal_signals=not args.blind,
            skip_existing=not args.no_skip_existing,
        )
    )


if __name__ == '__main__':
    main()