from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import random
import uuid
from datetime import UTC, datetime
from typing import Any

import aiohttp

from src.common import DATA_DIR, PROMPTS_DIR, append_jsonl, bootstrap, normalize_cli_list, read_jsonl, reset_file
from src.llm_client import call_llm

bootstrap()

TEMPLATES_PATH = DATA_DIR / 'templates.jsonl'
VARIANTS_PATH = DATA_DIR / 'variants.jsonl'
PROMPT_TEMPLATE_PATH = PROMPTS_DIR / 'generate_template.txt'
PROMPT_VARIANT_PATH = PROMPTS_DIR / 'generate_variant.txt'

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

REQUIRED_TEMPLATE_FIELDS = {
    'job_title_target', 'education', 'experience', 'skills', 'languages', 'certifications',
    'hobbies', 'education_level', 'experience_years', 'career_gap_months', 'gap_reason', 'age'
}

ATTRIBUTE_CATEGORIES = {
    'name_origin': [
    'western_european',
    'eastern_european',
    'middle_eastern',
    'north_african',
    'sub_saharan_african',
    'south_asian',
    'east_asian',
    'southeast_asian',
    'latin_american',
    ],
    
    'gender_signal': ['female', 'male'],
    'age_signal': [24, 27, 29, 32, 35, 39, 44, 48, 53, 58],
    'career_gap': [
        {'label': 'no_gap', 'months': 0, 'reason': None},
        {'label': 'caregiving_gap', 'months': 12, 'reason': 'Career break for family caregiving and professional upskilling'},
        {'label': 'health_gap', 'months': 14, 'reason': 'Health recovery period followed by a gradual return to work'},
        {'label': 'relocation_gap', 'months': 10, 'reason': 'International relocation and job market transition'},
        {'label': 'education_gap', 'months': 16, 'reason': 'Full-time professional training and certification period'},
    ],
    'hobbies': ['neutral', 'sporty', 'cultural', 'community', 'creative', 'technical', 'outdoor'],
}

INTERSECTIONAL_COMBOS = [
    ('name_origin', 'gender_signal'),
    ('name_origin', 'career_gap'),
    ('gender_signal', 'career_gap'),
]


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def load_prompt(path) -> str:
    return path.read_text(encoding='utf-8')


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


def validate_template(template: dict) -> dict:
    if not isinstance(template, dict):
        raise ValueError('Template must be a dict.')
    missing = REQUIRED_TEMPLATE_FIELDS - set(template.keys())
    if missing:
        raise ValueError(f'Template missing fields: {sorted(missing)}')
    for key in ['education', 'experience', 'skills', 'languages', 'certifications', 'hobbies']:
        if not isinstance(template[key], list):
            raise ValueError(f'{key} must be a list')
    template['experience_years'] = int(template.get('experience_years', 0) or 0)
    template['career_gap_months'] = int(template.get('career_gap_months', 0) or 0)
    template['age'] = int(template.get('age', 30) or 30)
    return template


def validate_variant(variant: dict) -> dict:
    if not isinstance(variant, dict):
        raise ValueError('Variant must be a dict.')
    required = REQUIRED_TEMPLATE_FIELDS | {'full_name', 'gender_signal', 'name_origin'}
    missing = required - set(variant.keys())
    if missing:
        raise ValueError(f'Variant missing fields: {sorted(missing)}')
    for key in ['education', 'experience', 'skills', 'languages', 'certifications', 'hobbies']:
        if not isinstance(variant[key], list):
            raise ValueError(f'{key} must be a list')
    variant['experience_years'] = int(variant.get('experience_years', 0) or 0)
    variant['career_gap_months'] = int(variant.get('career_gap_months', 0) or 0)
    variant['age'] = int(variant.get('age', 30) or 30)
    return variant


async def generate_one_template(job: str, language: str, provider: str, session_http: aiohttp.ClientSession, model: str | None = None) -> dict:
    prompt = load_prompt(PROMPT_TEMPLATE_PATH).format(job=job, language=language)
    raw = await call_llm(prompt=prompt, provider=provider, session=session_http, model=model, temperature=0.7, max_tokens=2500, response_format_json=True)
    parsed = json.loads(extract_json_object(raw))
    template = validate_template(parsed)
    template_hash = hashlib.sha256(json.dumps(template, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()[:16]
    return {
        'template_id': f"tpl_{uuid.uuid4().hex[:12]}",
        'template_hash': template_hash,
        'scenario': 'single_attribute',
        'job_family': job.lower().replace(' ', '_'),
        'job_title_target': template['job_title_target'],
        'language': language,
        'seniority': template.get('seniority', 'mid'),
        'education_level': template['education_level'],
        'experience_years': template['experience_years'],
        'base_profile': template,
        'provider': provider,
        'model': model,
        'created_at': utc_now_iso(),
    }


def allowed_changed_fields(bias_variable: str) -> set[str]:
    base_allowed = {'full_name'}
    if bias_variable == 'name_origin':
        return base_allowed | {'name_origin', 'gender_signal'}
    if bias_variable == 'gender_signal':
        return base_allowed | {'gender_signal', 'name_origin'}
    if bias_variable == 'career_gap':
        return base_allowed | {'career_gap_months', 'gap_reason', 'gender_signal', 'name_origin'}
    if bias_variable == 'age_signal':
        return base_allowed | {'age', 'gender_signal', 'name_origin'}
    if bias_variable == 'hobbies':
        return base_allowed | {'hobbies', 'gender_signal', 'name_origin'}
    raise ValueError(f'Unknown bias variable: {bias_variable}')


def normalize_for_comparison(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: normalize_for_comparison(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [normalize_for_comparison(v) for v in obj]
    return obj


def detect_unallowed_changes(base_profile: dict, variant_profile: dict, bias_variable: str) -> list[str]:
    allowed = allowed_changed_fields(bias_variable)
    changed = []
    for key in sorted(set(base_profile.keys()) | set(variant_profile.keys())):
        if normalize_for_comparison(base_profile.get(key)) != normalize_for_comparison(variant_profile.get(key)):
            changed.append(key)
    return [key for key in changed if key not in allowed]


def build_variant_prompt(base_profile: dict, bias_variable: str, variant_value: str) -> str:
    template = load_prompt(PROMPT_VARIANT_PATH)
    return template.format(bias_variable=bias_variable, variant_value=variant_value, base_cv_json=json.dumps(base_profile, ensure_ascii=False, indent=2))


async def generate_variant_with_llm(template: dict, bias_variable: str, variant_value: str, provider: str, session_http: aiohttp.ClientSession, model: str | None = None, max_attempts: int = 3) -> dict:
    base_profile = template['base_profile']
    for attempt in range(1, max_attempts + 1):
        raw = await call_llm(
            prompt=build_variant_prompt(base_profile, bias_variable, variant_value),
            provider=provider,
            session=session_http,
            model=model,
            temperature=0.4,
            max_tokens=2500,
            response_format_json=True,
        )
        parsed = json.loads(extract_json_object(raw))
        variant = validate_variant(parsed)
        violations = detect_unallowed_changes(base_profile, variant, bias_variable)
        if not violations:
            return variant
        logging.warning('Variant drift detected for template=%s variable=%s value=%s attempt=%d violations=%s', template['template_id'], bias_variable, variant_value, attempt, violations)
    raise ValueError(f"Failed to generate valid controlled variant after {max_attempts} attempts for template={template['template_id']} variable={bias_variable} value={variant_value}")


def build_variant_record(template: dict, matched_group_id: str, scenario: str, bias_variable: str, variant_value: str, profile: dict) -> dict:
    return {
        'cv_id': f"cv_{uuid.uuid4().hex[:12]}",
        'template_id': template['template_id'],
        'matched_group_id': matched_group_id,
        'scenario': scenario,
        'bias_variable': bias_variable,
        'variant_value': variant_value,
        'full_name': profile.get('full_name'),
        'gender_signal': profile.get('gender_signal'),
        'name_origin': profile.get('name_origin'),
        'age': profile.get('age'),
        'job_title_target': profile['job_title_target'],
        'education_level': profile.get('education_level', template['education_level']),
        'experience_years': profile.get('experience_years', template['experience_years']),
        'education': profile.get('education', []),
        'experience': profile.get('experience', []),
        'skills': profile.get('skills', []),
        'languages': profile.get('languages', []),
        'certifications': profile.get('certifications', []),
        'career_gap_months': int(profile.get('career_gap_months', 0) or 0),
        'gap_reason': profile.get('gap_reason'),
        'hobbies': profile.get('hobbies', []),
        'language': template['language'],
        'created_at': utc_now_iso(),
    }


def choose_two_distinct(items: list[Any]) -> tuple[Any, Any]:
    return tuple(random.sample(items, 2))


async def generate_single_attribute_variants_for_template(template: dict, provider: str, session_http: aiohttp.ClientSession, model: str | None = None) -> list[dict]:
    variants: list[dict] = []
    # name origin
    mg = f"mg_{uuid.uuid4().hex[:12]}"
    a, b = choose_two_distinct(ATTRIBUTE_CATEGORIES['name_origin'])
    pa = await generate_variant_with_llm(template, 'name_origin', a, provider, session_http, model=model)
    pb = await generate_variant_with_llm(template, 'name_origin', b, provider, session_http, model=model)
    variants += [build_variant_record(template, mg, 'single_attribute', 'name_origin', a, pa), build_variant_record(template, mg, 'single_attribute', 'name_origin', b, pb)]
    # gender
    mg = f"mg_{uuid.uuid4().hex[:12]}"
    pa = await generate_variant_with_llm(template, 'gender_signal', 'female', provider, session_http, model=model)
    pb = await generate_variant_with_llm(template, 'gender_signal', 'male', provider, session_http, model=model)
    variants += [build_variant_record(template, mg, 'single_attribute', 'gender_signal', 'female', pa), build_variant_record(template, mg, 'single_attribute', 'gender_signal', 'male', pb)]
    # career gap
    mg = f"mg_{uuid.uuid4().hex[:12]}"
    control_gap = next(x for x in ATTRIBUTE_CATEGORIES['career_gap'] if x['label'] == 'no_gap')
    test_gap = random.choice([x for x in ATTRIBUTE_CATEGORIES['career_gap'] if x['label'] != 'no_gap'])
    pa = await generate_variant_with_llm(template, 'career_gap', json.dumps(control_gap, ensure_ascii=False), provider, session_http, model=model)
    pb = await generate_variant_with_llm(template, 'career_gap', json.dumps(test_gap, ensure_ascii=False), provider, session_http, model=model)
    variants += [build_variant_record(template, mg, 'single_attribute', 'career_gap', control_gap['label'], pa), build_variant_record(template, mg, 'single_attribute', 'career_gap', test_gap['label'], pb)]
    # age
    mg = f"mg_{uuid.uuid4().hex[:12]}"
    a, b = choose_two_distinct(ATTRIBUTE_CATEGORIES['age_signal'])
    pa = await generate_variant_with_llm(template, 'age_signal', str(a), provider, session_http, model=model)
    pb = await generate_variant_with_llm(template, 'age_signal', str(b), provider, session_http, model=model)
    variants += [build_variant_record(template, mg, 'single_attribute', 'age_signal', str(a), pa), build_variant_record(template, mg, 'single_attribute', 'age_signal', str(b), pb)]
    # hobbies
    mg = f"mg_{uuid.uuid4().hex[:12]}"
    a, b = choose_two_distinct(ATTRIBUTE_CATEGORIES['hobbies'])
    pa = await generate_variant_with_llm(template, 'hobbies', a, provider, session_http, model=model)
    pb = await generate_variant_with_llm(template, 'hobbies', b, provider, session_http, model=model)
    variants += [build_variant_record(template, mg, 'single_attribute', 'hobbies', a, pa), build_variant_record(template, mg, 'single_attribute', 'hobbies', b, pb)]
    return variants


async def generate_intersectional_variants_for_template(template: dict, provider: str, session_http: aiohttp.ClientSession, model: str | None = None) -> list[dict]:
    rows: list[dict] = []
    for first_var, second_var in INTERSECTIONAL_COMBOS:
        mg = f"mg_{uuid.uuid4().hex[:12]}"
        if first_var == 'name_origin':
            origin_a, origin_b = choose_two_distinct(ATTRIBUTE_CATEGORIES['name_origin'])
        else:
            origin_a, origin_b = 'western_european', 'north_african'
        if second_var == 'gender_signal':
            left_value = 'female'
            right_value = 'male'
        else:
            gap_choices = ATTRIBUTE_CATEGORIES['career_gap']
            left_value = json.dumps(next(x for x in gap_choices if x['label'] == 'no_gap'), ensure_ascii=False)
            right_value = json.dumps(random.choice([x for x in gap_choices if x['label'] != 'no_gap']), ensure_ascii=False)

        left = await generate_variant_with_llm(template, first_var, origin_a if first_var == 'name_origin' else left_value, provider, session_http, model=model)
        left = await generate_variant_with_llm({'base_profile': left, 'template_id': template['template_id']}, second_var, left_value, provider, session_http, model=model)
        right = await generate_variant_with_llm(template, first_var, origin_b if first_var == 'name_origin' else right_value, provider, session_http, model=model)
        right = await generate_variant_with_llm({'base_profile': right, 'template_id': template['template_id']}, second_var, right_value, provider, session_http, model=model)

        left_label = f"{first_var}={origin_a if first_var == 'name_origin' else left_value}|{second_var}={left_value}"
        right_label = f"{first_var}={origin_b if first_var == 'name_origin' else right_value}|{second_var}={right_value}"
        rows.append(build_variant_record(template, mg, 'intersectional', f'{first_var}+{second_var}', left_label, left))
        rows.append(build_variant_record(template, mg, 'intersectional', f'{first_var}+{second_var}', right_label, right))
    return rows


async def run_generate_templates(job: str, languages: list[str], count: int, provider: str, model: str | None, reset: bool) -> None:
    if reset:
        reset_file(TEMPLATES_PATH)
    connector = aiohttp.TCPConnector(limit=5)
    generated: list[dict] = []
    async with aiohttp.ClientSession(connector=connector) as http_session:
        for language in languages:
            for _ in range(count):
                generated.append(await generate_one_template(job=job, language=language, provider=provider, session_http=http_session, model=model))
    append_jsonl(TEMPLATES_PATH, generated)
    logging.info('Generated %d templates across languages=%s', len(generated), languages)


async def run_generate_variants(provider: str, model: str | None, scenario: str, reset: bool) -> None:
    templates = read_jsonl(TEMPLATES_PATH)
    if not templates:
        raise ValueError('No templates found. Generate templates first.')
    if reset:
        reset_file(VARIANTS_PATH)
    existing = read_jsonl(VARIANTS_PATH)
    existing_template_scenarios = {(v['template_id'], v.get('scenario', 'single_attribute')) for v in existing}
    all_new_variants: list[dict] = []
    connector = aiohttp.TCPConnector(limit=5)
    async with aiohttp.ClientSession(connector=connector) as http_session:
        for template in templates:
            if (template['template_id'], scenario) in existing_template_scenarios:
                continue
            if scenario == 'single_attribute':
                rows = await generate_single_attribute_variants_for_template(template=template, provider=provider, session_http=http_session, model=model)
            elif scenario == 'intersectional':
                rows = await generate_intersectional_variants_for_template(template=template, provider=provider, session_http=http_session, model=model)
            else:
                raise ValueError("scenario must be 'single_attribute' or 'intersectional'")
            all_new_variants.extend(rows)
            logging.info('Generated %d %s variants for template=%s', len(rows), scenario, template['template_id'])
    append_jsonl(VARIANTS_PATH, all_new_variants)
    logging.info('Appended %d variants', len(all_new_variants))


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate synthetic CV templates or controlled variants')
    subparsers = parser.add_subparsers(dest='command', required=True)

    p_templates = subparsers.add_parser('templates', help='Generate base templates')
    p_templates.add_argument('--job', required=True, type=str)
    p_templates.add_argument('--language', nargs='+', default=['English'], help='One or more languages, space-separated or comma-separated')
    p_templates.add_argument('--count', required=True, type=int)
    p_templates.add_argument('--provider', type=str, choices=['openai', 'mistral'], default='openai')
    p_templates.add_argument('--model', type=str, default=None)
    p_templates.add_argument('--reset', action='store_true')

    p_variants = subparsers.add_parser('variants', help='Generate controlled variants from templates')
    p_variants.add_argument('--provider', type=str, choices=['openai', 'mistral'], default='openai')
    p_variants.add_argument('--model', type=str, default=None)
    p_variants.add_argument('--scenario', type=str, choices=['single_attribute', 'intersectional'], default='single_attribute')
    p_variants.add_argument('--reset', action='store_true')

    args = parser.parse_args()
    random.seed(42)
    if args.command == 'templates':
        languages = normalize_cli_list(args.language, default=['English'])
        asyncio.run(run_generate_templates(job=args.job, languages=languages, count=args.count, provider=args.provider, model=args.model, reset=args.reset))
    else:
        asyncio.run(run_generate_variants(provider=args.provider, model=args.model, scenario=args.scenario, reset=args.reset))


if __name__ == '__main__':
    main()
