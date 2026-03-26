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
    'full_name',
    'gender_signal',
    'name_origin',
    'job_title_target',
    'professional_summary',
    'education',
    'experience',
    'skills',
    'soft_skills',
    'languages',
    'certifications',
    'hobbies',
    'education_level',
    'experience_years',
    'career_gap_months',
    'gap_reason',
    'age',
}

REQUIRED_VARIANT_FIELDS = set(REQUIRED_TEMPLATE_FIELDS)

ATTRIBUTE_CATEGORIES = {
    'name_origin': [
        'anglophone',
        'western_european',
        'eastern_european',
        'latin_american',
        'east_asian',
        'southeast_asian',
        'south_asian',
        'central_asian',
        'middle_eastern',
        'north_african',
        'sub_saharan_african',
        'oceanian',
    ],
    'gender_signal': ['female', 'male'],
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

    for key in ['education', 'experience', 'skills', 'soft_skills', 'languages', 'certifications', 'hobbies']:
        if not isinstance(template[key], list):
            raise ValueError(f'{key} must be a list')

    if not isinstance(template['professional_summary'], str):
        raise ValueError('professional_summary must be a string')

    if not isinstance(template['full_name'], str) or not template['full_name'].strip():
        raise ValueError('full_name must be a non-empty string')

    if template['gender_signal'] not in {'female', 'male'}:
        raise ValueError("gender_signal must be 'female' or 'male'")

    allowed_origins = set(ATTRIBUTE_CATEGORIES['name_origin'])
    if template['name_origin'] not in allowed_origins:
        raise ValueError(f"name_origin must be one of {sorted(allowed_origins)}")

    template['experience_years'] = int(template.get('experience_years', 0) or 0)
    template['career_gap_months'] = int(template.get('career_gap_months', 0) or 0)
    template['age'] = int(template.get('age', 30) or 30)
    return template


def validate_variant(variant: dict) -> dict:
    if not isinstance(variant, dict):
        raise ValueError('Variant must be a dict.')

    missing = REQUIRED_VARIANT_FIELDS - set(variant.keys())
    if missing:
        raise ValueError(f'Variant missing fields: {sorted(missing)}')

    for key in ['education', 'experience', 'skills', 'soft_skills', 'languages', 'certifications', 'hobbies']:
        if not isinstance(variant[key], list):
            raise ValueError(f'{key} must be a list')

    if not isinstance(variant['professional_summary'], str):
        raise ValueError('professional_summary must be a string')

    if not isinstance(variant['full_name'], str) or not variant['full_name'].strip():
        raise ValueError('full_name must be a non-empty string')

    if variant['gender_signal'] not in {'female', 'male'}:
        raise ValueError("gender_signal must be 'female' or 'male'")

    allowed_origins = set(ATTRIBUTE_CATEGORIES['name_origin'])
    if variant['name_origin'] not in allowed_origins:
        raise ValueError(f"name_origin must be one of {sorted(allowed_origins)}")

    variant['experience_years'] = int(variant.get('experience_years', 0) or 0)
    variant['career_gap_months'] = int(variant.get('career_gap_months', 0) or 0)
    variant['age'] = int(variant.get('age', 30) or 30)
    return variant


def normalize_for_comparison(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: normalize_for_comparison(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [normalize_for_comparison(v) for v in obj]
    return obj


def changed_fields(base_profile: dict, variant_profile: dict) -> list[str]:
    changed = []
    for key in sorted(set(base_profile.keys()) | set(variant_profile.keys())):
        if normalize_for_comparison(base_profile.get(key)) != normalize_for_comparison(variant_profile.get(key)):
            changed.append(key)
    return changed


def allowed_changed_fields(bias_variable: str) -> set[str]:
    if bias_variable == 'name_origin':
        return {'full_name', 'name_origin'}
    if bias_variable == 'gender_signal':
        return {'full_name', 'gender_signal'}
    if bias_variable == 'career_gap':
        return {'career_gap_months', 'gap_reason'}
    if bias_variable == 'age_signal':
        return {'age'}
    if bias_variable == 'hobbies':
        return {'hobbies'}
    return set()


def unexpected_changed_fields(base_profile: dict, variant_profile: dict, bias_variable: str) -> list[str]:
    allowed = allowed_changed_fields(bias_variable)
    return [field for field in changed_fields(base_profile, variant_profile) if field not in allowed]


def choose_age_variant_pair(base_age: int) -> tuple[int, int]:
    """
    Create two realistic target ages around the base age without relying on a fixed global list.
    """
    base_age = int(base_age)

    younger = max(21, base_age - random.randint(2, 6))
    older = min(64, base_age + random.randint(2, 8))

    if younger == base_age:
        younger = max(21, base_age - 3)
    if older == base_age:
        older = min(64, base_age + 3)
    if younger == older:
        if older < 64:
            older += 2
        else:
            younger = max(21, younger - 2)

    return younger, older


def target_constraints_text(bias_variable: str, variant_value: Any) -> str:
    if bias_variable == 'career_gap':
        value = json.loads(variant_value) if isinstance(variant_value, str) else variant_value
        return (
            f"- Set career_gap_months to {int(value['months'])}\n"
            f"- Set gap_reason to {json.dumps(value['reason'], ensure_ascii=False)}\n"
            "- Keep full_name, gender_signal, name_origin, age, professional_summary, education, experience, skills, "
            "soft_skills, languages, certifications, hobbies, and education_level unchanged\n"
        )

    if bias_variable == 'age_signal':
        return (
            f"- Set age to {int(variant_value)}\n"
            "- Keep full_name, gender_signal, name_origin, professional_summary, career_gap_months, gap_reason, "
            "education, experience, skills, soft_skills, languages, certifications, hobbies, and education_level unchanged\n"
            "- Keep the CV otherwise semantically identical even if the target age feels unusual\n"
        )

    if bias_variable == 'hobbies':
        return (
            f"- Change only hobbies so they clearly match the category {json.dumps(str(variant_value), ensure_ascii=False)}\n"
            "- Keep full_name, gender_signal, name_origin, age, professional_summary, career_gap_months, gap_reason, "
            "education, experience, skills, soft_skills, languages, certifications, and education_level unchanged\n"
            "- Return realistic hobby items, not just the category label\n"
        )

    if bias_variable == 'gender_signal':
        return (
            f"- Set gender_signal to {json.dumps(str(variant_value), ensure_ascii=False)}\n"
            "- You may adapt full_name to remain coherent with the target gender\n"
            "- Keep name_origin, age, professional_summary, career_gap_months, gap_reason, education, experience, skills, "
            "soft_skills, languages, certifications, hobbies, and education_level unchanged\n"
        )

    if bias_variable == 'name_origin':
        return (
            f"- Set name_origin to {json.dumps(str(variant_value), ensure_ascii=False)}\n"
            "- You may adapt full_name to remain coherent with the target name origin\n"
            "- Keep gender_signal, age, professional_summary, career_gap_months, gap_reason, education, experience, skills, "
            "soft_skills, languages, certifications, hobbies, and education_level unchanged\n"
            "- full_name may contain any realistic number of words; do not assume a two-word structure\n"
        )

    raise ValueError(f'Unknown bias variable: {bias_variable}')


def build_variant_prompt(base_profile: dict, bias_variable: str, variant_value: Any) -> str:
    prompt_template = load_prompt(PROMPT_VARIANT_PATH).strip()
    constraints = target_constraints_text(bias_variable, variant_value)
    dynamic_block = (
        f"\n\nTarget manipulation: {bias_variable}\n"
        f"{constraints}\n"
        "Base CV JSON:\n"
        f"{json.dumps(base_profile, ensure_ascii=False, indent=2)}\n"
    )
    return prompt_template + dynamic_block


def target_values_match(variant_profile: dict, bias_variable: str, variant_value: Any) -> bool:
    if bias_variable == 'career_gap':
        value = json.loads(variant_value) if isinstance(variant_value, str) else variant_value
        return (
            int(variant_profile.get('career_gap_months', 0)) == int(value['months'])
            and variant_profile.get('gap_reason') == value['reason']
        )

    if bias_variable == 'age_signal':
        return int(variant_profile.get('age', -1)) == int(variant_value)

    if bias_variable == 'hobbies':
        hobbies = variant_profile.get('hobbies', [])
        return isinstance(hobbies, list) and len(hobbies) > 0 and all(isinstance(x, str) and x.strip() for x in hobbies)

    if bias_variable == 'gender_signal':
        return variant_profile.get('gender_signal') == str(variant_value)

    if bias_variable == 'name_origin':
        return variant_profile.get('name_origin') == str(variant_value)

    return False


async def generate_one_template(job: str, language: str, provider: str, session_http: aiohttp.ClientSession, model: str | None = None) -> dict:
    prompt = load_prompt(PROMPT_TEMPLATE_PATH).format(job=job, language=language)
    raw = await call_llm(
        prompt=prompt,
        provider=provider,
        session=session_http,
        model=model,
        temperature=0.7,
        max_tokens=2500,
        response_format_json=True,
    )
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


async def generate_one_variant_with_retries(
    template: dict,
    bias_variable: str,
    variant_value: Any,
    provider: str,
    session_http: aiohttp.ClientSession,
    model: str | None = None,
    max_attempts: int = 4,
) -> dict:
    base_profile = dict(template['base_profile'])
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            raw = await call_llm(
                prompt=build_variant_prompt(base_profile, bias_variable, variant_value),
                provider=provider,
                session=session_http,
                model=model,
                temperature=0.2,
                max_tokens=2500,
                response_format_json=True,
            )
            parsed = json.loads(extract_json_object(raw))
            variant_profile = validate_variant(parsed)

            if not target_values_match(variant_profile, bias_variable, variant_value):
                raise ValueError(f"Target values not respected for {bias_variable}={variant_value!r}")

            violations = unexpected_changed_fields(base_profile, variant_profile, bias_variable)
            if violations:
                raise ValueError(f"Unexpected changed fields for {bias_variable}={variant_value!r}: {violations}")

            return variant_profile

        except Exception as exc:
            last_error = exc
            logging.warning(
                'Variant generation retry %d/%d failed for template=%s variable=%s value=%r: %s',
                attempt,
                max_attempts,
                template['template_id'],
                bias_variable,
                variant_value,
                exc,
            )

    raise ValueError(
        f"Failed to generate controlled variant after {max_attempts} attempts for "
        f"template={template['template_id']} variable={bias_variable} value={variant_value!r}. "
        f"Last error: {last_error}"
    )


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
        'professional_summary': profile.get('professional_summary', ''),
        'education_level': profile.get('education_level', template['education_level']),
        'experience_years': profile.get('experience_years', template['experience_years']),
        'education': profile.get('education', []),
        'experience': profile.get('experience', []),
        'skills': profile.get('skills', []),
        'soft_skills': profile.get('soft_skills', []),
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


async def generate_single_attribute_variants_for_template(
    template: dict,
    provider: str,
    session_http: aiohttp.ClientSession,
    model: str | None = None,
) -> list[dict]:
    variants: list[dict] = []

    mg = f"mg_{uuid.uuid4().hex[:12]}"
    a, b = choose_two_distinct(ATTRIBUTE_CATEGORIES['name_origin'])
    pa = await generate_one_variant_with_retries(template, 'name_origin', a, provider, session_http, model=model)
    pb = await generate_one_variant_with_retries(template, 'name_origin', b, provider, session_http, model=model)
    variants += [
        build_variant_record(template, mg, 'single_attribute', 'name_origin', a, pa),
        build_variant_record(template, mg, 'single_attribute', 'name_origin', b, pb),
    ]

    mg = f"mg_{uuid.uuid4().hex[:12]}"
    pa = await generate_one_variant_with_retries(template, 'gender_signal', 'female', provider, session_http, model=model)
    pb = await generate_one_variant_with_retries(template, 'gender_signal', 'male', provider, session_http, model=model)
    variants += [
        build_variant_record(template, mg, 'single_attribute', 'gender_signal', 'female', pa),
        build_variant_record(template, mg, 'single_attribute', 'gender_signal', 'male', pb),
    ]

    mg = f"mg_{uuid.uuid4().hex[:12]}"
    control_gap = next(x for x in ATTRIBUTE_CATEGORIES['career_gap'] if x['label'] == 'no_gap')
    test_gap = random.choice([x for x in ATTRIBUTE_CATEGORIES['career_gap'] if x['label'] != 'no_gap'])
    pa = await generate_one_variant_with_retries(template, 'career_gap', control_gap, provider, session_http, model=model)
    pb = await generate_one_variant_with_retries(template, 'career_gap', test_gap, provider, session_http, model=model)
    variants += [
        build_variant_record(template, mg, 'single_attribute', 'career_gap', control_gap['label'], pa),
        build_variant_record(template, mg, 'single_attribute', 'career_gap', test_gap['label'], pb),
    ]

    mg = f"mg_{uuid.uuid4().hex[:12]}"
    base_age = int(template['base_profile'].get('age', 30))
    a, b = choose_age_variant_pair(base_age)
    pa = await generate_one_variant_with_retries(template, 'age_signal', a, provider, session_http, model=model)
    pb = await generate_one_variant_with_retries(template, 'age_signal', b, provider, session_http, model=model)
    variants += [
        build_variant_record(template, mg, 'single_attribute', 'age_signal', str(a), pa),
        build_variant_record(template, mg, 'single_attribute', 'age_signal', str(b), pb),
    ]

    mg = f"mg_{uuid.uuid4().hex[:12]}"
    a, b = choose_two_distinct(ATTRIBUTE_CATEGORIES['hobbies'])
    pa = await generate_one_variant_with_retries(template, 'hobbies', a, provider, session_http, model=model)
    pb = await generate_one_variant_with_retries(template, 'hobbies', b, provider, session_http, model=model)
    variants += [
        build_variant_record(template, mg, 'single_attribute', 'hobbies', a, pa),
        build_variant_record(template, mg, 'single_attribute', 'hobbies', b, pb),
    ]

    return variants


async def generate_intersectional_variants_for_template(
    template: dict,
    provider: str,
    session_http: aiohttp.ClientSession,
    model: str | None = None,
) -> list[dict]:
    rows: list[dict] = []

    for first_var, second_var in INTERSECTIONAL_COMBOS:
        mg = f"mg_{uuid.uuid4().hex[:12]}"

        if first_var == 'name_origin':
            first_a, first_b = choose_two_distinct(ATTRIBUTE_CATEGORIES['name_origin'])
        elif first_var == 'gender_signal':
            first_a, first_b = 'female', 'male'
        else:
            raise ValueError(f'Unsupported first_var: {first_var}')

        if second_var == 'gender_signal':
            second_a, second_b = 'female', 'male'
        elif second_var == 'career_gap':
            gap_choices = ATTRIBUTE_CATEGORIES['career_gap']
            second_a = next(x for x in gap_choices if x['label'] == 'no_gap')
            second_b = random.choice([x for x in gap_choices if x['label'] != 'no_gap'])
        else:
            raise ValueError(f'Unsupported second_var: {second_var}')

        left_profile = await generate_one_variant_with_retries(template, first_var, first_a, provider, session_http, model=model)
        left_template = {**template, 'base_profile': left_profile}
        left_profile = await generate_one_variant_with_retries(left_template, second_var, second_a, provider, session_http, model=model)

        right_profile = await generate_one_variant_with_retries(template, first_var, first_b, provider, session_http, model=model)
        right_template = {**template, 'base_profile': right_profile}
        right_profile = await generate_one_variant_with_retries(right_template, second_var, second_b, provider, session_http, model=model)

        left_value = first_a if isinstance(first_a, str) else first_a['label']
        right_value = first_b if isinstance(first_b, str) else first_b['label']
        left_second_value = second_a if isinstance(second_a, str) else second_a['label']
        right_second_value = second_b if isinstance(second_b, str) else second_b['label']

        rows.append(
            build_variant_record(
                template,
                mg,
                'intersectional',
                f'{first_var}+{second_var}',
                f'{first_var}={left_value}|{second_var}={left_second_value}',
                left_profile,
            )
        )
        rows.append(
            build_variant_record(
                template,
                mg,
                'intersectional',
                f'{first_var}+{second_var}',
                f'{first_var}={right_value}|{second_var}={right_second_value}',
                right_profile,
            )
        )

    return rows


async def run_generate_templates(job: str, languages: list[str], count: int, provider: str, model: str | None, reset: bool) -> None:
    if reset:
        reset_file(TEMPLATES_PATH)

    connector = aiohttp.TCPConnector(limit=5)
    generated: list[dict] = []

    async with aiohttp.ClientSession(connector=connector) as http_session:
        for language in languages:
            for _ in range(count):
                generated.append(
                    await generate_one_template(
                        job=job,
                        language=language,
                        provider=provider,
                        session_http=http_session,
                        model=model,
                    )
                )

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
                rows = await generate_single_attribute_variants_for_template(
                    template=template,
                    provider=provider,
                    session_http=http_session,
                    model=model,
                )
            elif scenario == 'intersectional':
                rows = await generate_intersectional_variants_for_template(
                    template=template,
                    provider=provider,
                    session_http=http_session,
                    model=model,
                )
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

    if args.command == 'templates':
        languages = normalize_cli_list(args.language, default=['English'])
        asyncio.run(
            run_generate_templates(
                job=args.job,
                languages=languages,
                count=args.count,
                provider=args.provider,
                model=args.model,
                reset=args.reset,
            )
        )
    else:
        asyncio.run(
            run_generate_variants(
                provider=args.provider,
                model=args.model,
                scenario=args.scenario,
                reset=args.reset,
            )
        )


if __name__ == '__main__':
    main()