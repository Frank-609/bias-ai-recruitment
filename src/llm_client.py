"""Async HTTP client for OpenAI and Mistral chat-completions style APIs."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import aiohttp

from src.common import bootstrap

bootstrap()


class LLMError(Exception):
    pass


PROVIDERS = {
    'openai': {
        'url': 'https://api.openai.com/v1/chat/completions',
        'api_key_env': 'OPENAI_API_KEY',
        'default_model': 'gpt-4o-mini',
    },
    'mistral': {
        'url': 'https://api.mistral.ai/v1/chat/completions',
        'api_key_env': 'MISTRAL_API_KEY',
        'default_model': 'mistral-small-latest',
    },
}


def get_provider_config(provider: str) -> dict[str, str]:
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider '{provider}'. Valid: {list(PROVIDERS)}")
    return PROVIDERS[provider]


async def call_llm(
    prompt: str,
    provider: str,
    session: aiohttp.ClientSession | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 2000,
    timeout_seconds: int = 120,
    retries: int = 3,
    response_format_json: bool = False,
    random_seed: int | None = None,
) -> str:
    cfg = get_provider_config(provider)
    api_key = os.getenv(cfg['api_key_env'])
    if not api_key:
        raise EnvironmentError(f"Missing API key for {provider}. Set {cfg['api_key_env']}.")

    req_model = model or cfg['default_model']
    payload: dict[str, Any] = {
        'model': req_model,
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': temperature,
        'max_tokens': max_tokens,
    }

    if response_format_json:
        payload['response_format'] = {'type': 'json_object'}

    if provider == 'mistral' and random_seed is not None:
        payload['random_seed'] = random_seed

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }

    owns_session = session is None
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    if owns_session:
        session = aiohttp.ClientSession(timeout=timeout)
    assert session is not None

    try:
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                async with session.post(cfg['url'], headers=headers, json=payload, timeout=timeout) as resp:
                    text = await resp.text()
                    if resp.status in {429, 500, 502, 503, 504}:
                        raise LLMError(f"Transient API error ({resp.status}) from {provider}: {text[:800]}")
                    if resp.status >= 400:
                        raise LLMError(f"API error ({resp.status}) from {provider}: {text[:800]}")
                    try:
                        data = json.loads(text)
                    except json.JSONDecodeError as exc:
                        raise LLMError(f"Non-JSON API response from {provider}: {text[:800]}") from exc
                    try:
                        return data['choices'][0]['message']['content']
                    except (KeyError, IndexError, TypeError) as exc:
                        raise LLMError(f"Unexpected response schema from {provider}: {data}") from exc
            except (aiohttp.ClientError, asyncio.TimeoutError, LLMError) as exc:
                last_error = exc
                if attempt == retries:
                    break
                await asyncio.sleep(min(2 ** (attempt - 1), 8))
        raise LLMError(f"LLM call failed after {retries} attempts: {last_error}")
    finally:
        if owns_session:
            await session.close()
