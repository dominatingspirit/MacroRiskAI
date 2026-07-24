"""
Thin LLM wrapper. LLMs are ONLY ever used here for summarization / explanation /
recommendation text. All numeric calculations happen deterministically or via
the sklearn/statsmodels/xgboost/lightgbm models in app/models — never in this file.
"""

import json
from app.config.settings import get_settings

settings = get_settings()


def _mock_completion(prompt: str) -> str:
    """Deterministic offline fallback so the API works with zero keys configured."""
    return json.dumps({
        "summary": "The company shows a stable financial position with moderate exposure "
                    "to rising input costs; margins are likely to compress modestly under "
                    "the forecasted inflation regime.",
        "strengths": ["Healthy liquidity position", "Consistent EBITDA generation"],
        "weaknesses": ["Elevated sensitivity to commodity price inflation", "Thin pricing power in the near term"],
    })


def complete(prompt: str, system: str | None = None, max_tokens: int = 1000) -> str:
    provider = settings.llm_provider

    if provider == "anthropic" and settings.anthropic_api_key:
        import anthropic
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        resp = client.messages.create(
            model=settings.llm_model,
            max_tokens=max_tokens,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")

    if provider == "openai" and settings.openai_api_key:
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            max_tokens=max_tokens,
            messages=messages,
        )
        return resp.choices[0].message.content

    return _mock_completion(prompt)


def complete_json(prompt: str, system: str | None = None, max_tokens: int = 1200) -> dict:
    raw = complete(prompt, system=system, max_tokens=max_tokens)
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"raw_response": raw}
