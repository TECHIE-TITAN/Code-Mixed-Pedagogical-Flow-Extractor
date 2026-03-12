"""
pipeline/llm_client.py — Unified LLM client
Wraps both OpenAI and Gemini behind one interface so all agents
can call get_completion() without caring which provider is active.

Uses the new `google-genai` SDK (google.genai) which replaces the
deprecated `google-generativeai` (google.generativeai).

Models used:
  Gemini fast  → gemini-2.0-flash      (free tier)
  Gemini smart → gemini-2.5-flash      (free tier, stronger reasoning)
  OpenAI fast  → gpt-4o-mini
  OpenAI smart → gpt-4o
"""

import json
import logging
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    LLM_MAX_RETRIES,
    OPENAI_API_KEY,
    GEMINI_API_KEY,
    LLM_FAST_MODEL,
    LLM_SMART_MODEL,
    GEMINI_FAST_MODEL,
    GEMINI_SMART_MODEL,
)

logger = logging.getLogger(__name__)

# ── Lazy Gemini client singleton ──────────────────────────────────────────────
_gemini_client = None

def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


def get_completion(system_prompt: str, user_prompt: str, mode: str = "fast") -> str:
    """
    Send a prompt to the active LLM provider and return the response text.

    Args:
        system_prompt: The system/instruction prompt
        user_prompt:   The user message / content to process
        mode:          "fast"  → cheaper/faster model (normalization)
                       "smart" → stronger model (extraction, reasoning)

    Returns:
        Raw string response from the LLM (should be valid JSON as instructed by system_prompt)
    """
    if LLM_PROVIDER == "gemini":
        return _gemini_completion(system_prompt, user_prompt, mode)
    else:
        return _openai_completion(system_prompt, user_prompt, mode)


# ── OpenAI ────────────────────────────────────────────────────────────────────

def _openai_completion(system_prompt: str, user_prompt: str, mode: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    model = LLM_FAST_MODEL if mode == "fast" else LLM_SMART_MODEL

    for attempt in range(LLM_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=LLM_TEMPERATURE,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            wait = 2 ** attempt
            logger.warning(f"[llm_client] OpenAI attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
            if attempt < LLM_MAX_RETRIES - 1:
                time.sleep(wait)
    raise RuntimeError("[llm_client] OpenAI: all retries exhausted.")


# ── Gemini (google-genai SDK) ─────────────────────────────────────────────────

def _gemini_completion(system_prompt: str, user_prompt: str, mode: str) -> str:
    from google import genai
    from google.genai import types

    client = _get_gemini_client()
    model_name = GEMINI_FAST_MODEL if mode == "fast" else GEMINI_SMART_MODEL

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=LLM_TEMPERATURE,
        response_mime_type="application/json",
    )

    for attempt in range(LLM_MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=config,
            )
            raw = response.text.strip()

            # Strip accidental markdown fences Gemini sometimes adds
            if raw.startswith("```"):
                lines = raw.splitlines()
                raw = "\n".join(
                    line for line in lines if not line.startswith("```")
                ).strip()

            # Validate it's parseable JSON before returning
            json.loads(raw)
            return raw

        except Exception as e:
            wait = 2 ** attempt
            logger.warning(f"[llm_client] Gemini attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
            if attempt < LLM_MAX_RETRIES - 1:
                time.sleep(wait)

    raise RuntimeError("[llm_client] Gemini: all retries exhausted.")
