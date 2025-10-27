# gemini_client.py
from __future__ import annotations

import json, mimetypes, os, re, time, threading
from pathlib import Path
from typing import Any, Dict, Tuple

import google.generativeai as genai  # pip install google-generativeai
from google.api_core import exceptions as google_exceptions

def _extract_json_object(text: str) -> str:
    """
    Extract the first top-level {...} JSON object from a model reply.
    Strips code fences if the model disobeys.
    """
    text = text.strip()
    # remove ```json / ``` wrappers if present
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # fast path
    try:
        json.loads(text)
        return text
    except Exception:
        pass

    # bracket match
    start = text.find("{")
    if start == -1:
        raise ValueError("No '{' found in model output.")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    raise ValueError("Unbalanced JSON braces in model output.")

def _load_image_for_gemini(path: Path) -> Dict[str, Any]:
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        # default to png; adjust if you know your inputs
        mime = "image/png"
    data = path.read_bytes()
    return {"mime_type": mime, "data": data}

class GeminiHazardClient:
    """Client with built-in rate limiting for free tier (10 RPM)."""

    # Class-level lock and timestamp tracking for rate limiting across threads
    _rate_limit_lock = threading.Lock()
    _last_request_time = 0.0

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash",
                 temperature: float = 0.2, top_p: float = 0.8, max_retries: int = 3,
                 rpm_limit: int = 10):
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.gcfg = genai.types.GenerationConfig(
            temperature=temperature, top_p=top_p, candidate_count=1, response_mime_type="application/json"
        )
        self.max_retries = max_retries
        self.rpm_limit = rpm_limit
        # Calculate minimum seconds between requests to stay under RPM limit
        # Add 10% safety margin
        self.min_request_interval = (60.0 / rpm_limit) * 1.1 if rpm_limit > 0 else 0

    def _rate_limit_wait(self):
        """Thread-safe rate limiting to stay within RPM limits."""
        if self.rpm_limit <= 0:
            return  # No rate limiting

        with self._rate_limit_lock:
            now = time.time()
            time_since_last = now - self._last_request_time

            if time_since_last < self.min_request_interval:
                sleep_time = self.min_request_interval - time_since_last
                time.sleep(sleep_time)

            self._last_request_time = time.time()

    def analyze(self, image_path: Path, prompt_text: str) -> Tuple[Dict[str, Any], str]:
        """
        Returns (parsed_json_dict, raw_text).
        Includes rate limiting and 429 error handling.
        """
        img_part = _load_image_for_gemini(image_path)
        contents = [prompt_text, img_part]

        # retry on transient errors or JSON parse errors
        backoff = 1.0
        last_txt = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                # Apply rate limiting before each request
                self._rate_limit_wait()

                resp = self.model.generate_content(contents, generation_config=self.gcfg)
                last_txt = resp.text.strip() if hasattr(resp, "text") else str(resp)
                obj = _extract_json_object(last_txt)
                data = json.loads(obj)
                return data, last_txt

            except google_exceptions.ResourceExhausted as e:
                # 429 Rate Limit Error - use longer backoff
                if attempt == self.max_retries:
                    raise RuntimeError(f"Rate limit exceeded after {self.max_retries} retries: {e}")
                wait_time = backoff * 5  # Longer wait for rate limits
                print(f"  Rate limit hit, waiting {wait_time:.1f}s (attempt {attempt}/{self.max_retries})")
                time.sleep(wait_time)
                backoff *= 2.0

            except Exception as e:
                if attempt == self.max_retries:
                    raise
                time.sleep(backoff)
                backoff *= 2.0

        return {}, last_txt  # unreachable
