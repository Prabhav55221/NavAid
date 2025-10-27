# gemini_client.py
from __future__ import annotations

import json, mimetypes, os, re, time
from pathlib import Path
from typing import Any, Dict, Tuple

import google.generativeai as genai  # pip install google-generativeai

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
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash",
                 temperature: float = 0.2, top_p: float = 0.8, max_retries: int = 3):
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.gcfg = genai.types.GenerationConfig(
            temperature=temperature, top_p=top_p, candidate_count=1, response_mime_type="application/json"
        )
        self.max_retries = max_retries

    def analyze(self, image_path: Path, prompt_text: str) -> Tuple[Dict[str, Any], str]:
        """
        Returns (parsed_json_dict, raw_text).
        """
        img_part = _load_image_for_gemini(image_path)
        contents = [prompt_text, img_part]

        # retry on transient errors or JSON parse errors
        backoff = 1.0
        last_txt = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.model.generate_content(contents, generation_config=self.gcfg)
                last_txt = resp.text.strip() if hasattr(resp, "text") else str(resp)
                obj = _extract_json_object(last_txt)
                data = json.loads(obj)
                return data, last_txt
            except Exception as e:
                if attempt == self.max_retries:
                    raise
                time.sleep(backoff)
                backoff *= 2.0
        return {}, last_txt  # unreachable
