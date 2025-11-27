import json
import os
from pathlib import Path

import pytest
import requests
from dotenv import load_dotenv


def _load_config():
    cfg_path = Path(__file__).resolve().parent.parent / "config.json"
    with cfg_path.open() as f:
        return json.load(f)


@pytest.mark.integration
def test_gemini_embedding_endpoint():
    """
    Verifies that the configured Gemini embedding model is reachable with the provided API key.
    Skips if GEMINI_API_KEY is not set or provider is not gemini.
    """
    load_dotenv()
    cfg = _load_config()
    memory_cfg = cfg.get("memory", {})

    if memory_cfg.get("embedding_provider") != "gemini":
        pytest.skip("Embedding provider is not gemini in config.json")

    model = memory_cfg.get("embedding_model")
    if not model:
        pytest.skip("No embedding_model configured")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set")

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
    payload = {
        "model": model,
        "content": {"parts": [{"text": "ping"}]},
    }
    resp = requests.post(url, params={"key": api_key}, json=payload, timeout=15)

    assert resp.status_code == 200, f"Gemini embedding request failed: {resp.status_code} {resp.text}"
    data = resp.json()
    embedding = data.get("embedding", {}).get("values") or data.get("embedding", {}).get("value")
    assert embedding, f"Gemini embedding response missing values: {data}"
    assert isinstance(embedding, list), "Embedding should be a list of floats"
    assert len(embedding) > 0, "Embedding vector is empty"
