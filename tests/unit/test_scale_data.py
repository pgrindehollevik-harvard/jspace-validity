import json

import pytest

from jspace.scale_config import config_sha256
from jspace.scale_data import load_staged_data, payload_sha256


def test_data_digest_is_canonical_and_sensitive_to_content():
    first = {
        "wikitext": ["alpha"],
        "coherence": [{"id": "gsm8k-train-1", "problem": "1+1"}],
        "non_scientific_metadata": "ignored",
    }
    reordered = {
        "coherence": [{"problem": "1+1", "id": "gsm8k-train-1"}],
        "wikitext": ["alpha"],
    }
    changed = {**reordered, "wikitext": ["beta"]}
    assert payload_sha256(first) == payload_sha256(reordered)
    assert payload_sha256(first) != payload_sha256(changed)


def test_self_consistent_but_unregistered_data_is_rejected(tmp_path):
    path = tmp_path / "data.json"
    payload = {
        "config_sha256": config_sha256(),
        "wikitext": ["not the frozen sample"],
        "coherence": [],
    }
    payload["payload_sha256"] = payload_sha256(payload)
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="not the preregistered subset"):
        load_staged_data(path)
