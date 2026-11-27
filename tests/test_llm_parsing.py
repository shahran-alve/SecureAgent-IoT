"""Parsing of open-weight model output. These tests exercise the parser only; no model is run."""
from secureagent_iot.llm_adapter import extract_json_object


def test_json_with_prose_and_escapes():
    out = 'Sure! {"device_id": "cam_01", "action_type": "notify", "reason": "a {b} \\" c"} trailing'
    assert extract_json_object(out)["reason"] == 'a {b} " c'


def test_no_json_is_none():
    assert extract_json_object("I refuse.") is None


def test_skips_invalid_object():
    assert extract_json_object("{bad} {\"ok\": 1}") == {"ok": 1}


def test_nested_object():
    assert extract_json_object('x {"a": {"b": [1, {"c": 2}]}} y') == {"a": {"b": [1, {"c": 2}]}}
