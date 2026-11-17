import json

import httpx
from fastapi.testclient import TestClient

from clinical_lora.api_baseline import AnthropicExtractor
from clinical_lora.data import Example
from clinical_lora.serve import Extractor, create_app

GOOD = '{"chemicals": ["aspirin"], "diseases": ["stroke"]}'


class StubTok:
    chat_template = None


def make_client(outputs):
    calls = []

    def gen(prompt, attempt):
        calls.append(attempt)
        return outputs[min(attempt, len(outputs) - 1)]

    app = create_app(Extractor(gen, StubTok(), max_attempts=3), {"stub": True})
    return TestClient(app), calls


def test_health():
    c, _ = make_client([GOOD])
    assert c.get("/health").json() == {"status": "ok", "stub": True}


def test_extract_valid_first_try():
    c, calls = make_client([GOOD])
    r = c.post("/extract", json={"text": "Aspirin and stroke."})
    assert r.status_code == 200
    body = r.json()
    assert body["extraction"] == {"chemicals": ["aspirin"], "diseases": ["stroke"]}
    assert body["valid_json"] and not body["repaired"] and body["attempts"] == 1
    assert calls == [0]


def test_extract_retries_after_garbage():
    c, calls = make_client(["not json at all", GOOD])
    body = c.post("/extract", json={"text": "x"}).json()
    assert body["extraction"]["chemicals"] == ["aspirin"]
    assert body["attempts"] == 2 and not body["valid_json"] and calls == [0, 1]


def test_extract_falls_back_to_repair():
    c, _ = make_client(['{"chemicals": ["a"], "diseases": ["b"'])
    body = c.post("/extract", json={"text": "x"}).json()
    assert body["repaired"] and body["extraction"] == {"chemicals": ["a"], "diseases": ["b"]}
    assert body["attempts"] == 3


def test_extract_502_when_unparseable_and_422_on_bad_input():
    c, _ = make_client(["garbage"])
    assert c.post("/extract", json={"text": "x"}).status_code == 502
    assert c.post("/extract", json={"text": ""}).status_code == 422
    assert c.post("/extract", json={}).status_code == 422


def test_anthropic_baseline_with_mock_transport():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": GOOD}],
                "usage": {"input_tokens": 120, "output_tokens": 15},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    ext = AnthropicExtractor(model="m", api_key="k", client=client)
    shots = [Example("s", {"chemicals": ["x"], "diseases": []})]
    preds = ext.predict([Example("doc", {})], shots)
    assert preds[0].entities == {"chemicals": ["aspirin"], "diseases": ["stroke"]}
    assert (preds[0].prompt_tokens, preds[0].new_tokens) == (120, 15)
    assert seen["body"]["system"] and seen["body"]["messages"][-1]["role"] == "user"
    assert len(seen["body"]["messages"]) == 3  # one shot (user, assistant) + query
