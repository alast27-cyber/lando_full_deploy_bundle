from app.main import app, MAX_INPUT_CHARS


def test_index_route():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.content_type
    assert "Tiny-ando API" in response.get_data(as_text=True)
    assert "/healthz" in response.get_data(as_text=True)
    assert "/chat" in response.get_data(as_text=True)


def test_healthz():
    client = app.test_client()
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json == {"status": "ok"}


def test_chat_requires_message():
    client = app.test_client()

    response = client.post("/chat", json={})
    assert response.status_code == 400
    assert response.json == {"error": "message is required"}

    response = client.post("/chat", json={"message": "   "})
    assert response.status_code == 400
    assert response.json == {"error": "message is required"}

    response = client.post("/chat", json={"message": 123})
    assert response.status_code == 400
    assert response.json == {"error": "message is required"}


def test_chat_rejects_non_object_json_payload():
    client = app.test_client()

    response = client.post("/chat", json=["hello"])
    assert response.status_code == 400
    assert response.json == {"error": "message is required"}


def test_chat_rejects_overly_long_message():
    client = app.test_client()
    response = client.post("/chat", json={"message": "x" * (MAX_INPUT_CHARS + 1)})
    assert response.status_code == 400
    assert response.json == {"error": f"message exceeds {MAX_INPUT_CHARS} characters"}


def test_chat_uses_fallback_if_model_unavailable(monkeypatch):
    client = app.test_client()

    def _boom():
        raise RuntimeError("model init failed")

    monkeypatch.setattr("app.main._get_model_components", _boom)

    response = client.post("/chat", json={"message": "hello"})
    assert response.status_code == 200
    assert "Lando fallback" in response.json["reply"]


def test_chat_returns_generated_reply(monkeypatch):
    client = app.test_client()

    monkeypatch.setattr("app.main.USE_TRANSFORMERS_MODEL", True)
    monkeypatch.setattr("app.main._generate_reply", lambda _: "hi from lando")

    response = client.post("/chat", json={"message": "hello"})
    assert response.status_code == 200
    assert response.json == {"reply": "hi from lando"}


def test_agent_chat_returns_mode_metadata(monkeypatch):
    client = app.test_client()
    monkeypatch.setattr("app.main.USE_TRANSFORMERS_MODEL", True)
    monkeypatch.setattr("app.main._generate_reply", lambda _: "planned")

    response = client.post("/agent/chat", json={"message": "please plan the next steps"})
    assert response.status_code == 200
    assert response.json["reply"] == "planned"
    assert response.json["agent"]["mode"] == "planner"
    assert response.json["agent"]["version"] == "2"


def test_agent_chat_requires_message():
    client = app.test_client()
    response = client.post("/agent/chat", json={})
    assert response.status_code == 400
    assert response.json == {"error": "message is required"}


def test_max_input_chars_defaults_when_env_invalid(monkeypatch):
    import importlib
    import app.main as main

    monkeypatch.setenv("MAX_INPUT_CHARS", "not-a-number")
    importlib.reload(main)
    assert main.MAX_INPUT_CHARS == 1000

    monkeypatch.setenv("MAX_INPUT_CHARS", "0")
    importlib.reload(main)
    assert main.MAX_INPUT_CHARS == 1000

    monkeypatch.setenv("MAX_INPUT_CHARS", "42")
    importlib.reload(main)
    assert main.MAX_INPUT_CHARS == 42


def test_agent_chat_summarizer_mode(monkeypatch):
    client = app.test_client()
    monkeypatch.setattr("app.main.USE_TRANSFORMERS_MODEL", True)
    monkeypatch.setattr("app.main._generate_reply", lambda _: "summary")

    response = client.post("/agent/chat", json={"message": "please summarize this"})
    assert response.status_code == 200
    assert response.json["agent"]["mode"] == "summarizer"


def test_agent_chat_defaults_to_chat_mode(monkeypatch):
    client = app.test_client()
    monkeypatch.setattr("app.main.USE_TRANSFORMERS_MODEL", True)
    monkeypatch.setattr("app.main._generate_reply", lambda _: "ok")

    response = client.post("/agent/chat", json={"message": "hello there"})
    assert response.status_code == 200
    assert response.json["agent"]["mode"] == "chat"


def test_chat_falls_back_on_unexpected_generation_error(monkeypatch):
    client = app.test_client()

    def _boom(_):
        raise Exception("unexpected")

    monkeypatch.setattr("app.main._generate_reply", _boom)

    response = client.post("/chat", json={"message": "hello"})
    assert response.status_code == 200
    assert "Lando fallback" in response.json["reply"]


def test_global_error_handler_returns_json_500(monkeypatch):
    client = app.test_client()

    def _explode(_):
        raise Exception("fatal")

    monkeypatch.setattr("app.main._extract_message", _explode)

    response = client.post("/chat", json={"message": "hello"})
    assert response.status_code == 500
    assert response.json == {"error": "internal server error"}


def test_chat_uses_fallback_when_model_disabled(monkeypatch):
    client = app.test_client()

    monkeypatch.setattr("app.main.USE_TRANSFORMERS_MODEL", False)
    def _boom(_):
        raise Exception("should not run")

    monkeypatch.setattr("app.main._generate_reply", _boom)

    response = client.post("/chat", json={"message": "hello"})
    assert response.status_code == 200
    assert "Lando fallback" in response.json["reply"]
