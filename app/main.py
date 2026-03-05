import os

from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "distilgpt2")


def _read_max_input_chars(default=1000):
    """Read MAX_INPUT_CHARS from env and fall back safely when invalid."""
    raw_value = os.getenv("MAX_INPUT_CHARS", str(default))
    try:
        parsed_value = int(raw_value)
    except (TypeError, ValueError):
        return default

    if parsed_value <= 0:
        return default

    return parsed_value


MAX_INPUT_CHARS = _read_max_input_chars()
_tokenizer = None
_model = None
_model_load_error = False


def _get_model_components():
    """Lazily initialize and cache model dependencies."""
    global _tokenizer, _model, _model_load_error

    if _model_load_error:
        raise RuntimeError("model unavailable")

    if _tokenizer is None or _model is None:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            _model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        except (ImportError, OSError, RuntimeError, ValueError):
            _model_load_error = True
            raise

    return _tokenizer, _model


def _extract_message(payload):
    """Extract and validate a chat message from a JSON payload."""
    if not isinstance(payload, dict):
        return None

    raw_message = payload.get("message")
    if not isinstance(raw_message, str):
        return None

    user_message = raw_message.strip()
    if not user_message:
        return None

    return user_message


def _fallback_reply(user_message):
    """Graceful fallback used when model dependencies are unavailable."""
    condensed = " ".join(user_message.split())[:200]
    return f"Lando fallback: I received your message: {condensed}"


def _agent_mode(message):
    lowered = message.lower()
    if any(word in lowered for word in ("plan", "steps", "todo")):
        return "planner"
    if any(word in lowered for word in ("summarize", "summary", "tl;dr")):
        return "summarizer"
    return "chat"


def _agent_response(user_message):
    """Stage-2 lightweight agent orchestration with mode routing metadata."""
    mode = _agent_mode(user_message)
    reply = _generate_reply(user_message)
    return {"reply": reply, "agent": {"mode": mode, "version": "2"}}


def _generate_reply(user_message):
    """Generate a chat reply for a validated message."""
    try:
        tokenizer, model = _get_model_components()
        inputs = tokenizer(user_message, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=50)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except (ImportError, OSError, RuntimeError, ValueError):
        return _fallback_reply(user_message)


@app.route("/", methods=["GET"])
def index():
    return """
    <html>
      <head><title>Tiny-ando API</title></head>
      <body>
        <h1>Tiny-ando API</h1>
        <p>Status: ok</p>
        <ul>
          <li><a href="/healthz">GET /healthz</a></li>
          <li>POST /chat</li>
          <li>POST /agent/chat</li>
        </ul>
      </body>
    </html>
    """


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok"})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    user_message = _extract_message(data)

    if user_message is None:
        return jsonify({"error": "message is required"}), 400

    if len(user_message) > MAX_INPUT_CHARS:
        return jsonify({"error": f"message exceeds {MAX_INPUT_CHARS} characters"}), 400

    reply = _generate_reply(user_message)
    return jsonify({"reply": reply})


@app.route("/agent/chat", methods=["POST"])
def agent_chat():
    data = request.get_json(silent=True)
    user_message = _extract_message(data)

    if user_message is None:
        return jsonify({"error": "message is required"}), 400

    if len(user_message) > MAX_INPUT_CHARS:
        return jsonify({"error": f"message exceeds {MAX_INPUT_CHARS} characters"}), 400

    return jsonify(_agent_response(user_message))


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
