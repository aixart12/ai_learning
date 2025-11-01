# app.py
import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from agents import build_multi_agent_graph

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    print("Warning: OPENAI_API_KEY not set. Set it in .env for real runs.")

app = Flask(__name__)

# Build the compiled agent graph once (in memory)
agent_pipeline = build_multi_agent_graph()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "multi-agent-flask"}), 200

@app.route("/generate", methods=["POST"])
def generate():
    """
    POST JSON: { "topic": "Explain LangGraph", "max_steps": 1 (optional) }
    Returns: { "topic": "...", "final": "...", "draft": "...", "feedback": "..." }
    """
    payload = request.get_json(force=True, silent=True) or {}
    topic = payload.get("topic") or payload.get("query")
    if not topic:
        return jsonify({"error": "missing 'topic' in body"}), 400

    # Prepare initial state
    state = {"query": topic}

    try:
        # Invoke the compiled graph synchronously
        result_state = agent_pipeline.invoke(state)
    except Exception as e:
        # Catch LLM/tool errors and return useful message
        return jsonify({"error": "agent_error", "detail": str(e)}), 500

    # Normalize response pieces
    draft = result_state.get("draft_article") or ""
    reviewed = result_state.get("reviewed_article") or ""
    research = result_state.get("research_summary") or ""

    return jsonify({
        "topic": topic,
        "research_summary": research,
        "draft": draft,
        "final": reviewed
    }), 200

if __name__ == "__main__":
    # For local dev only. In production run with gunicorn.
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)