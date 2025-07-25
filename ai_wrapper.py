"""
AI-powered translation wrapper for Varada's Military Skills Translator.
"""

import os
import openai
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask and CORS
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

# Set OpenAI API Key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Get Assistant ID from environment or use default
DEFAULT_ASSISTANT_ID = "asst_8B38DoipFqPGw7fx1xllFSPR"
ASSISTANT_ID = os.environ.get("ASSISTANT_ID", DEFAULT_ASSISTANT_ID)

def run_assistant(mos_code: str) -> str:
    """Send the given code to the assistant and return its response text."""
    thread = openai.beta.threads.create()
    openai.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=mos_code
    )
    openai.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=ASSISTANT_ID
    )
    messages = openai.beta.threads.messages.list(thread_id=thread.id)
    if not messages.data:
        return ""
    return messages.data[0].content[0].text.value

@app.route('/api/translate', methods=['POST', 'GET'])
def translate():
    """Endpoint for translating a military code via the AI assistant."""
    if request.method == 'GET':
        code = request.args.get('code', '').strip().upper()
    else:
        payload = request.get_json(force=True) or {}
        code = str(payload.get('code', '')).strip().upper()

    if not code:
        return jsonify({"notFound": True})

    try:
        ai_reply = run_assistant(code)
    except Exception as exc:
        return jsonify({"notFound": True, "error": str(exc)})

    if not ai_reply or "no match" in ai_reply.lower() or len(ai_reply.strip()) < 40:
        return jsonify({"notFound": True})

    print(f"AI response: {ai_reply}")
    return jsonify({"reply": ai_reply})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
