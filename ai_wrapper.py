"""
AI-powered translation wrapper for Varada's Military Skills Translator.

This Flask app provides an endpoint that accepts a military occupational
code (MOS/AFSC/NEC) and optionally a branch. It forwards the code to
an OpenAI Assistant that has been configured with the O*NET crosswalk
CSV. The assistant looks up the code, returns the official military
title, suggests civilian roles, and provides a short explanation of
transferable skills.

To use this wrapper you must set two environment variables:

* OPENAI_API_KEY: your secret OpenAI API key
* ASSISTANT_ID: the identifier of your configured assistant. If not
  provided, it will default to the value hardcoded below.

The endpoint accepts POST requests at /api/translate with a JSON
payload containing the code (and optionally a branch). It returns
JSON with either a "notFound" flag or the assistant's reply text.

Example:

    POST /api/translate
    {
        "code": "25B",
        "branch": "Army"
    }

Response:

    {
        "reply": "25B – Information Technology Specialist (Army)\n\nCivilian Role Matches:\n• Network Support Technician\n• Systems Administrator\n• IT Help Desk Analyst\n\nWhy this fits:\nAs a 25B, you’ve configured secure networks, managed communication systems, and supported mission‑critical tech in high‑pressure environments. These skills directly transfer into IT operations, cybersecurity, and systems‑admin roles in the civilian world."
    }

If the assistant returns the fallback message indicating no match was
found, the wrapper sets {"notFound": true} so the front-end can
present its own fallback UI.

Run this server locally with:

    export OPENAI_API_KEY=sk-...
    export ASSISTANT_ID=asst_...
    python ai_wrapper.py

Then set AI_ENDPOINT in your HTML to http://localhost:8000/api/translate
or the URL where this service is deployed.
"""

import os
import openai
from flask import Flask, request, jsonify

# Initialise OpenAI client. The API key must be provided via the
# OPENAI_API_KEY environment variable. Do not hardcode your key here.
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Use the assistant ID from environment or fall back to a default.
# This default corresponds to the assistant created by the user.
DEFAULT_ASSISTANT_ID = "asst_9ldvYz8zGDJStpcVvBgEazPD"
ASSISTANT_ID = os.environ.get("ASSISTANT_ID", DEFAULT_ASSISTANT_ID)


app = Flask(__name__)


def run_assistant(mos_code: str) -> str:
    """Send the given code to the assistant and return its response text."""
    # Create a new thread for each request. Threads allow the
    # assistant to maintain context per user, but here we use a
    # transient thread since each lookup is independent.
    thread = openai.beta.threads.create()
    # Add the user's code as a message.
    openai.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=mos_code
    )
    # Run the assistant and block until completion.
    openai.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=ASSISTANT_ID
    )
    # Retrieve the latest message from the assistant.
    messages = openai.beta.threads.messages.list(thread_id=thread.id)
    if not messages.data:
        return ""
    return messages.data[0].content[0].text.value


@app.route('/api/translate', methods=['POST'])
def translate():
    """Endpoint for translating a military code via the AI assistant."""
    payload = request.get_json(force=True) or {}
    code = str(payload.get('code', '')).strip()
    if not code:
        return jsonify({"notFound": True})
    try:
        ai_reply = run_assistant(code)
    except Exception as exc:
        # If any error occurs during the API call, return notFound
        # The front-end will handle this gracefully.
        return jsonify({"notFound": True, "error": str(exc)})
    # Check for fallback phrase. If present, signal notFound.
    fallback_phrase = "We’re still expanding our translator"
    if fallback_phrase in ai_reply:
        return jsonify({"notFound": True})
    return jsonify({"reply": ai_reply})


if __name__ == '__main__':
    # Bind to all interfaces on port 8000. You can adjust the port
    # for production deployments or use a WSGI server like gunicorn.
    app.run(host='0.0.0.0', port=8000)