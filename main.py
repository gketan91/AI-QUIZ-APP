from flask import Flask, request, jsonify, render_template
import os
import json
import vertexai
from vertexai.generative_models import GenerativeModel

# Quiz defaults
TOPIC = "DevOPS"
NUM_Q = 5
DIFF = "intermediate"
LANG = "English"
MODEL = "gemini-2.0-flash-001"

PROMPT = """
Generate a quiz according to the following specifications:

- topic: {topic}
- num_q: {num_q}
- diff:  {diff}
- lang:  {lang}

Output should be (only) an unquoted json array of objects with keys "question", "responses", and "correct".
"""

app = Flask(__name__)
PORT = int(os.environ.get("PORT", 8080))

# Init Vertex AI
vertexai.init(project="nice-forge-458005-b6", location="us-central1")
parameters = {
    "candidate_count": 1,
    "max_output_tokens": 1024,
    "temperature": 0.5,
    "top_p": 0.8,
    "top_k": 40,
}
model = GenerativeModel(MODEL)

def check(args, name, default):
    return args.get(name, default)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/generate", methods=["GET"])
def generate():
    args = request.args.to_dict()
    topic = check(args, "topic", TOPIC)
    num_q = check(args, "num_q", NUM_Q)
    diff = check(args, "diff", DIFF)
    lang = check(args, "lang", LANG)

    prompt = PROMPT.format(topic=topic, num_q=num_q, diff=diff, lang=lang)
    response = model.generate_content(prompt, generation_config=parameters)
    print("Model Output:", response.text)

    try:
        quiz_json = json.loads(response.text)
        return jsonify(quiz_json)
    except json.JSONDecodeError as e:
        return jsonify({
            "error": "Invalid JSON returned from model.",
            "details": str(e),
            "raw_response": response.text
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
