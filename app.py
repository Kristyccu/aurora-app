from transformers import pipeline
from flask import Flask, request, jsonify, render_template
import uuid
import os

app = Flask(__name__, template_folder="templates")

# Load conversational model (lightweight version)
chatbot = pipeline("text-generation", model="microsoft/DialoGPT-small")

# Store simple conversation history
conversations = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_conversation():
    user_id = str(uuid.uuid4())
    conversations[user_id] = []
    return jsonify({"user_id": user_id, "message": "Hi there! How are you feeling today?"})

@app.route('/message', methods=['POST'])
def handle_message():
    data = request.get_json()
    user_id = data['user_id']
    user_input = data['message']

    if user_id not in conversations:
        return jsonify({"error": "Invalid session."}), 400

    conversations[user_id].append(f"User: {user_input}")

    full_context = "\n".join(conversations[user_id]) + "\nAI:"
    response = chatbot(full_context, max_length=100, pad_token_id=50256, do_sample=True, top_k=50)[0]["generated_text"]

    bot_reply = response.split("AI:")[-1].strip()
    conversations[user_id].append(f"AI: {bot_reply}")

    return jsonify({"response": bot_reply})

@app.route('/consent', methods=['POST'])
def handle_consent():
    data = request.get_json()
    consent = data['consent']
    if consent:
        return jsonify({"message": "Thank you. A school counsellor will reach out to you shortly."})
    else:
        return jsonify({"message": "No problem. I'm here if you want to talk again."})

if __name__ == '__main__':
    app.run(debug=True)
