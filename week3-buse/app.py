from flask import Flask, request, jsonify, render_template
from retrieval_aug_promp import handle_interaction  # Adjust this import based on your module structure

app = Flask(__name__)

# Store chat history in-memory (not persistent)
chat_history = []

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", chat_history=chat_history)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form.get("user_input")
    # Handle interaction using your existing function
    output = handle_interaction(user_input)
    
    # Update chat history for the UI
    chat_history = output['chat_history']
    
    return jsonify({
        "role": output['role'],
        "user": user_input,
        "assistant": output['answer'],
        "chat_history": chat_history
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
