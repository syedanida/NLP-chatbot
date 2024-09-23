from flask import Flask, request, jsonify, render_template, session # type: ignore
from flask_cors import CORS # type: ignore
from chatbot import answer_question

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.secret_key = 'your_secret_key'  # Replace with a secure key

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')

    # Store previous questions in session
    if 'history' not in session:
        session['history'] = []
    session['history'].append(question)

    answer = answer_question(question)
    return jsonify({'answer': answer})

@app.route('/history', methods=['GET'])
def history():
    return jsonify({'history': session.get('history', [])})

if __name__ == '__main__':
    app.run(debug=True)

# improvements: Should send when click 'enter', 
# then tag as 'history' for the below, then make it more interacting