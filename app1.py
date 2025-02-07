from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os

app = Flask(__name__)

# Load the saved model and tokenizer
model_path = "bert_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Set device (cuda if available, otherwise cpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.get_json()
        review = data['review']

        # Tokenize the review text
        inputs = tokenizer(review, return_tensors="pt", padding=True, truncation=True)
        inputs.to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Get predicted label
        predicted_label = torch.argmax(outputs.logits).item()

        # Mapping predicted labels to sentiments
        sentiments = {0: 'Negative', 1: 'Positive'}
        sentiment = sentiments[predicted_label]

        return jsonify({'sentiment': sentiment})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_multiple_sentiments', methods=['POST'])
def analyze_multiple_sentiments():
    try:
        # Ensure a file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded.'}), 400

        file = request.files['file']

        if not file or file.filename == '':
            return jsonify({'error': 'Invalid file.'}), 400

        # Read the text file, split into lines (assuming each line is a review)
        reviews = file.read().decode('utf-8').splitlines()

        results = []

        for review in reviews:
            # Tokenize the review text
            inputs = tokenizer(review, return_tensors="pt", padding=True, truncation=True)
            inputs.to(device)

            # Perform inference
            with torch.no_grad():
                outputs = model(**inputs)

            # Get predicted label
            predicted_label = torch.argmax(outputs.logits).item()

            # Mapping predicted labels to sentiments
            sentiments = {0: 'Negative', 1: 'Positive'}
            sentiment = sentiments[predicted_label]

            results.append({'review': review, 'sentiment': sentiment})

        return jsonify({'results': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
