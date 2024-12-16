from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load the model
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="./sentiment_model",
    tokenizer="./sentiment_model"
)

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json["text"]
    result = sentiment_pipeline(text)[0]
    return jsonify({
        "text": text,
        "sentiment": "positive" if result["label"] == "LABEL_1" else "negative",
        "confidence": result["score"]
    })

if __name__ == "__main__":
    app.run(debug=True)

