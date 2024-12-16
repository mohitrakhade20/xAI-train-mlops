from flask import Flask, request, jsonify, render_template_string
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Dataset class for training
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train_model():
    # Sample training data
    texts = [
        "This product is amazing!",
        "Terrible service, would not recommend",
        "Okay experience, nothing special",
        # Add more examples in practice
    ]
    labels = [1, 0, 1]  # 1 for positive, 0 for negative

    # Initialize model and tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2
    )

    # Create datasets
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)

    # Training settings
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Training loop
    model.train()
    for epoch in range(3):
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(**{k: v for k, v in batch.items()})
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    # Save model
    model.save_pretrained("./sentiment_model")
    tokenizer.save_pretrained("./sentiment_model")

app = Flask(__name__)

# HTML template with embedded CSS
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Sentiment Analysis</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        textarea {
            width: 100%;
            height: 150px;
            padding: 12px;
            margin: 10px 0;
            border: 1px solid #ddd;
            border-radius: 4px;
            resize: vertical;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 12px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            width: 100%;
            font-size: 16px;
        }
        button:hover {
            background-color: #45a049;
        }
        #result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 4px;
            display: none;
        }
        .positive {
            background-color: #dff0d8;
            border: 1px solid #d6e9c6;
            color: #3c763d;
        }
        .negative {
            background-color: #f2dede;
            border: 1px solid #ebccd1;
            color: #a94442;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Sentiment Analysis</h1>
        <textarea id="text" placeholder="Enter text to analyze..."></textarea>
        <button onclick="analyzeSentiment()">Analyze Sentiment</button>
        <div id="result"></div>
    </div>

    <script>
        async function analyzeSentiment() {
            const text = document.getElementById('text').value;
            const result = document.getElementById('result');
            
            if (!text) return;
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ text })
                });
                
                const data = await response.json();
                
                result.style.display = 'block';
                result.className = data.sentiment;
                result.innerHTML = `
                    <strong>Sentiment:</strong> ${data.sentiment}<br>
                    <strong>Confidence:</strong> ${(data.confidence * 100).toFixed(1)}%
                `;
            } catch (error) {
                result.style.display = 'block';
                result.className = '';
                result.innerHTML = 'Error analyzing sentiment';
            }
        }
    </script>
</body>
</html>
"""

# Load the model (done at startup)
sentiment_pipeline = None
try:
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="./sentiment_model",
        tokenizer="./sentiment_model"
    )
except:
    print("No trained model found. Please run training first.")
    sentiment_pipeline = pipeline("sentiment-analysis")  # Use default model as fallback

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

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