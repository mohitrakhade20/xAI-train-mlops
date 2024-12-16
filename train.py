import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

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
