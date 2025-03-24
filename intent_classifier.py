import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os

def load_model():
    if not os.path.exists("intent_model.pth"):
        print("🚨 Model file not found! Train the model first.")
        return None  # Return None if model is missing

    model = CNNIntentClassifier(256, 50, num_classes, max_length=30)
    model.load_state_dict(torch.load("intent_model.pth"))
    model.eval()
    return model


# Load dataset
print("🔄 Loading dataset...")
df = pd.read_excel("total_dataset.xlsx")

df["intent"] = df["intent"].astype('category')
df["intent_codes"] = df["intent"].cat.codes  # Numerical labels
labels = df["intent_codes"].tolist()

if len(set(labels)) < 2:
    print("🚨 ERROR: Only one unique class found in dataset!")
    exit()

num_classes = max(labels) + 1
texts = df["utterance"].tolist()
label_mapping = dict(enumerate(df["intent"].astype('category').cat.categories))
intent_to_response = df.set_index("intent")["responses"].to_dict()

# Dataset class
class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=30):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokenized = self.tokenizer(text)[:self.max_length]
        tokenized += [0] * (self.max_length - len(tokenized))  # Padding
        return torch.tensor(tokenized, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# CNN Model
class CNNIntentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, max_length=30):
        super(CNNIntentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(embed_dim, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc = nn.Linear(128 * (max_length // 2), num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.pool(torch.relu(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

def train_and_save_model():
    print("🛠 Training intent classifier...")

    tokenizer = lambda text: [min(ord(c), 255) for c in text]
    dataset = IntentDataset(texts, labels, tokenizer, max_length=30)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    model = CNNIntentClassifier(256, 50, num_classes, max_length=30).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(50):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), "intent_model.pth")
    print("✅ Model trained and saved.")

def load_model():
    if not os.path.exists("intent_model.pth"):
        print("🚨 Model file not found! Train the model first.")
        return None  # Return None if model is missing

    model = CNNIntentClassifier(256, 50, num_classes, max_length=30)
    model.load_state_dict(torch.load("intent_model.pth"))
    model.eval()
    return model

intent_model = load_model()

def predict_intent(text):
    tokenizer = lambda text: [min(ord(c), 255) for c in text[:30]]
    input_tensor = torch.tensor([tokenizer(text)], dtype=torch.long)
    
    with torch.no_grad():
        output = intent_model(input_tensor)
    
    predicted_intent = torch.argmax(output, dim=1).item()
    return label_mapping.get(predicted_intent, "Unknown Intent")
