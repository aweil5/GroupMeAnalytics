import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import torch
from transformers import AutoTokenizer, AutoModel
import json

class MessagePopularityPredictor:
    def __init__(self, use_bert=False):
        self.use_bert = use_bert
        if use_bert:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
        else:
            self.model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000)),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])

    def _calculate_popularity_score(self, message):
        """Calculate a popularity score based on favorites and reactions."""
        score = 0
        # Add points for each favorite
        if 'favorited_by' in message:
            score += len(message['favorited_by'])
        
        # Add points for reactions
        if 'reactions' in message:
            for reaction in message['reactions']:
                score += len(reaction['user_ids'])
        
        return score

    def _get_bert_embeddings(self, texts):
        """Get BERT embeddings for a list of texts."""
        embeddings = []
        with torch.no_grad():
            for text in texts:
                # Tokenize and get BERT embeddings
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                outputs = self.bert_model(**inputs)
                # Use the [CLS] token embedding as the text representation
                embeddings.append(outputs.last_hidden_state[:, 0, :].numpy())
        return np.vstack(embeddings)

    def prepare_data(self, messages):
        """Prepare the data for training."""
        texts = []
        scores = []
        
        for message in messages:
            if message['text']:  # Only include messages with text
                texts.append(message['text'])
                scores.append(self._calculate_popularity_score(message))
        
        # Convert scores to binary labels (1 for above median, 0 for below)
        median_score = np.median(scores)
        labels = [1 if score > median_score else 0 for score in scores]
        
        return texts, labels

    def train(self, messages):
        """Train the model on the provided messages."""
        texts, labels = self.prepare_data(messages)
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        if self.use_bert:
            # Get BERT embeddings
            X_train_embeddings = self._get_bert_embeddings(X_train)
            X_test_embeddings = self._get_bert_embeddings(X_test)
            
            # Train Random Forest on BERT embeddings
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            self.classifier.fit(X_train_embeddings, y_train)
            
            # Evaluate
            y_pred = self.classifier.predict(X_test_embeddings)
        else:
            # Train the pipeline
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
        
        print("\nModel Performance:")
        print(classification_report(y_test, y_pred))

    def predict(self, text):
        """Predict if a message will be popular."""
        if self.use_bert:
            embedding = self._get_bert_embeddings([text])
            probability = self.classifier.predict_proba(embedding)[0][1]
        else:
            probability = self.model.predict_proba([text])[0][1]
        
        return {
            'probability': probability,
            'prediction': 'Popular' if probability > 0.5 else 'Not Popular',
            'confidence': f'{probability * 100:.2f}%'
        }
    
def save_models(model, base_path='models/'):

    import os
    os.makedirs(base_path, exist_ok=True)
    
    # Save models
    with open(f'{base_path}MessagePredictor.pkl', 'wb') as f:
        pickle.dump(model, f)

        
# Example usage:
if __name__ == "__main__":
    # Load your messages from a JSON file
    with open('messages.json', 'r') as f:
        messages = json.load(f)
    
    # Initialize and train the model
    # Use basic TF-IDF + Random Forest
    predictor = MessagePopularityPredictor(use_bert=False)
    predictor.train(messages)

    # Save the trained model to a file
    save_models(predictor)
    # Example predictions
    test_messages = [
        "Looking forward to our next meeting!",
        "k",
        "Who's going to the event tomorrow?",
    ]
    
    print("\nExample Predictions:")
    for msg in test_messages:
        result = predictor.predict(msg)
        print(f"\nMessage: {msg}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']}")