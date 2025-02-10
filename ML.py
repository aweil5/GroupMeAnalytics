import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_clusters(features, clusters, user_ids=None):
    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=clusters, cmap='viridis')
    plt.colorbar(scatter)
    
    with open("analytics.json", "r") as f:
        main_group_info = json.load(f)
    # Add labels if provided
    if user_ids:
        for i, user_id in enumerate(user_ids):
            if user_id in main_group_info['users']:
                annotate = main_group_info['users'][user_id]['real_name'] if 'real_name' in main_group_info['users'][user_id] else user_id
                plt.annotate(annotate, (features_2d[i, 0], features_2d[i, 1]))
            else:
                plt.annotate(user_id, (features_2d[i, 0], features_2d[i, 1]))
    
    plt.title('User Clusters')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()

def prepare_user_features(analytics):
    features = []
    user_ids = []
    
    for user_id, stats in analytics['users'].items():
        user_features = [
            stats['message_count'],
            stats['total_likes'],
            stats['avg_likes_per_message'],
            stats['total_replies_received'],
            stats['messages_with_media'],
            stats['longest_unliked_streak'],
            stats['liked_by_count'],
            stats['likes_given_count']
        ]
        
        # Add hourly activity distribution
        
        features.append(user_features)
        user_ids.append(user_id)
    
    return np.array(features), user_ids

def cluster_users(analytics, n_clusters=4):
    features, user_ids = prepare_user_features(analytics)
    
    # Normalize features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(normalized_features)
    
    # Add cluster assignments to analytics
    for user_id, cluster in zip(user_ids, clusters):
        analytics['users'][user_id]['cluster'] = int(cluster)
    
    return kmeans, scaler

def prepare_message_features(message, user_stats):
    return [
        len(message.get('text', '')) if message.get('text') else 0,
        len(message.get('attachments', [])),
        len(message.get('replies', [])),
        user_stats['avg_likes_per_message'],
        user_stats['avg_replies_per_message'],
        user_stats['total_likes'],
        user_stats['message_count']
    ]

def train_message_quality_model(analytics, messages, quality_threshold=0.7):
    X = []
    y = []
    
    for message in messages:
        user_id = message['user_id']
        if user_id not in analytics['users']:
            continue
            
        user_stats = analytics['users'][user_id]
        features = prepare_message_features(message, user_stats)
        
        # Define high-quality messages based on likes and replies
        likes = len(message.get('favorited_by', []))
        replies = len(message.get('replies', []))
        max_likes = user_stats['most_liked_message_likes']
        quality_score = (likes / max_likes if max_likes > 0 else 0)
        is_high_quality = quality_score >= quality_threshold
        
        X.append(features)
        y.append(int(is_high_quality))
    
    X = np.array(X)
    y = np.array(y)
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    return model

import pickle

def save_models(kmeans_model, scaler, quality_model, base_path='models/'):
    # Create models directory if it doesn't exist
    import os
    os.makedirs(base_path, exist_ok=True)
    
    # Save models
    with open(f'{base_path}kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans_model, f)
    
    with open(f'{base_path}scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    with open(f'{base_path}quality_model.pkl', 'wb') as f:
        pickle.dump(quality_model, f)

def load_models(base_path='models/'):
    with open(f'{base_path}kmeans_model.pkl', 'rb') as f:
        kmeans_model = pickle.load(f)
    
    with open(f'{base_path}scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    with open(f'{base_path}quality_model.pkl', 'rb') as f:
        quality_model = pickle.load(f)
        
    return kmeans_model, scaler, quality_model

def predict_message_quality(message, user_stats, model):
    features = prepare_message_features(message, user_stats)
    return model.predict_proba([features])[0][1]  # Probability of high quality


if __name__ == "__main__":
    
    # Load data
    print("Loading data...")
    with open("analytics.json", "r") as f:
        analytics = json.load(f)
    
    with open("messages.json", "r") as f:
        messages = json.load(f)
    
    # Train clustering model
    print("Training models...")
    kmeans_model, scaler = cluster_users(analytics)
    
    # Visualize clusters
    print("Visualizing clusters...")
    features, user_ids = prepare_user_features(analytics)
    print(user_ids)
    visualize_clusters(features, kmeans_model.labels_, user_ids)
    
    # Train quality model
    quality_model = train_message_quality_model(analytics, messages)
    
    # Save all models
    save_models(kmeans_model, scaler, quality_model)
    
    # Optional: Test prediction on a sample message
    # if messages:
    #     sample_message = messages[0]
    #     user_stats = analytics['users'][sample_message['user_id']]
    #     quality_prob = predict_message_quality(sample_message, user_stats, quality_model)
    #     print(f"Sample message quality probability: {quality_prob:.2f}")