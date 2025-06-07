#!/usr/bin/env python3
"""
Pre-train and save the KNN model for fast inference
"""

import json
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

def load_training_data():
    """Load the public cases as training data"""
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    X = []
    y = []
    
    for case in data:
        inp = case['input']
        days = inp['trip_duration_days']
        miles = inp['miles_traveled']
        receipts = inp['total_receipts_amount']
        
        # Create feature vector with engineered features
        features = [
            days,
            miles,
            receipts,
            np.sqrt(receipts),
            np.log1p(receipts),
            miles / days if days > 0 else 0,
            receipts / days if days > 0 else 0
        ]
        
        X.append(features)
        y.append(case['expected_output'])
    
    return np.array(X), np.array(y)

def train_and_save_model():
    """Train the model and save it along with the scaler"""
    print("Loading training data...")
    X, y = load_training_data()
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Training KNN model...")
    model = KNeighborsRegressor(n_neighbors=3, weights='distance')
    model.fit(X_scaled, y)
    
    print("Saving model and scaler...")
    with open('knn_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)
    
    print("Model saved successfully!")

if __name__ == '__main__':
    train_and_save_model() 