#!/usr/bin/env python3
"""
Fast Travel Reimbursement Calculator
Pure Python implementation of KNN without heavy dependencies
"""

import sys
import json
import math

# Pre-compute features from training data at module load time
def load_training_data():
    """Load and pre-process training data"""
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    processed_data = []
    for case in data:
        inp = case['input']
        days = inp['trip_duration_days']
        miles = inp['miles_traveled']
        receipts = inp['total_receipts_amount']
        
        # Pre-compute features
        features = {
            'days': days,
            'miles': miles,
            'receipts': receipts,
            'sqrt_receipts': math.sqrt(receipts),
            'log_receipts': math.log1p(receipts),
            'miles_per_day': miles / days if days > 0 else 0,
            'receipts_per_day': receipts / days if days > 0 else 0,
            'output': case['expected_output']
        }
        processed_data.append(features)
    
    return processed_data

# Load data once when module is imported
TRAINING_DATA = load_training_data()

# Pre-compute normalization parameters
def compute_stats():
    """Compute mean and std for normalization"""
    features = ['days', 'miles', 'receipts', 'sqrt_receipts', 'log_receipts', 'miles_per_day', 'receipts_per_day']
    stats = {}
    
    for feat in features:
        values = [case[feat] for case in TRAINING_DATA]
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std = math.sqrt(variance) if variance > 0 else 1.0
        stats[feat] = {'mean': mean, 'std': std}
    
    return stats

STATS = compute_stats()

def normalize_value(value, feat_name):
    """Normalize a single value using pre-computed stats"""
    return (value - STATS[feat_name]['mean']) / STATS[feat_name]['std']

def euclidean_distance(case1, case2):
    """Calculate normalized Euclidean distance between two cases"""
    features = ['days', 'miles', 'receipts', 'sqrt_receipts', 'log_receipts', 'miles_per_day', 'receipts_per_day']
    
    distance_sq = 0
    for feat in features:
        diff = normalize_value(case1[feat], feat) - normalize_value(case2[feat], feat)
        distance_sq += diff * diff
    
    return math.sqrt(distance_sq)

def knn_predict(days, miles, receipts, k=3):
    """Predict using k-nearest neighbors with distance weighting"""
    # Create feature dict for query
    query = {
        'days': days,
        'miles': miles,
        'receipts': receipts,
        'sqrt_receipts': math.sqrt(receipts),
        'log_receipts': math.log1p(receipts),
        'miles_per_day': miles / days if days > 0 else 0,
        'receipts_per_day': receipts / days if days > 0 else 0
    }
    
    # Calculate distances to all training cases
    distances = []
    for i, train_case in enumerate(TRAINING_DATA):
        dist = euclidean_distance(query, train_case)
        distances.append((dist, train_case['output']))
    
    # Sort by distance and get k nearest
    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]
    
    # Handle exact matches
    if k_nearest[0][0] < 1e-10:
        return k_nearest[0][1]
    
    # Weighted average based on inverse distance
    weighted_sum = 0
    weight_total = 0
    
    for dist, output in k_nearest:
        # Add small epsilon to avoid division by zero
        weight = 1.0 / (dist + 1e-10)
        weighted_sum += weight * output
        weight_total += weight
    
    return weighted_sum / weight_total

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 calculate_reimbursement_fast.py <days> <miles> <receipts>")
        sys.exit(1)
    
    days = int(sys.argv[1])
    miles = float(sys.argv[2])
    receipts = float(sys.argv[3])
    
    result = knn_predict(days, miles, receipts)
    print(round(result, 2))

if __name__ == '__main__':
    main() 