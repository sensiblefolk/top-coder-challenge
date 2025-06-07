#!/usr/bin/env python3
"""
Conservative Travel Reimbursement Calculator
Uses ensemble KNN with adaptive k-values and regularization
"""

import sys
import json
import math
from collections import defaultdict

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

# Pre-compute normalization parameters with regularization
def compute_stats():
    """Compute mean and std for normalization with regularization"""
    features = ['days', 'miles', 'receipts', 'sqrt_receipts', 'log_receipts', 'miles_per_day', 'receipts_per_day']
    stats = {}
    
    for feat in features:
        values = [case[feat] for case in TRAINING_DATA]
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        # Add small regularization to std to prevent division by very small numbers
        std = math.sqrt(variance) if variance > 0 else 1.0
        std = max(std, 0.1)  # Minimum std to prevent over-weighting
        stats[feat] = {'mean': mean, 'std': std}
    
    return stats

STATS = compute_stats()

# Pre-compute data density for adaptive k-values
def compute_density_map():
    """Create a density map to identify sparse regions"""
    density_map = defaultdict(int)
    
    for case in TRAINING_DATA:
        # Create discretized buckets
        days_bucket = case['days']
        miles_bucket = int(case['miles'] / 100) * 100
        receipts_bucket = int(case['receipts'] / 200) * 200
        
        key = (days_bucket, miles_bucket, receipts_bucket)
        density_map[key] += 1
    
    return dict(density_map)

DENSITY_MAP = compute_density_map()

def normalize_value(value, feat_name):
    """Normalize a single value using pre-computed stats"""
    return (value - STATS[feat_name]['mean']) / STATS[feat_name]['std']

def euclidean_distance(case1, case2):
    """Calculate normalized Euclidean distance with feature weighting"""
    features = ['days', 'miles', 'receipts', 'sqrt_receipts', 'log_receipts', 'miles_per_day', 'receipts_per_day']
    
    # Feature weights based on importance (from ML analysis)
    weights = {
        'days': 1.0,
        'miles': 1.0,
        'receipts': 1.2,
        'sqrt_receipts': 1.5,  # Most important feature
        'log_receipts': 1.3,
        'miles_per_day': 0.8,
        'receipts_per_day': 0.8
    }
    
    distance_sq = 0
    for feat in features:
        diff = normalize_value(case1[feat], feat) - normalize_value(case2[feat], feat)
        distance_sq += weights[feat] * (diff * diff)
    
    return math.sqrt(distance_sq)

def get_adaptive_k(days, miles, receipts):
    """Determine k value based on local data density"""
    # Check density in local region
    days_bucket = days
    miles_bucket = int(miles / 100) * 100
    receipts_bucket = int(receipts / 200) * 200
    
    key = (days_bucket, miles_bucket, receipts_bucket)
    local_density = DENSITY_MAP.get(key, 0)
    
    # Check nearby buckets
    nearby_density = 0
    for d_offset in [-1, 0, 1]:
        for m_offset in [-100, 0, 100]:
            for r_offset in [-200, 0, 200]:
                nearby_key = (
                    days_bucket + d_offset,
                    miles_bucket + m_offset,
                    receipts_bucket + r_offset
                )
                nearby_density += DENSITY_MAP.get(nearby_key, 0)
    
    # Adaptive k based on density
    if local_density >= 5:
        return 3  # Dense region, use smaller k
    elif nearby_density >= 20:
        return 5  # Medium density, use medium k
    elif nearby_density >= 10:
        return 7  # Sparse region, use larger k
    else:
        return 10  # Very sparse, use maximum k

def knn_predict_conservative(days, miles, receipts):
    """Conservative prediction using ensemble of different k values"""
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
    
    # Sort by distance
    distances.sort(key=lambda x: x[0])
    
    # Get adaptive k
    adaptive_k = get_adaptive_k(days, miles, receipts)
    
    # Ensemble predictions with different k values
    k_values = [3, 5, adaptive_k, min(10, adaptive_k + 2)]
    predictions = []
    weights = []
    
    for k in k_values:
        k_nearest = distances[:k]
        
        # Handle exact matches
        if k_nearest[0][0] < 1e-10:
            return k_nearest[0][1]
        
        # Weighted average based on inverse distance
        weighted_sum = 0
        weight_total = 0
        
        for dist, output in k_nearest:
            # Use inverse square distance for stronger locality
            weight = 1.0 / ((dist + 0.01) ** 2)
            weighted_sum += weight * output
            weight_total += weight
        
        prediction = weighted_sum / weight_total
        predictions.append(prediction)
        
        # Weight ensemble members by their k value (prefer smaller k in dense regions)
        ensemble_weight = 1.0 / math.sqrt(k) if local_density > 10 else math.sqrt(k)
        weights.append(ensemble_weight)
    
    # Calculate weighted ensemble average
    total_weight = sum(weights)
    ensemble_prediction = sum(p * w for p, w in zip(predictions, weights)) / total_weight
    
    # Apply conservative smoothing based on distance to nearest neighbors
    min_dist = distances[0][0]
    if min_dist > 0.5:  # Far from training data
        # Blend with global average for very distant points
        global_avg = sum(case['output'] for case in TRAINING_DATA) / len(TRAINING_DATA)
        smoothing_factor = min(0.3, min_dist / 10)
        ensemble_prediction = (1 - smoothing_factor) * ensemble_prediction + smoothing_factor * global_avg
    
    return ensemble_prediction

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 calculate_reimbursement_conservative.py <days> <miles> <receipts>")
        sys.exit(1)
    
    days = int(sys.argv[1])
    miles = float(sys.argv[2])
    receipts = float(sys.argv[3])
    
    # Check if this is within reasonable bounds based on training data
    max_days = max(case['days'] for case in TRAINING_DATA)
    max_miles = max(case['miles'] for case in TRAINING_DATA)
    max_receipts = max(case['receipts'] for case in TRAINING_DATA)
    
    if days > max_days * 1.5 or miles > max_miles * 1.5 or receipts > max_receipts * 1.5:
        print(f"# Warning: Input significantly outside training range", file=sys.stderr)
    
    result = knn_predict_conservative(days, miles, receipts)
    print(round(result, 2))

if __name__ == '__main__':
    # Pre-compute local density for the query point
    if len(sys.argv) == 4:
        days = int(sys.argv[1])
        miles = float(sys.argv[2])
        receipts = float(sys.argv[3])
        
        days_bucket = days
        miles_bucket = int(miles / 100) * 100
        receipts_bucket = int(receipts / 200) * 200
        key = (days_bucket, miles_bucket, receipts_bucket)
        local_density = DENSITY_MAP.get(key, 0)
    else:
        local_density = 0
    
    main() 