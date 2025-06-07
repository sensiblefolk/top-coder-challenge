#!/usr/bin/env python3
"""
Find exact formula by examining simple cases
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
import math

def load_data():
    """Load and prepare the data"""
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    rows = []
    for case in data:
        row = {
            'days': case['input']['trip_duration_days'],
            'miles': case['input']['miles_traveled'],
            'receipts': case['input']['total_receipts_amount'],
            'reimbursement': case['expected_output']
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Add engineered features
    df['miles_per_day'] = df['miles'] / df['days']
    df['receipts_per_day'] = df['receipts'] / df['days']
    df['log_receipts'] = np.log1p(df['receipts'])
    df['sqrt_receipts'] = np.sqrt(df['receipts'])
    df['miles_squared'] = df['miles'] ** 2
    df['days_squared'] = df['days'] ** 2
    
    return df

def test_formula_1(row):
    """Test a more complex formula based on patterns"""
    days = row['days']
    miles = row['miles']
    receipts = row['receipts']
    
    # Base calculation
    base = 50 * days + 0.45 * miles + 0.38 * receipts + 270
    
    # Apply caps and adjustments
    if receipts > 2000:
        base = base * 0.7  # Heavy penalty for high receipts
    
    if days == 1 and miles > 800:
        base = base * 0.5  # Penalty for excessive 1-day miles
        
    return base

def test_formula_2(row):
    """Test formula with logarithmic receipt processing"""
    days = row['days']
    miles = row['miles']
    receipts = row['receipts']
    
    # Base components
    per_diem = days * 100
    mileage = miles * 0.45
    
    # Logarithmic receipt processing
    if receipts > 0:
        receipt_component = 150 * math.log1p(receipts / 100)
    else:
        receipt_component = 0
    
    return per_diem + mileage + receipt_component

def test_formula_3(row):
    """Test formula with complex interaction effects"""
    days = row['days']
    miles = row['miles']
    receipts = row['receipts']
    miles_per_day = miles / days
    receipts_per_day = receipts / days
    
    # Base calculation with interaction terms
    base = 270 + 50 * days + 0.45 * miles
    
    # Receipt processing with diminishing returns
    if receipts < 100:
        receipt_effect = receipts * 0.4
    elif receipts < 1000:
        receipt_effect = 40 + (receipts - 100) * 0.35
    else:
        receipt_effect = 40 + 900 * 0.35 + (receipts - 1000) * 0.15
    
    # Efficiency adjustments
    if miles_per_day > 200 and days <= 3:
        efficiency_bonus = 50
    else:
        efficiency_bonus = 0
    
    # High receipt penalty
    if receipts_per_day > 300:
        penalty = (receipts_per_day - 300) * days * 0.5
    else:
        penalty = 0
    
    return base + receipt_effect + efficiency_bonus - penalty

def analyze_residuals(df, formula_func, name):
    """Analyze residuals for a given formula"""
    predictions = df.apply(formula_func, axis=1)
    residuals = df['reimbursement'] - predictions
    
    print(f"\n=== {name} ===")
    print(f"Mean Absolute Error: ${abs(residuals).mean():.2f}")
    print(f"Exact matches (±$0.01): {sum(abs(residuals) <= 0.01)}")
    print(f"Close matches (±$1.00): {sum(abs(residuals) <= 1.00)}")
    
    # Show worst cases
    worst_indices = abs(residuals).nlargest(5).index
    print("\nWorst predictions:")
    for idx in worst_indices:
        row = df.iloc[idx]
        pred = predictions.iloc[idx]
        actual = row['reimbursement']
        print(f"  {row['days']}d, {row['miles']:.0f}mi, ${row['receipts']:.2f} → "
              f"Predicted: ${pred:.2f}, Actual: ${actual:.2f}, Error: ${abs(pred-actual):.2f}")
    
    return predictions, residuals

def find_patterns_in_residuals(df, residuals):
    """Look for patterns in the residuals"""
    df['residual'] = residuals
    df['abs_residual'] = abs(residuals)
    
    print("\n=== RESIDUAL PATTERNS ===")
    
    # Group by days
    print("\nAverage residual by trip length:")
    for days in range(1, 15):
        subset = df[df['days'] == days]
        if len(subset) > 5:
            print(f"  {days} days: ${subset['residual'].mean():.2f} (n={len(subset)})")
    
    # Look for receipt patterns
    print("\nAverage residual by receipt range:")
    ranges = [(0, 100), (100, 500), (500, 1000), (1000, 2000), (2000, 3000)]
    for low, high in ranges:
        subset = df[(df['receipts'] >= low) & (df['receipts'] < high)]
        if len(subset) > 0:
            print(f"  ${low}-${high}: ${subset['residual'].mean():.2f} (n={len(subset)})")

def main():
    """Main analysis function"""
    df = load_data()
    
    # Test different formulas
    formulas = [
        (test_formula_1, "Complex Formula with Caps"),
        (test_formula_2, "Logarithmic Receipt Processing"),
        (test_formula_3, "Interaction Effects Formula")
    ]
    
    best_formula = None
    best_error = float('inf')
    best_predictions = None
    
    for formula_func, name in formulas:
        predictions, residuals = analyze_residuals(df, formula_func, name)
        error = abs(residuals).mean()
        if error < best_error:
            best_error = error
            best_formula = name
            best_predictions = predictions
    
    print(f"\n\nBest formula: {best_formula} with MAE: ${best_error:.2f}")
    
    # Try machine learning approach
    print("\n\n=== MACHINE LEARNING APPROACH ===")
    
    # Prepare features
    feature_cols = ['days', 'miles', 'receipts', 'miles_per_day', 'receipts_per_day', 
                    'log_receipts', 'sqrt_receipts']
    X = df[feature_cols]
    y = df['reimbursement']
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X, y)
    rf_predictions = rf.predict(X)
    rf_error = mean_absolute_error(y, rf_predictions)
    print(f"Random Forest MAE: ${rf_error:.2f}")
    
    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    gb.fit(X, y)
    gb_predictions = gb.predict(X)
    gb_error = mean_absolute_error(y, gb_predictions)
    print(f"Gradient Boosting MAE: ${gb_error:.2f}")
    
    # Feature importance
    print("\nFeature Importance (Random Forest):")
    for feat, imp in zip(feature_cols, rf.feature_importances_):
        print(f"  {feat}: {imp:.3f}")
    
    # Analyze specific cases to understand the pattern
    print("\n\n=== SPECIFIC CASE ANALYSIS ===")
    
    # Look at 1-day trips with varying receipts
    print("\n1-day trips with different receipt levels:")
    one_day = df[df['days'] == 1].sort_values('receipts')
    for idx in [0, len(one_day)//4, len(one_day)//2, 3*len(one_day)//4, len(one_day)-1]:
        if idx < len(one_day):
            row = one_day.iloc[idx]
            print(f"  {row['miles']:.0f}mi, ${row['receipts']:.2f} → ${row['reimbursement']:.2f}")

if __name__ == "__main__":
    main()
