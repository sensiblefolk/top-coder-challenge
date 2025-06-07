#!/usr/bin/env python3
"""
Analyze patterns in the data to identify coefficients
"""
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor

def load_data() -> pd.DataFrame:
    """Load the public cases data"""
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame for easier analysis
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
    
    # Add calculated fields
    df['miles_per_day'] = df['miles'] / df['days']
    df['receipts_per_day'] = df['receipts'] / df['days']
    df['efficiency_ratio'] = df['miles_per_day'] / 100  # Normalized efficiency
    
    return df

def analyze_trip_categories(df: pd.DataFrame) -> Dict:
    """Analyze different trip categories based on Kevin's insights"""
    categories = {}
    
    # Quick High-Mileage: 1-3 days, 200+ miles/day
    quick_high = df[(df['days'] <= 3) & (df['miles_per_day'] >= 200)]
    categories['quick_high_mileage'] = {
        'count': len(quick_high),
        'avg_reimbursement': quick_high['reimbursement'].mean(),
        'avg_per_day': (quick_high['reimbursement'] / quick_high['days']).mean()
    }
    
    # 5-day sweet spot
    five_day = df[df['days'] == 5]
    categories['five_day_trips'] = {
        'count': len(five_day),
        'avg_reimbursement': five_day['reimbursement'].mean(),
        'avg_per_day': (five_day['reimbursement'] / 5).mean()
    }
    
    # Efficiency sweet spot: 180-220 miles/day
    efficiency_sweet = df[(df['miles_per_day'] >= 180) & (df['miles_per_day'] <= 220)]
    categories['efficiency_sweet_spot'] = {
        'count': len(efficiency_sweet),
        'avg_reimbursement': efficiency_sweet['reimbursement'].mean(),
        'avg_per_day': (efficiency_sweet['reimbursement'] / efficiency_sweet['days']).mean()
    }
    
    # Vacation penalty: 8+ days with high spending
    vacation_penalty = df[(df['days'] >= 8) & (df['receipts_per_day'] > 150)]
    categories['vacation_penalty'] = {
        'count': len(vacation_penalty),
        'avg_reimbursement': vacation_penalty['reimbursement'].mean(),
        'avg_per_day': (vacation_penalty['reimbursement'] / vacation_penalty['days']).mean()
    }
    
    return categories

def analyze_mileage_tiers(df: pd.DataFrame) -> None:
    """Analyze mileage reimbursement patterns"""
    print("\n=== MILEAGE TIER ANALYSIS ===")
    
    # Group by mileage ranges
    ranges = [(0, 100), (101, 500), (501, 1000), (1001, 2000)]
    
    for low, high in ranges:
        subset = df[(df['miles'] >= low) & (df['miles'] <= high)]
        if len(subset) > 0:
            # Calculate implied rate per mile (rough approximation)
            # Subtract estimated per diem to isolate mileage component
            estimated_per_diem = subset['days'] * 100
            mileage_component = subset['reimbursement'] - estimated_per_diem
            implied_rate = mileage_component / subset['miles']
            
            print(f"\nMiles {low}-{high}:")
            print(f"  Count: {len(subset)}")
            print(f"  Avg implied rate: ${implied_rate.mean():.3f}/mile")
            print(f"  Std dev: ${implied_rate.std():.3f}")

def analyze_receipt_patterns(df: pd.DataFrame) -> None:
    """Analyze receipt processing patterns"""
    print("\n=== RECEIPT PATTERN ANALYSIS ===")
    
    # Group by receipt ranges
    ranges = [(0, 50), (50, 800), (800, 1500), (1500, 3000)]
    
    for low, high in ranges:
        subset = df[(df['receipts'] >= low) & (df['receipts'] <= high)]
        if len(subset) > 0:
            # Calculate receipt utilization rate
            receipt_utilization = subset['reimbursement'] / subset['receipts']
            
            print(f"\nReceipts ${low}-${high}:")
            print(f"  Count: {len(subset)}")
            print(f"  Avg reimbursement: ${subset['reimbursement'].mean():.2f}")
            print(f"  Avg utilization: {receipt_utilization.mean():.2%}")

def find_five_day_bonus(df: pd.DataFrame) -> None:
    """Analyze 5-day trip bonus"""
    print("\n=== 5-DAY TRIP BONUS ANALYSIS ===")
    
    # Compare 5-day trips to adjacent durations
    for days in [4, 5, 6]:
        subset = df[df['days'] == days]
        avg_per_day = (subset['reimbursement'] / subset['days']).mean()
        print(f"{days}-day trips: ${avg_per_day:.2f}/day average")
    
    # Look for sweet spot combo: 5 days + 180+ miles/day + <$100/day receipts
    sweet_combo = df[
        (df['days'] == 5) & 
        (df['miles_per_day'] >= 180) & 
        (df['receipts_per_day'] < 100)
    ]
    
    if len(sweet_combo) > 0:
        print(f"\nSweet spot combo (5 days, 180+ mi/day, <$100/day receipts):")
        print(f"  Count: {len(sweet_combo)}")
        print(f"  Avg reimbursement: ${sweet_combo['reimbursement'].mean():.2f}")
        print(f"  Avg per day: ${(sweet_combo['reimbursement'] / 5).mean():.2f}")

def main():
    """Main analysis function"""
    print("Loading and analyzing public cases data...")
    
    df = load_data()
    print(f"\nTotal cases: {len(df)}")
    print(f"Days range: {df['days'].min()}-{df['days'].max()}")
    print(f"Miles range: {df['miles'].min():.0f}-{df['miles'].max():.0f}")
    print(f"Receipts range: ${df['receipts'].min():.2f}-${df['receipts'].max():.2f}")
    print(f"Reimbursement range: ${df['reimbursement'].min():.2f}-${df['reimbursement'].max():.2f}")
    
    # Analyze categories
    categories = analyze_trip_categories(df)
    print("\n=== TRIP CATEGORY ANALYSIS ===")
    for name, stats in categories.items():
        print(f"\n{name}:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: ${value:.2f}")
            else:
                print(f"  {key}: {value}")
    
    # Detailed analyses
    analyze_mileage_tiers(df)
    analyze_receipt_patterns(df)
    find_five_day_bonus(df)
    
    # Save DataFrame for further analysis
    df.to_csv('public_cases_analysis.csv', index=False)
    print("\n\nAnalysis saved to public_cases_analysis.csv")

if __name__ == "__main__":
    main()
