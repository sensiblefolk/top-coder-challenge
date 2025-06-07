#!/usr/bin/env python3
"""
Deep analysis of receipt patterns to understand the 65% feature importance
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

def load_data() -> pd.DataFrame:
    """Load the public cases data"""
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
    df['miles_per_day'] = df['miles'] / df['days']
    df['receipts_per_day'] = df['receipts'] / df['days']
    
    return df

def analyze_base_per_diem(df: pd.DataFrame) -> float:
    """Try to identify base per diem by looking at trips with minimal other factors"""
    print("\n=== BASE PER DIEM ANALYSIS ===")
    
    # Look at trips with very low miles and receipts to isolate per diem
    low_activity = df[(df['miles'] < 50) & (df['receipts'] < 50)]
    
    if len(low_activity) > 0:
        low_activity['implied_per_diem'] = low_activity['reimbursement'] / low_activity['days']
        print(f"Low activity trips (miles<50, receipts<50):")
        print(f"  Count: {len(low_activity)}")
        print(f"  Avg implied per diem: ${low_activity['implied_per_diem'].mean():.2f}")
        
        # Show some examples
        print("\n  Examples:")
        for _, row in low_activity.head(5).iterrows():
            print(f"    {row['days']}d, {row['miles']:.0f}mi, ${row['receipts']:.2f} → ${row['reimbursement']:.2f} (${row['implied_per_diem']:.2f}/day)")
    
    return 100.0  # Default assumption

def analyze_mileage_component(df: pd.DataFrame, base_per_diem: float) -> Dict:
    """Analyze mileage reimbursement patterns more precisely"""
    print("\n=== MILEAGE COMPONENT ANALYSIS ===")
    
    # Focus on trips with low receipts to isolate mileage effect
    low_receipt = df[df['receipts'] < 100]
    
    if len(low_receipt) > 0:
        # Estimate mileage component by subtracting estimated per diem
        low_receipt['estimated_per_diem'] = low_receipt['days'] * base_per_diem
        low_receipt['mileage_component'] = low_receipt['reimbursement'] - low_receipt['estimated_per_diem']
        low_receipt['implied_rate'] = low_receipt['mileage_component'] / low_receipt['miles']
        
        # Group by mileage tiers
        print("\nImplied mileage rates (low receipt trips):")
        for tier in [(0, 100), (100, 500), (500, 1000), (1000, 2000)]:
            tier_data = low_receipt[(low_receipt['miles'] >= tier[0]) & (low_receipt['miles'] < tier[1])]
            if len(tier_data) > 5:
                valid_rates = tier_data['implied_rate'][tier_data['implied_rate'] > 0]
                if len(valid_rates) > 0:
                    print(f"  {tier[0]}-{tier[1]} miles: ${valid_rates.mean():.3f}/mile (n={len(valid_rates)})")
    
    return {}

def analyze_receipt_processing(df: pd.DataFrame, base_per_diem: float) -> None:
    """Analyze how receipts affect reimbursement"""
    print("\n=== RECEIPT PROCESSING ANALYSIS ===")
    
    # Group by trip length to see different patterns
    for days in [1, 3, 5, 8]:
        subset = df[df['days'] == days]
        if len(subset) > 10:
            print(f"\n{days}-day trips:")
            
            # Sort by receipts and show pattern
            sorted_subset = subset.sort_values('receipts')
            
            # Show low, medium, high receipt examples
            examples = [
                sorted_subset.iloc[len(sorted_subset)//10],  # 10th percentile
                sorted_subset.iloc[len(sorted_subset)//2],   # 50th percentile
                sorted_subset.iloc[9*len(sorted_subset)//10] # 90th percentile
            ]
            
            for ex in examples:
                base = days * base_per_diem
                miles_comp = ex['miles'] * 0.5  # Rough estimate
                receipt_effect = ex['reimbursement'] - base - miles_comp
                print(f"  ${ex['receipts']:.2f} receipts → ${ex['reimbursement']:.2f} reimb (receipt effect: ${receipt_effect:.2f})")

def find_trip_type_patterns(df: pd.DataFrame) -> None:
    """Find patterns for different trip types"""
    print("\n=== TRIP TYPE PATTERN ANALYSIS ===")
    
    # Define trip types based on Kevin's insights
    trip_types = {
        'Quick High-Mileage': (df['days'] <= 3) & (df['miles_per_day'] >= 200),
        '5-Day Sweet Spot': (df['days'] == 5) & (df['miles_per_day'] >= 180) & (df['miles_per_day'] <= 220),
        'Efficiency Bonus': (df['miles_per_day'] >= 180) & (df['miles_per_day'] <= 220),
        'Vacation Penalty': (df['days'] >= 8) & (df['receipts_per_day'] > 150),
        'Standard Short': (df['days'] <= 3) & (df['miles_per_day'] < 200),
        'Standard Long': (df['days'] >= 8) & (df['receipts_per_day'] <= 150)
    }
    
    for name, condition in trip_types.items():
        subset = df[condition]
        if len(subset) > 5:
            avg_per_day = (subset['reimbursement'] / subset['days']).mean()
            print(f"\n{name}:")
            print(f"  Count: {len(subset)}")
            print(f"  Avg reimbursement: ${subset['reimbursement'].mean():.2f}")
            print(f"  Avg per day: ${avg_per_day:.2f}")
            print(f"  Avg miles/day: {subset['miles_per_day'].mean():.1f}")
            print(f"  Avg receipts/day: ${subset['receipts_per_day'].mean():.2f}")

def analyze_five_day_mystery(df: pd.DataFrame) -> None:
    """Deep dive into why 5-day trips seem to have lower per-day rates"""
    print("\n=== 5-DAY TRIP MYSTERY ANALYSIS ===")
    
    # Compare different day lengths
    print("\nAverage metrics by trip length:")
    for days in range(1, 9):
        subset = df[df['days'] == days]
        if len(subset) > 5:
            print(f"\n{days}-day trips (n={len(subset)}):")
            print(f"  Avg reimbursement: ${subset['reimbursement'].mean():.2f}")
            print(f"  Avg per day: ${(subset['reimbursement'] / days).mean():.2f}")
            print(f"  Avg miles: {subset['miles'].mean():.0f}")
            print(f"  Avg receipts: ${subset['receipts'].mean():.2f}")
    
    # Look specifically at 5-day trips with the "sweet spot" conditions
    sweet_5 = df[
        (df['days'] == 5) & 
        (df['miles_per_day'] >= 180) & 
        (df['miles_per_day'] <= 220) &
        (df['receipts_per_day'] < 100)
    ]
    
    regular_5 = df[
        (df['days'] == 5) & 
        ~((df['miles_per_day'] >= 180) & 
          (df['miles_per_day'] <= 220) &
          (df['receipts_per_day'] < 100))
    ]
    
    print(f"\n5-day sweet spot trips (180-220 mi/day, <$100/day receipts): n={len(sweet_5)}")
    if len(sweet_5) > 0:
        print(f"  Avg reimbursement: ${sweet_5['reimbursement'].mean():.2f}")
        print(f"  Avg per day: ${(sweet_5['reimbursement'] / 5).mean():.2f}")
    
    print(f"\nOther 5-day trips: n={len(regular_5)}")
    if len(regular_5) > 0:
        print(f"  Avg reimbursement: ${regular_5['reimbursement'].mean():.2f}")
        print(f"  Avg per day: ${(regular_5['reimbursement'] / 5).mean():.2f}")

def main():
    """Main analysis function"""
    df = load_data()
    
    base_per_diem = analyze_base_per_diem(df)
    analyze_mileage_component(df, base_per_diem)
    analyze_receipt_processing(df, base_per_diem)
    find_trip_type_patterns(df)
    analyze_five_day_mystery(df)
    
    # Save some specific examples for testing
    print("\n\n=== SAVING TEST CASES ===")
    test_cases = []
    
    # Get examples of different trip types
    for days in [1, 3, 5, 8]:
        examples = df[df['days'] == days].sample(n=min(5, len(df[df['days'] == days])))
        for _, row in examples.iterrows():
            test_cases.append({
                'days': int(row['days']),
                'miles': float(row['miles']),
                'receipts': float(row['receipts']),
                'expected': float(row['reimbursement'])
            })
    
    with open('test_cases.json', 'w') as f:
        json.dump(test_cases, f, indent=2)
    
    print(f"Saved {len(test_cases)} test cases to test_cases.json")

if __name__ == "__main__":
    main()
