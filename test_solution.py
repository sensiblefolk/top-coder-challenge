#!/usr/bin/env python3
"""
Quick test script to verify the KNN solution is working correctly
"""

import subprocess
import json

def test_cases():
    """Test a few known cases to verify accuracy"""
    # These are actual cases from public_cases.json
    test_data = [
        # (days, miles, receipts, expected)
        (3, 93, 1.42, 364.51),
        (1, 55, 3.6, 126.06),
        (1, 47, 17.97, 128.91),
        (2, 13, 4.67, 203.52),
        (3, 88, 5.78, 380.37),
        (7, 1006, 1181.33, 2279.82),
        (1, 451, 555.49, 162.18)
    ]
    
    print("Testing KNN solution on sample cases...")
    print("=" * 50)
    
    all_passed = True
    
    for days, miles, receipts, expected in test_data:
        # Run the implementation
        cmd = ['./run.sh', str(days), str(miles), str(receipts)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            actual = float(result.stdout.strip())
            error = abs(actual - expected)
            passed = error < 0.01
            
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} | {days} days, {miles} miles, ${receipts} receipts")
            print(f"      Expected: ${expected:.2f}, Got: ${actual:.2f}, Error: ${error:.2f}")
            
            if not passed:
                all_passed = False
        else:
            print(f"❌ FAIL | Script error for {days} days, {miles} miles, ${receipts} receipts")
            print(f"      Error: {result.stderr}")
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("🎉 All tests passed! The solution is working correctly.")
        print("\nNext steps:")
        print("1. Run ./eval.sh for full evaluation")
        print("2. Run ./generate_results.sh to create submission file")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return all_passed

def verify_dependencies():
    """Check that all required files exist"""
    required_files = [
        'calculate_reimbursement_fast.py',
        'run.sh',
        'public_cases.json'
    ]
    
    print("Checking dependencies...")
    all_found = True
    
    for file in required_files:
        try:
            with open(file, 'r'):
                print(f"✅ Found: {file}")
        except FileNotFoundError:
            print(f"❌ Missing: {file}")
            all_found = False
    
    print()
    return all_found

if __name__ == '__main__':
    print("🧾 Travel Reimbursement Solution Test")
    print()
    
    if verify_dependencies():
        test_cases()
    else:
        print("❌ Missing required files. Please ensure all files are present.") 