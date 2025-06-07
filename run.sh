#!/bin/bash

# Black Box Challenge - Your Implementation
# This script should take three parameters and output the reimbursement amount
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>

# Python implementation - using conservative approach for better generalization
python3 calculate_reimbursement_conservative.py "$1" "$2" "$3" 