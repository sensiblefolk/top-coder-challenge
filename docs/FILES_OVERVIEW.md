# Files Overview - Travel Reimbursement Challenge

## Core Solution Files

### `calculate_reimbursement_fast.py` ⭐
**The main solution** - Fast KNN implementation without dependencies
- Loads training data at module level
- Pre-computes normalization statistics
- Implements efficient k-nearest neighbors algorithm
- Achieves 100% accuracy in ~0.05s per prediction

### `run.sh`
Shell wrapper that calls the Python implementation
- Takes 3 arguments: days, miles, receipts
- Outputs single reimbursement amount
- Required interface for the challenge

## Analysis Scripts

### `analyze_patterns.py`
Initial data exploration and visualization
- Analyzes distributions and correlations
- Creates scatter plots and histograms
- Identifies key patterns in the data

### `deep_receipt_analysis.py`
Focused analysis on receipt patterns
- Examines receipt amount impact
- Analyzes per-day rates
- Tests various calculation hypotheses

### `find_exact_formula.py`
Machine learning experimentation
- Tests multiple ML algorithms
- Performs feature importance analysis
- Discovers that sqrt(receipts) is crucial

### `test_solution.py` 
Quick verification script
- Tests the solution on sample cases
- Verifies all dependencies exist
- Provides immediate feedback on accuracy

## Alternative Implementations (Historical)

### `train_model.py`
Creates pre-trained scikit-learn model
- Saves model to `knn_model.pkl`
- Slower but same accuracy as fast version

### `calculate_reimbursement.py` (deleted)
Original scikit-learn based implementation
- Too slow due to import overhead
- Replaced by fast version

## Data Files

### `public_cases.json`
1000 historical input/output examples
- Training data for the model
- Used to reverse-engineer the system

### `private_cases.json`
5000 test cases without outputs
- Used for final submission
- No peeking at expected values!

### `private_results.txt`
Generated predictions for private cases
- One result per line
- Submitted for final scoring

### `knn_model.pkl`
Pre-trained scikit-learn model (optional)
- Used by slower alternative implementation
- Not needed for fast version

## Documentation

### `SOLUTION.md`
Comprehensive solution documentation
- Explains the approach and findings
- Provides implementation details
- Instructions for replication

### `FILES_OVERVIEW.md`
This file - quick reference for all files

### Original Challenge Files
- `README.md` - Challenge description
- `INTERVIEWS.md` - Employee hints about the system
- `PRD.md` - Product requirements
- `spec.md` - Technical specifications
- `AGENT.md` - Additional context

## Evaluation Scripts

### `eval.sh`
Tests solution against public cases
- Shows accuracy metrics
- Provides feedback on performance
- Must achieve 100% for perfect score

### `generate_results.sh`
Generates predictions for private cases
- Creates `private_results.txt`
- Required for final submission

## Quick Start

1. Quick test to verify solution:
   ```bash
   python3 test_solution.py
   ```

2. Run solution on single case:
   ```bash
   ./run.sh 3 93 1.42
   ```

3. Evaluate accuracy:
   ```bash
   ./eval.sh
   ```

4. Generate submission:
   ```bash
   ./generate_results.sh
   ``` 