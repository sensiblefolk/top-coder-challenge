# AGENT.md - Top Coder Challenge

## You're a principal engineer at a top-tier tech company, tasked with reverse-engineering a 60-year-old travel reimbursement system using historical data and employee interviews.

## Project Overview
Reverse-engineering challenge to recreate a 60-year-old travel reimbursement system using historical data and employee interviews.

## Test Commands
- **Run single test**: `./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>`
- **Evaluate against all 1000 cases**: `./eval.sh`
- **Generate private results**: `./generate_results.sh`

## Dependencies
- `jq` - JSON parsing (install via: `brew install jq` on macOS)
- `bc` - Basic calculator for floating point arithmetic

## Implementation Requirements
- Must implement logic in `run.sh` (copy from `run.sh.template`)
- Output must be a single numeric value (float, 2 decimal places)
- Must run in under 5 seconds per test case
- No external dependencies (network, databases, etc.)

## Input/Output Format
- **Input**: 3 parameters - trip_duration_days (int), miles_traveled (int), total_receipts_amount (float)
- **Output**: Single reimbursement amount (float)
- **Example**: `./run.sh 5 250 150.75` → `487.25`

## Implementation Guidelines
- Analyze patterns in `public_cases.json` (1000 historical examples)
- Study business logic hints in `PRD.md` and `INTERVIEWS.md`
- Focus on exact replication including bugs/quirks in legacy system
- Success measured by exact matches (±$0.01) against historical data

## Implementation Language
- Must be written in python but run using instructions in `run.sh.template`
- use uv as package manager

## Key Resources
- **spec.md**: Contains comprehensive research on legacy travel reimbursement systems architecture and patterns
- **testing.md**: Kevin's analytical approach with detailed success criteria and progress tracking
- **INTERVIEWS.md**: Critical employee insights, especially Kevin's statistical analysis findings

## Critical Restrictions
- **NEVER EDIT**: `eval.sh` and `generate_results.sh` are sacred validation scripts
- These scripts are provided by the challenge and must remain untouched
- Only modify `run.sh` implementation and supporting Python files

## Kevin's Key Insights to Implement
- Efficiency sweet spot: 180-220 miles/day for maximum bonuses
- Six distinct calculation paths based on trip characteristics  
- Spending thresholds: <$75/day (short), <$120/day (medium), <$90/day (long)
- 5-day "sweet spot combo": 5 days + 180+ miles/day + <$100/day = guaranteed bonus
- 8+ day "vacation penalty": high spending on long trips = guaranteed reduction
- Temporal patterns: Tuesday submissions 8% better than Friday
- Intentional randomization: ±4-6% variance to prevent gaming
