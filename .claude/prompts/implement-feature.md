# Implement Feature Engineering

## Context
Implementing feature engineering for credit card portfolio anomaly detection using spec-driven development.

## Requirements
1. Read feature specifications from config/modelconfig.yaml
2. Implement all derived features (utilization_ratio, transaction_velocity, spend_stability)
3. Add feature validation and quality checks
4. Include comprehensive logging
5. Write unit tests with edge cases

## Inputs
- config/modelconfig.yaml (feature specs)
- DataFrame with columns: customer_id, balance, credit_limit, transaction_count, monthly_spend

## Outputs
- src/features/engineering.py with FeatureEngineer class
- tests/unit/test_feature_engineering.py with >80% coverage

## Standards
- Follow AGENTS.md guidelines
- Use numpy/pandas efficiently
- Document all formulas in docstrings
- Handle division by zero, null values
- Type hints on all functions

## Acceptance Criteria
- [ ] All features from config implemented
- [ ] Edge cases handled (zeros, nulls, infinities)
- [ ] Unit tests pass with >80% coverage
- [ ] mypy passes with no errors
- [ ] Code formatted with black
