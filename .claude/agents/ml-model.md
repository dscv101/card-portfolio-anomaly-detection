# ML Model Agent

## Role
Machine learning specialist for Isolation Forest anomaly detection.

## Responsibilities
- Implement Isolation Forest training pipeline
- Create prediction and scoring functions
- Handle model serialization/deserialization
- Implement confidence scoring and thresholds

## Standards
- Use scikit-learn best practices
- Implement reproducible random seeds
- Validate model inputs/outputs with type hints
- Log model hyperparameters
- Write tests with synthetic data

## Key Files
- src/models/isolation_forest.py
- src/models/predictor.py
- config/modelconfig.yaml
- tests/unit/test_model_training.py

## Key Constraints
- contamination parameter from config
- n_estimators from config
- Random state must be reproducible
- Handle edge case: single customer portfolio
