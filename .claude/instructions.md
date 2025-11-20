# Repository Instructions for AI Agents

## Project: Card Portfolio Anomaly Detection

### Core Principles
1. **Spec-Driven Development**: All behavior driven by YAML configs (modelconfig.yaml, dataconfig.yaml)
2. **Configuration over Constants**: No magic numbers in code
3. **Reproducibility**: Same inputs + same config = same outputs
4. **Testability**: Every module must have >80% test coverage

### Code Standards (AGENTS.md)
- Python 3.9+ with type hints everywhere
- Format with Black (line length 88)
- Lint with Ruff + Flake8
- Type check with mypy --strict
- Test with pytest
- Follow PEP 8 naming conventions

### Architecture
```
Data Layer → Feature Engineering → Model Training → Reporting
     ↓              ↓                    ↓              ↓
  Validation   Transformation       Prediction    CSV/JSON
```

### Module Responsibilities
- **src/data/**: Load and validate input data from config sources
- **src/features/**: Engineer features per modelconfig.yaml specs
- **src/models/**: Train/predict with Isolation Forest
- **src/reporting/**: Generate output reports
- **src/utils/**: Config loading, logging, shared utilities

### Testing Requirements
- Unit tests in tests/unit/ mirroring src/ structure
- Integration tests in tests/integration/ for pipelines
- Spec tests in tests/spectests/ validating requirements
- Use pytest fixtures from docs/test-templates.md

### Before Submitting Code
1. Run: `black src/ tests/`
2. Run: `ruff check src/ tests/`
3. Run: `mypy src/`
4. Run: `pytest --cov=src --cov-report=term-missing`
5. Ensure coverage >80%
