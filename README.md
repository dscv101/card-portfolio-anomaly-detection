# Card Portfolio Anomaly Detection

Python-based anomaly detection system for credit card customer portfolios using IsolationForest and spec-driven development.

## Overview

This system detects anomalous behavior patterns in credit card customer portfolios using unsupervised machine learning (Isolation Forest algorithm). It follows spec-driven development principles with extensive configuration support and comprehensive logging.

## Features

- **Unsupervised Anomaly Detection**: Uses scikit-learn's Isolation Forest algorithm
- **Configurable**: All parameters externalized to YAML configuration files
- **Observable**: Comprehensive logging with JSON and text formats
- **Testable**: Extensive test coverage with pytest
- **Maintainable**: Clean separation of concerns and modular architecture

## Project Structure

```
anomaly-detection/
├── src/                    # Source code
│   ├── data/              # Data loading and validation
│   ├── features/          # Feature engineering
│   ├── models/            # Model training and prediction
│   ├── reporting/         # Report generation
│   └── utils/             # Shared utilities (logging, config)
├── config/                # Configuration files
│   ├── modelconfig.yaml   # Model and feature configuration
│   └── dataconfig.yaml    # Data source and validation config
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   ├── e2e/              # End-to-end tests
│   └── spectests/        # Specification tests
├── datasamples/          # Sample data files
├── outputs/              # Generated reports and results
├── logs/                 # Application logs
├── docs/                 # Documentation
├── main.py               # Main entry point
├── cli.py                # Command-line interface
└── requirements.txt      # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/dscv101/card-portfolio-anomaly-detection.git
   cd card-portfolio-anomaly-detection
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   
   # On Linux/Mac:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   # Install production dependencies
   pip install -r requirements.txt
   
   # Install development dependencies (for testing and linting)
   pip install -r requirements-dev.txt
   ```

4. **Verify installation**:
   ```bash
   python -c "import pandas, sklearn, yaml; print('All dependencies installed successfully!')"
   ```

### Configuration

Configuration files are located in the `config/` directory:

- **`modelconfig.yaml`**: Controls model parameters, feature engineering, and reporting
- **`dataconfig.yaml`**: Defines data sources, schema, and validation rules

Copy `.env.example` to `.env` and customize if needed:
```bash
cp .env.example .env
```

### Usage

**Run the main pipeline**:
```bash
python main.py --model-config config/modelconfig.yaml --data-config config/dataconfig.yaml
```

**Use the CLI interface**:
```bash
# Validate data
python cli.py validate --data-config config/dataconfig.yaml

# Run anomaly detection
python cli.py detect --model-config config/modelconfig.yaml

# Generate reports
python cli.py report --input outputs/anomalies.csv --output-dir outputs/
```

### Testing

Run the test suite:
```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
```

### Code Quality

Format and lint code:
```bash
# Format with Black
black src/ tests/

# Lint with Ruff
ruff check src/ tests/

# Type check with mypy
mypy src/
```

## Development

### Phase 0: Project Setup ✅

- [x] Directory structure
- [x] Python environment and dependencies
- [x] Configuration files (modelconfig.yaml, dataconfig.yaml)
- [x] Logging infrastructure
- [x] Testing framework setup

### Upcoming Phases

- **Phase 1**: Data loading and validation
- **Phase 2**: Feature engineering
- **Phase 3**: Model training and prediction
- **Phase 4**: Reporting and visualization

See [docs/tasks.md](docs/tasks.md) for detailed task breakdown.

## Documentation

- [REQUIREMENTS.md](docs/REQUIREMENTS.md) - System requirements and specifications
- [design.md](docs/design.md) - Technical architecture and design decisions
- [tasks.md](docs/tasks.md) - Detailed task breakdown and estimates
- [AGENTS.md](AGENTS.md) - Python best practices for AI agent contributions

## Contributing

Please read [AGENTS.md](AGENTS.md) for coding standards and best practices. All code must:

- Follow PEP 8 style guide
- Be formatted with Black
- Pass Ruff and Flake8 linting
- Include type hints and pass mypy checks
- Have comprehensive test coverage

## License

MIT License - See LICENSE file for details

## Contact

Data Science Team - For questions and support

---

**Spec-Driven Development**: Configuration over code constants. Same inputs + same config = same outputs.

