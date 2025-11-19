# AGENTS.md

## Python Best Practices for Agent Contributions

**All code in this repository is expected to be:**
- *Simple*
- *Consistent*
- *Readable by humans and AI reviewers*

The review and CI process will enforce the following:

---

### 📐 Code Style & Structure

- **PEP8**: Adhere to [PEP 8](https://peps.python.org/pep-0008/) for layout and naming.
- **Format with Black**: Auto-format code with [Black](https://black.readthedocs.io/en/stable/).
- **Lint with Ruff & Flake8**: Remove unused code, enforce import order, consistent whitespace, no trailing spaces.
- **Max line length: 88** (Black default).
- **Avoid complexity**: No clever hacks, deep nesting, or one-liners that obscure behavior.
- **Use type hints**: All public functions and class methods should have explicit type annotations.

---

### 🧩 Naming Conventions

| Element       | Convention            | Example                  |
|---------------|----------------------|--------------------------|
| Variable      | `snake_case`         | `total_count`            |
| Function/method| `snake_case`        | `get_data()`             |
| Class         | `PascalCase`         | `TransactionProcessor`   |
| Constant      | `UPPER_SNAKE_CASE`   | `MAX_RETRIES`            |
| Module        | `short_snake_case.py`| `core.py`, `data_utils.py`|

---

### 📋 Documentation

- **Docstrings**: Each module, class, and function must include a concise docstring describing what it does, its args and return type.
- **Simple language**: Docstrings and comments should be short, clear, and direct.
- **Module docstring**: At the top of each module, summarize its purpose.

---

### ✅ Type Checking

- Use type hints everywhere; run [mypy](https://mypy.readthedocs.io/en/stable/) (`strict` mode enabled).
- Fix all type errors before submitting code.

---

### 🛡️ Testing

- Use [pytest](https://docs.pytest.org/) for all tests.
- Organize tests in a `tests/` directory mirroring the package structure.
- All core logic must be covered by unit tests; strive for >80% coverage.
- Each test should check a single behavior or edge case.

---

### 🚦 Project Structure

```
repo_root/
├── src/                   # Production code
├── tests/                 # All tests (mirrors src/)
├── config/                # YAML/config files
├── docs/                  # Documentation
├── pyproject.toml         # Tooling config (Black, Ruff)
├── mypy.ini               # Mypy settings (if any)
├── .flake8                # Flake8 settings (if any)
├── .coderabbit.yaml       # AI review settings
└── README.md              # Project summary
```

---

### 🧰 Tooling Overview

- **Formatting**: Black (pyproject.toml)
- **Linting**: Ruff, Flake8 (pyproject/.flake8)
- **Static Types**: mypy (mypy.ini)
- **AI Code Reviews**: CodeRabbit (.coderabbit.yaml)

All code must pass these tools—*before* review.

---

### ⭐️ Principles

- Strive for clarity and correctness.
- Simpler is always better.
- Write code as if others (and future you) will read and extend it.
- Document decisions in ADRs if needed.

---

*AGENTS: Please follow these best practices to ensure your contributions are maintainable, understandable, and easy to review. If unsure, prefer clarity and simplicity—let the tools and reviewers flag any exceptions.*
