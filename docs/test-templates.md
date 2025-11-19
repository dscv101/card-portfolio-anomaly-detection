# Test Templates: Python Anomaly Detection System

**Project:** Card Portfolio Anomaly Detection  
**Version:** 1.0.0  
**Status:** Draft  
**Date:** 2025-11-19  
**Owner:** Data Science Team  
**Related Specs:** requirements.md, design.md, tasks.md

---

## Overview

This document provides reusable test templates for implementing the testing strategy defined in design.md. Templates are organized by test type and aligned with tasks.md.

---

## Template 1: Unit Test Template

**Purpose:** Test individual functions/methods in isolation  
**Location:** `tests/unit/`  
**File Naming:** `test_{module_name}.py`

### Basic Structure

```python
"""
Unit tests for {module_name}.

Tests cover:
- Happy path scenarios
- Edge cases and boundary conditions
- Error handling
- Input validation
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from src.{module_path} import ClassName


class TestClassName:
    """Test suite for ClassName."""
    
    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return {
            'param1': 'value1',
            'param2': 100,
            'logging': {'level': 'WARNING'}
        }
    
    @pytest.fixture
    def sample_data(self):
        """Provide sample input data."""
        return pd.DataFrame({
            'customer_id': ['C001', 'C002', 'C003'],
            'value': [100, 200, 300]
        })
    
    # Happy Path Tests
    def test_initialization_success(self, config):
        """Test successful initialization with valid config."""
        instance = ClassName(config)
        assert instance is not None
        assert instance.param1 == config['param1']
    
    # Edge Cases
    def test_empty_dataframe_handling(self, instance):
        """Test behavior with empty input."""
        empty_df = pd.DataFrame()
        result = instance.method_name(empty_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    # Error Handling
    def test_invalid_config_raises_error(self):
        """Test that invalid configuration raises appropriate error."""
        invalid_config = {'missing_required_key': True}
        with pytest.raises(ValueError, match="Required config key"):
            ClassName(invalid_config)
```

[Continue with all test templates...]

See attached test-templates.md file for complete templates.