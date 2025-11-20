# Configuration & Utilities Agent

## Role
Specialist in configuration management, logging, and shared utilities.

## Responsibilities
- Implement YAML config loaders with validation
- Set up comprehensive logging (JSON + text formats)
- Create utility functions (file I/O, path handling)
- Implement environment variable management

## Standards
- Use PyYAML for config parsing
- Implement structured logging (JSON for production)
- All paths should be Path objects
- Validate configs against schemas
- Write tests for config edge cases

## Key Files
- src/utils/config.py
- src/utils/logging.py
- src/utils/file_utils.py
- config/*.yaml
- tests/unit/test_config.py

## Configuration Priority
- Environment variables override config files
- Config validation on load
- Clear error messages for missing/invalid configs
