# Design Architecture Document: Python Anomaly Detection System

**Project:** Card Portfolio Anomaly Detection  
**Version:** 1.0.0  
**Status:** Draft  
**Date:** 2025-11-19  
**Owner:** Data Science Team  
**Related Specs:** requirements.md, tasks.md

---

## Overview

### Design Philosophy

This design follows **spec-driven development** principles where the architecture serves the requirements specification. The system is designed for:

- **Reproducibility:** Same inputs + same config = same outputs
- **Observability:** Every decision point logged and traceable
- **Maintainability:** Clear separation of concerns, YAML-driven configuration
- **Testability:** Each module independently testable with well-defined contracts
- **Human-Centered:** Non-technical stakeholders can understand outputs and provide feedback

### Architecture Decision Records (ADR)

**ADR-001: Single-File Python Modules Over Classes**
- **Decision:** Use functional modules with clear input/output contracts rather than heavy object hierarchies
- **Rationale:** Easier for LLM-assisted development to reason about, simpler testing, less boilerplate

**ADR-002: YAML Configuration Over Code Constants**
- **Decision:** Externalize all tunable parameters to YAML files
- **Rationale:** Non-developers can tune models, spec remains executable with different configs

**ADR-003: Scikit-learn Isolation Forest**
- **Decision:** Use sklearn's IsolationForest as primary algorithm for v1.0
- **Rationale:** Well-documented, stable, meets performance requirements, enables quick iteration. PyOD reserved for v2.0.

**ADR-004: Batch Processing Over Real-Time**
- **Decision:** Weekly/monthly batch jobs, not streaming/real-time detection
- **Rationale:** Aligns with business cadence, simplifies architecture, meets 15-minute performance target

**ADR-005: Pandas Over Spark**
- **Decision:** Use pandas for data processing instead of distributed framework
- **Rationale:** Dataset size (~1M rows/week) fits in memory, simpler debugging, faster iteration

---

[Content continues with complete design document...]

See attached design.md file for complete architecture.