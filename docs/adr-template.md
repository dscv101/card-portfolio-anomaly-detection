# Architecture Decision Record (ADR) Template

## 1. Title

Short, descriptive title of the decision

## 2. Status

- Proposed
- Accepted
- Deprecated
- Superseded

## 3. Context

- What is the problem this decision addresses?
- What constraints or requirements are relevant?

## 4. Decision

- What is the architecture decision?
- What options did you consider?
- Why did you choose this option?

## 5. Consequences

- What are the implications (positive/negative)?
- What trade-offs are involved?

## 6. Related Decisions/Docs

- Links to relevant ADRs, requirements, specs, implementation notes

---

## Example: ADR-003 - Isolation Forest for Anomaly Detection

### 1. Title

Adopt Scikit-learn Isolation Forest as the Primary Model

### 2. Status

Accepted

### 3. Context

The anomaly detection system must identify positive outliers in cross-sectional weekly/monthly credit card transactional aggregates. Business stakeholders require explainable results, fast scoring, and robust unsupervised learning.

### 4. Decision

We will standardize on the scikit-learn `IsolationForest` algorithm for the model scoring module. Other models (e.g., LOF, One-Class SVM) were considered, but IsolationForest offers:
- Tunable contamination parameter
- Strong performance in similar banking data
- Excellent integration with pandas/numpy pipeline

### 5. Consequences

**Positive:**
- Simple interface for batch jobs
- Easily parameterized for future improvements
- Well-documented for handoff to other data scientists

**Negative:**
- Not optimal for high-dimensional or heavily imbalanced cases (may revisit if portfolio changes)

### 6. Related Decisions/Docs

- Requirements.md: Success Criteria 2, 3
- Design.md: ModelScorer module
- Task.md: Phase 3 Tasks