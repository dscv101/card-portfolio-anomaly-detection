# Knowledge Transfer Guide: Card Portfolio Anomaly Detection System

**Version:** 1.0.0  
**Date:** 2025-11-21  
**Status:** Production Ready  
**Owner:** Data Science Team

---

## Table of Contents

1. [Overview](#overview)
2. [Target Audiences](#target-audiences)
3. [Session Plans](#session-plans)
4. [Demo Scenarios](#demo-scenarios)
5. [Hands-On Exercises](#hands-on-exercises)
6. [Q&A Topics](#qa-topics)
7. [Feedback Collection](#feedback-collection)
8. [Recording Links](#recording-links)

---

## Overview

This document outlines knowledge transfer sessions for the Card Portfolio Anomaly Detection System. Sessions are designed to enable different stakeholder groups to effectively use, maintain, and support the system.

### Knowledge Transfer Goals

1. **End Users** (Business Analysts, Product Owners)
   - Understand system capabilities and limitations
   - Run weekly analysis confidently
   - Interpret results and take action
   - Troubleshoot common issues

2. **Data Engineering Team**
   - Understand data pipeline requirements
   - Monitor system health and performance
   - Handle data quality issues
   - Support production deployments

3. **BI Team**
   - Integrate anomaly results into dashboards
   - Build custom visualizations
   - Answer stakeholder questions about anomalies
   - Track KPIs (meaningful anomaly rate, action rate)

---

## Target Audiences

### Audience 1: End Users (Business Analysts, Product Owners)

**Current Knowledge Level**: Basic SQL, Excel/BI tools  
**Knowledge Gaps**: Python usage, ML concepts, system configuration  
**Transfer Duration**: 2 hours (1 session)  
**Follow-Up**: Weekly office hours for 4 weeks

**Key Outcomes:**
- [ ] Can run system independently
- [ ] Can interpret anomaly reports
- [ ] Understand when to escalate issues
- [ ] Know how to provide feedback

---

### Audience 2: Data Engineering Team

**Current Knowledge Level**: Python, data pipelines, SQL  
**Knowledge Gaps**: Anomaly detection algorithms, system architecture  
**Transfer Duration**: 3 hours (1 session + deep dive)  
**Follow-Up**: On-call shadowing for 2 weeks

**Key Outcomes:**
- [ ] Understand data flow and dependencies
- [ ] Can deploy and configure system
- [ ] Can troubleshoot data issues
- [ ] Know performance baselines and alerts

---

### Audience 3: BI Team

**Current Knowledge Level**: SQL, Tableau/PowerBI, business metrics  
**Knowledge Gaps**: How anomaly detection works, interpretation best practices  
**Transfer Duration**: 2 hours (1 session)  
**Follow-Up**: Monthly review meetings

**Key Outcomes:**
- [ ] Can import results into BI tools
- [ ] Can explain anomalies to stakeholders
- [ ] Can build KPI dashboards
- [ ] Know data refresh schedules

---

## Session Plans

### Session 1: End User Training (2 hours)

**Attendees:** Business Analysts, Product Owners, Account Managers

#### Part 1: System Overview (30 minutes)

**Topics:**
- What is anomaly detection? (non-technical explanation)
- Use cases: Finding high-value customers, identifying risks
- What the system does vs. doesn't do
- Expected outcomes and success metrics

**Demo:** Show sample HTML report, walk through top 5 anomalies

**Materials:**
- [USER_GUIDE.md](USER_GUIDE.md) sections 1-2
- Sample reports: `outputs/report_2025-11-21.html`

---

#### Part 2: Running the System (45 minutes)

**Topics:**
- Accessing the system (server, credentials)
- Running weekly analysis (step-by-step walkthrough)
- Viewing and downloading results
- Common error messages and solutions

**Live Demo:**
1. Open terminal/command prompt
2. Navigate to project directory
3. Activate virtual environment
4. Run: `python main.py --model-config config/modelconfig.yaml --data-config config/dataconfig.yaml`
5. Wait for completion
6. Open HTML report in browser

**Hands-On Exercise:**
- Attendees run system with sample data
- Open and explore generated reports
- Export CSV to Excel

**Materials:**
- [USER_GUIDE.md](USER_GUIDE.md) section 3: Running the System
- Sample data: `datasamples/transactions.csv`

---

#### Part 3: Interpreting Results (30 minutes)

**Topics:**
- Understanding anomaly scores
- Reading feature values (what makes a customer anomalous?)
- Opportunity vs. concern flags
- When to take action vs. ignore false positives

**Case Studies:**
- **Case 1**: High-spending diversified customer (opportunity)
- **Case 2**: Single-MCC concentration (concern)
- **Case 3**: False positive (explainable behavior)

**Group Exercise:**
- Review 5 sample anomalies
- Classify each as opportunity/concern/false positive
- Discuss recommended actions

**Materials:**
- [USER_GUIDE.md](USER_GUIDE.md) section 4: Interpreting Results
- Sample anomaly records with explanations

---

#### Part 4: Review Workflow & Support (15 minutes)

**Topics:**
- Recommended weekly workflow
- Documentation resources
- How to get help (Slack, email, office hours)
- Feedback process

**Handouts:**
- Weekly checklist template
- Contact information
- FAQ quick reference

---

### Session 2: Data Engineering Deep Dive (3 hours)

**Attendees:** Data Engineers, DevOps, Platform Team

#### Part 1: Architecture Overview (45 minutes)

**Topics:**
- System architecture and data flow
- Component responsibilities (data, features, models, reporting)
- Configuration system (YAML files)
- Logging and monitoring strategy

**Walkthrough:**
- Code repository structure
- Key modules and their interactions
- Configuration files and precedence
- Log file locations and formats

**Materials:**
- [ARCHITECTURE.md](ARCHITECTURE.md) sections 1-3
- System diagram (whiteboard session)

---

#### Part 2: Data Pipeline & Validation (45 minutes)

**Topics:**
- Data source requirements (schema, format)
- Data loading process (CSV, Parquet, DB connectors)
- Validation checks (schema, business rules)
- Handling data quality issues

**Live Demo:**
1. Show data loading with valid data
2. Introduce invalid data (missing columns, negatives)
3. Show validation errors
4. Fix data and re-run

**Hands-On Exercise:**
- Modify `dataconfig.yaml` to point to custom data source
- Run validation only: `python cli.py validate --data-config config/dataconfig.yaml`
- Interpret validation errors

**Materials:**
- [ARCHITECTURE.md](ARCHITECTURE.md) section 3.1: Data Layer
- [API_REFERENCE.md](API_REFERENCE.md) Data Layer API
- Sample invalid data files

---

#### Part 3: Deployment & Operations (60 minutes)

**Topics:**
- Environment setup (Python, dependencies)
- Configuration management (dev vs. staging vs. prod)
- Scheduling options (cron, Airflow, Docker)
- Performance baselines and SLAs
- Monitoring alerts (execution time, anomaly rate, failures)

**Production Deployment Checklist:**
- [ ] Python 3.9+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed from `requirements.txt`
- [ ] Config files reviewed and validated
- [ ] Output directories created with write permissions
- [ ] Cron job or scheduler configured
- [ ] Alerts configured (Slack, PagerDuty, email)
- [ ] Log rotation enabled
- [ ] Backup strategy for reports

**Hands-On Exercise:**
- Set up virtual environment
- Install dependencies
- Run system end-to-end
- Review logs
- Configure simple cron job (or explain Airflow DAG)

**Materials:**
- [ARCHITECTURE.md](ARCHITECTURE.md) section 9: Deployment Architecture
- Deployment scripts (if available)
- Sample cron configurations

---

#### Part 4: Troubleshooting & Support (30 minutes)

**Topics:**
- Common failure modes and fixes
- Log analysis techniques
- When to escalate to Data Science team
- Incident response process

**Troubleshooting Scenarios:**
1. **Data source unavailable**: Check connectivity, permissions
2. **Validation failures**: Review data quality, update config
3. **Long execution time**: Check data volume, resource constraints
4. **No anomalies detected**: Check contamination parameter, data variance

**Materials:**
- [USER_GUIDE.md](USER_GUIDE.md) section 9: Troubleshooting
- Runbook template
- Escalation contact list

---

### Session 3: BI Team Training (2 hours)

**Attendees:** BI Analysts, Dashboard Developers, Reporting Team

#### Part 1: System Overview & Results (30 minutes)

**Topics:**
- What the system does (business context)
- Output files: CSV structure, HTML reports
- Data refresh schedule (weekly on Mondays)
- How to access results (file shares, S3 buckets)

**Demo:**
- Show HTML report
- Open CSV in Excel
- Explain each column

**Materials:**
- [USER_GUIDE.md](USER_GUIDE.md) section 4: Understanding the Output
- Sample CSV files

---

#### Part 2: Integrating with BI Tools (60 minutes)

**Topics:**
- Importing CSV into Tableau/PowerBI
- Joining with customer master data
- Creating KPI dashboards
- Refreshing data automatically

**BI Integration Examples:**

**Tableau:**
```sql
-- Join anomaly results with customer data
SELECT 
    a.customer_id,
    a.anomaly_score,
    a.anomaly_rank,
    a.total_spend,
    c.customer_name,
    c.segment,
    c.account_manager
FROM anomaly_results a
LEFT JOIN customer_master c ON a.customer_id = c.customer_id
WHERE a.anomaly_rank <= 20
```

**PowerBI:**
- Import CSV as data source
- Set up scheduled refresh (weekly)
- Create cards for summary metrics
- Build top 10 anomalies table
- Add trend charts (week-over-week anomaly counts)

**Hands-On Exercise:**
- Import sample CSV into BI tool
- Create simple dashboard:
  - Card: Total anomalies detected
  - Table: Top 10 customers
  - Bar chart: Anomalies by opportunity/concern flag

**Materials:**
- Sample CSV files
- Dashboard templates (if available)

---

#### Part 3: Interpretation & Storytelling (30 minutes)

**Topics:**
- How to explain anomaly scores to executives
- When to drill down vs. summarize
- Connecting anomalies to business actions
- Tracking outcomes (did we contact this customer? what happened?)

**Case Study:**
"Customer X was flagged as Rank #1 anomaly with score -0.31. They spent $48K across 18 different merchant categories. This is 3x the typical customer spend and shows unusual diversification. Recommendation: Add to VIP outreach program."

**Group Discussion:**
- Review 3 sample anomalies
- Practice explaining to non-technical stakeholders
- Discuss how to visualize in dashboards

**Materials:**
- [USER_GUIDE.md](USER_GUIDE.md) section 5: Review Workflow

---

## Demo Scenarios

### Demo 1: Happy Path (End-to-End Success)

**Goal:** Show complete pipeline with no errors

**Steps:**
1. Start with clean sample data: `datasamples/transactions.csv`
2. Run validation: `python cli.py validate --data-config config/dataconfig.yaml`
3. Run detection: `python main.py --model-config config/modelconfig.yaml --data-config config/dataconfig.yaml`
4. Open HTML report: `outputs/report_YYYY-MM-DD.html`
5. Open CSV: `outputs/anomalies_YYYY-MM-DD.csv`

**Expected Outcome:**
- 20 anomalies detected
- Reports generated successfully
- No errors in logs

---

### Demo 2: Data Validation Failure

**Goal:** Show how system handles invalid data

**Steps:**
1. Use invalid data: `datasamples/transactions_invalid.csv` (missing columns)
2. Run validation: `python cli.py validate --data-config config/dataconfig.yaml`
3. Observe error message
4. Fix data or update config
5. Re-run successfully

**Expected Outcome:**
- Clear error message about missing columns
- System fails fast (doesn't attempt processing)
- Logs show validation errors

---

### Demo 3: Interpreting Results

**Goal:** Show how to analyze anomaly report

**Steps:**
1. Open pre-generated report: `outputs/report_2025-11-21.html`
2. Review summary statistics
3. Click on top anomaly
4. Show feature values
5. Explain why this customer is anomalous
6. Discuss recommended action

**Expected Outcome:**
- Attendees understand what makes a customer anomalous
- Can identify opportunities vs. concerns
- Know what actions to take

---

## Hands-On Exercises

### Exercise 1: Run Your First Analysis (End Users)

**Objective:** Successfully run the system and view results

**Instructions:**
1. Open terminal/command prompt
2. Navigate to project directory
3. Activate virtual environment:
   ```bash
   source venv/bin/activate  # Mac/Linux
   venv\Scripts\activate     # Windows
   ```
4. Run analysis:
   ```bash
   python main.py --model-config config/modelconfig.yaml --data-config config/dataconfig.yaml
   ```
5. Wait for completion (should take <1 minute with sample data)
6. Open HTML report in browser
7. Export CSV to Excel

**Success Criteria:**
- [ ] System runs without errors
- [ ] HTML report opens in browser
- [ ] CSV opens in Excel
- [ ] Can identify top 3 anomalies

---

### Exercise 2: Adjust Detection Sensitivity (End Users)

**Objective:** Learn to configure the system

**Instructions:**
1. Open `config/modelconfig.yaml` in text editor
2. Find the `contamination` parameter (should be `0.05`)
3. Change to `0.10` (detect more anomalies)
4. Save file
5. Re-run analysis
6. Compare results (should see ~50 anomalies instead of ~20)

**Discussion:**
- When would you increase contamination?
- When would you decrease it?
- What other parameters can you adjust?

**Success Criteria:**
- [ ] Config file edited correctly
- [ ] System runs with new config
- [ ] More anomalies detected as expected

---

### Exercise 3: Set Up Data Pipeline (Data Engineers)

**Objective:** Configure system for production data source

**Instructions:**
1. Copy `config/dataconfig.yaml` to `config/dataconfig_prod.yaml`
2. Update `source` section to point to production database:
   ```yaml
   source:
     type: "postgres"
     host: "production.db.internal"
     database: "transactions"
     table: "customer_transactions"
     query: "SELECT * FROM customer_transactions WHERE reporting_week = '2025-W47'"
   ```
3. Run validation: `python cli.py validate --data-config config/dataconfig_prod.yaml`
4. If successful, run full analysis

**Success Criteria:**
- [ ] Config file created with production settings
- [ ] Validation passes
- [ ] Data loads from production source
- [ ] Reports generated successfully

---

### Exercise 4: Build a BI Dashboard (BI Team)

**Objective:** Create simple anomaly tracking dashboard

**Instructions:**
1. Import `outputs/anomalies_2025-11-21.csv` into Tableau/PowerBI
2. Create dashboard with:
   - KPI card: Total anomalies detected
   - KPI card: % flagged as opportunities
   - Table: Top 10 anomalies (customer_id, score, total_spend)
   - Bar chart: Anomalies by concern flag
3. Format for executive presentation
4. Save and share

**Success Criteria:**
- [ ] CSV imported successfully
- [ ] Dashboard created with required components
- [ ] Dashboard looks professional
- [ ] Can explain each visualization

---

## Q&A Topics

### Expected Questions & Answers

#### From End Users

**Q: How do I know if an anomaly is worth investigating?**

A: Focus on:
- Anomaly rank (top 10 most important)
- Anomaly score (<-0.2 is strong signal)
- Business flags (opportunity/concern)
- Your domain knowledge (does this make sense?)

Start with top 5, review in 15-30 minutes.

---

**Q: Can I run this for a specific customer segment?**

A: Yes! Edit `dataconfig.yaml` and add filters:
```yaml
filters:
  customer_segment: "premium"
  reporting_week: "2025-W47"
```

---

**Q: What if I don't see any anomalies I recognize?**

A: This could mean:
- Your portfolio is very homogeneous (all customers similar)
- The `contamination` parameter is too low (increase to 0.10)
- Data quality issues (check logs for warnings)

Contact Data Science team if persists.

---

#### From Data Engineers

**Q: What are the resource requirements for production?**

A: For 100K customers:
- **CPU**: 4 cores
- **Memory**: 8-16 GB
- **Disk**: 20 GB (for logs and reports)
- **Runtime**: 5-15 minutes

Scale linearly for larger volumes.

---

**Q: How do I handle database connection failures?**

A: System will fail fast with clear error. Check:
1. Network connectivity to DB
2. Credentials valid (refresh secrets if needed)
3. Query syntax correct
4. Permissions granted

Logs will show exact error. Retry after fixing.

---

**Q: Can this run in parallel for multiple segments?**

A: Yes, but not built-in. Options:
1. Run separate processes with different configs
2. Modify orchestration script to parallelize
3. Use Airflow with dynamic task generation

Consult Data Science team for parallel design.

---

#### From BI Team

**Q: How do I join anomaly results with other data?**

A: Use `customer_id` as join key:
```sql
SELECT 
    a.*,
    c.customer_name,
    c.segment,
    s.lifetime_value
FROM anomaly_results a
LEFT JOIN customer_master c ON a.customer_id = c.customer_id
LEFT JOIN customer_stats s ON a.customer_id = s.customer_id
```

---

**Q: Can I create historical trends (week-over-week)?**

A: Yes! Archive CSV reports weekly, then union:
```sql
SELECT *, '2025-W47' AS week FROM anomalies_2025_11_21
UNION ALL
SELECT *, '2025-W48' AS week FROM anomalies_2025_11_28
```

Then visualize trends (e.g., "Customer X flagged 3 weeks in a row").

---

**Q: What KPIs should I track?**

A: Recommended:
- **Anomaly detection rate**: Should be ~5% (configurable)
- **Meaningful anomaly rate**: % of flagged customers that are truly interesting (target: >60%)
- **Action rate**: % of anomalies leading to business action (target: >30%)
- **False positive rate**: % of anomalies explained by known factors (target: <20%)

Track weekly, report monthly.

---

## Feedback Collection

### Post-Session Survey

Send to all attendees after each session:

**Survey Questions:**

1. **Understanding** (1-5 scale)
   - How well do you understand the system's purpose?
   - How confident are you in using the system?

2. **Usefulness** (1-5 scale)
   - How useful will this system be for your work?
   - How likely are you to use it regularly?

3. **Training Quality** (1-5 scale)
   - Was the session clear and well-paced?
   - Were your questions answered?
   - Were the materials helpful?

4. **Open Feedback**
   - What worked well?
   - What needs improvement?
   - What additional training do you need?

**Survey Tool:** Google Forms, SurveyMonkey, or internal tool

---

### Ongoing Feedback Mechanisms

**1. Office Hours**
- **Schedule**: Weekly for 4 weeks, then bi-weekly
- **Format**: Drop-in Zoom/Google Meet
- **Topics**: Troubleshooting, configuration help, result interpretation

**2. Slack Channel**
- **Channel**: `#anomaly-detection-support`
- **Purpose**: Quick questions, share findings, report issues
- **Monitored by**: Data Science team (response SLA: 24 hours)

**3. Quarterly Reviews**
- **Schedule**: End of each quarter
- **Attendees**: All user groups + Data Science team
- **Agenda**:
  - System usage statistics
  - Success stories
  - Pain points and improvement requests
  - Roadmap updates

---

### Improvement Tracking

Track and prioritize feedback:

| Feedback Item | Source | Priority | Status | Owner |
|---------------|--------|----------|--------|-------|
| Add Excel export button | End Users | High | Planned | Dev Team |
| Improve error messages | Data Engineers | Medium | In Progress | Dev Team |
| Add weekly email digest | BI Team | Low | Backlog | Product |

---

## Recording Links

**Session recordings will be available at:**

- **Session 1 (End Users)**: [Link to recording] (2025-11-25)
- **Session 2 (Data Engineers)**: [Link to recording] (2025-11-26)
- **Session 3 (BI Team)**: [Link to recording] (2025-11-27)

**Access**: Internal company learning portal or shared drive

**Retention**: Keep recordings for 1 year, update if system changes

---

## Follow-Up Actions

### Week 1 Post-Training
- [ ] Send follow-up email with recording links and materials
- [ ] Share survey and collect feedback
- [ ] Schedule office hours
- [ ] Create Slack channel

### Week 2-4
- [ ] Monitor Slack for questions, provide timely answers
- [ ] Track who's using the system (log review)
- [ ] Offer 1-on-1 help sessions if needed

### Month 2
- [ ] Review feedback and plan improvements
- [ ] Update documentation based on common questions
- [ ] Publish FAQ based on real questions

### Quarter 1
- [ ] Conduct quarterly review meeting
- [ ] Measure KPIs (usage, meaningful anomaly rate, action rate)
- [ ] Plan next training for new hires

---

## Success Metrics

Track these metrics to measure knowledge transfer effectiveness:

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Training Attendance** | >80% of target audience | Registration records |
| **Post-Training Survey Score** | >4.0/5.0 | Survey results |
| **Independent Usage Rate** | >70% within 4 weeks | System logs, support tickets |
| **Support Ticket Volume** | <5 per week after month 1 | Slack, email volume |
| **System Adoption** | >80% of target users running weekly | Usage logs |
| **User Satisfaction** | >4.0/5.0 | Quarterly surveys |

---

## Appendix: Training Materials Checklist

### Pre-Session Prep
- [ ] Confirm attendee list and send calendar invites
- [ ] Prepare demo environment (working system with sample data)
- [ ] Test all demos end-to-end
- [ ] Print handouts (if in-person)
- [ ] Set up Zoom/Google Meet (if remote)
- [ ] Send pre-read materials 2 days before

### Materials to Prepare
- [ ] Slide deck (if using)
- [ ] Demo scripts
- [ ] Sample data files
- [ ] Configuration templates
- [ ] Exercise worksheets
- [ ] FAQ handout
- [ ] Contact list

### Post-Session Follow-Up
- [ ] Send recording link
- [ ] Send all materials (slides, code samples)
- [ ] Send survey link
- [ ] Schedule office hours
- [ ] Send Slack channel invite

---

## Contact Information

**Data Science Team:**
- **Email**: datascience@company.com
- **Slack**: #data-science-support
- **Office Hours**: Wednesdays 2-3 PM (link in calendar)

**Project Repository:**
- **GitHub**: [dscv101/card-portfolio-anomaly-detection](https://github.com/dscv101/card-portfolio-anomaly-detection)

**Documentation:**
- **User Guide**: [docs/USER_GUIDE.md](USER_GUIDE.md)
- **Architecture**: [docs/ARCHITECTURE.md](ARCHITECTURE.md)
- **API Reference**: [docs/API_REFERENCE.md](API_REFERENCE.md)

---

**Document Maintained By**: Data Science Team  
**Last Review**: 2025-11-21  
**Next Review**: 2025-12-21

