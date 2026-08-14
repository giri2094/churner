# Exploratory Data Analysis Findings

## Investigation 1 — Retention vs. Churn Baseline

**Project:** `churner`
**Investigation:** Retention vs. Churn baseline
**Analysis scope:** Overall distribution of the `Churn` target variable

---

## 1. Objective

Establish the overall retention and churn distribution of the customer population.

This provides the baseline against which subsequent exploratory investigations will compare churn rates across customer segments and attributes.

---

## 2. Dataset Population

The dataset contains:

* **Total customers:** 7,043
* **Retained customers:** 5,174
* **Churned customers:** 1,869

The counts reconcile exactly:

```text
5,174 + 1,869 = 7,043
```

---

## 3. Retention vs. Churn Distribution

| Customer Status | Customers |  Percentage |
| --------------- | --------: | ----------: |
| Retained        |     5,174 |      73.46% |
| Churned         |     1,869 |      26.54% |
| **Total**       | **7,043** | **100.00%** |

The observed churn rate is therefore **26.54%**, while **73.46%** of customers are classified as retained.

Retained customers outnumber churned customers by:

```text
5,174 - 1,869 = 3,305 customers
```

---

## 4. Interpretation

The dataset shows that approximately three quarters of the observed customer population is retained, while slightly more than one quarter is classified as churned.

The **26.54% churn rate is the baseline rate for subsequent exploratory analysis**.

Future investigations should compare the churn rate within specific customer segments against this overall baseline rather than interpreting segment percentages in isolation.

For example, if a particular customer segment later shows a churn rate substantially above 26.54%, that difference becomes an observation worth investigating.

---

## 5. What This Investigation Does Not Establish

This investigation examines only the distribution of the target variable.

It establishes:

> **How many customers churned.**

It does not establish:

> **Why customers churned.**

No relationship between `Churn` and other customer attributes has been measured in this investigation.

Therefore, no causal explanation for churn is made at this stage.

---

## 6. Analytical Principles

The following principles were applied:

1. **Counts are calculated directly from the dataset.**
2. **Percentages are derived from calculated counts.**
3. **The raw dataset is not modified.**
4. **No records are removed during this investigation.**
5. **The chart, table, and interpretation use the same calculated results.**
6. **Observed churn is not interpreted as causal evidence.**
7. **The overall churn rate is treated as a baseline for subsequent segment-level analysis.**

---

## 7. Reproducibility

The analysis is implemented in:

```text
scripts/explore_churn.py
```

The generated report is:

```text
reports/churn_eda_report.pdf
```

The report can be regenerated directly from the dataset using:

```text
python scripts/explore_churn.py
```

The verified result for this investigation is:

```text
Total customers: 7,043
Retained:         5,174  (73.46%)
Churned:          1,869  (26.54%)
```

---

## 8. Next Investigation

The next EDA investigation will examine customer attributes against the established **26.54% overall churn baseline**.

The objective will be to identify meaningful patterns and formulate hypotheses about which customer characteristics are associated with higher or lower observed churn.

Those hypotheses will remain hypotheses until supported by further analysis.
Next Investigation
Next Investigation 1
