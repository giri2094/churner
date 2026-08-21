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

## Investigation 2 — Tenure and Churn

**Project:** `churner`
**Investigation:** Tenure and churn
**Analysis scope:** The `tenure` attribute compared against the established churn baseline

---

## 1. Objective

Examine whether the `tenure` attribute is associated with observed churn, in two stages.

The first stage asked whether retained and churned customers show meaningfully different tenure distributions, describing `tenure` separately within each of the two churn populations.

The second stage followed up on the lower-tenure concentration that the first stage revealed. It measured the observed churn rate in the early-tenure region against the **26.54% overall churn baseline** established by Investigation 1, and then examined each individual tenure month on its own so that the shape of the pattern could be seen directly rather than assumed.

This is **exploratory descriptive analysis**. Its purpose is to describe what the recorded data shows and to formulate hypotheses worth carrying into modeling. No tenure bands were imposed, no threshold was searched for, and no predictive rule was derived.

---

## 2. Dataset Population

The investigation used the full customer population established in Investigation 1:

```text
Total customers: 7,043
Retained:        5,174
Churned:         1,869
Missing tenure:      0
Missing Churn:       0
```

Because neither `tenure` nor `Churn` has a missing value, every customer in the dataset can be placed in exactly one churn population and at exactly one tenure month. No record was imputed, excluded, or reassigned to achieve this.

---

## 3. Tenure Distribution Finding

The tenure distribution of each churn population was summarised separately:

| Statistic       | Retained | Churned |
| --------------- | -------: | ------: |
| Mean (rounded)  |    37.57 |   17.98 |
| Median          |       38 |      10 |
| Q1              |       15 |       2 |
| Q3              |       61 |      29 |
| IQR             |       46 |      27 |
| Min             |        0 |       1 |
| Max             |       72 |      72 |

The verified means at full precision are:

```text
Retained mean tenure: 37.56996521066873
Churned mean tenure:  17.979133226324237
```

Churned customers show **substantially lower observed tenure** than retained customers. The gap is visible in every measure of centre: the mean differs by roughly 19.59 months and the median by 28 months. It is also visible in position rather than only in average, since the churned Q3 of 29 months sits well below the retained median of 38 months, meaning three quarters of churned customers had a shorter recorded tenure than a typical retained customer.

The two populations differ in spread as well as centre. Churned tenure is more tightly concentrated toward the low end (IQR 27) than retained tenure (IQR 46), while both populations reach the same 72-month maximum. The distributions therefore overlap substantially: long-tenure churned customers exist, and the difference described here is a difference between populations, not a separation of them.

This is an **observed association** between recorded tenure and recorded churn status. It is not causal. The direction of any underlying mechanism is not established by these numbers, and both readings remain consistent with them: a customer may be more likely to leave early, and a customer who left necessarily stopped accruing tenure.

---

## 4. Early-Tenure Exploratory Finding

The tenure distribution above concentrated churned customers toward the low end of the range, so a low-tenure region was measured against the overall baseline. The verified result for tenure of 0 to 6 months inclusive is:

```text
Tenure range:          0–6 months inclusive
Customers:             1,481
Churned:                 784
Retained:                697
Churn rate:            52.94%
Population share:      21.03%
Churn contribution:    41.95%
Overall churn rate:    26.54%
Relative churn rate:   approximately 1.995×
Rate difference:       +26.40 percentage points
```

The population reconciles exactly:

```text
1,481 = 784 + 697
```

Customers with an observed tenure of 0 to 6 months show **substantially higher observed churn than the overall population**: 52.94% against the 26.54% baseline, a difference of 26.40 percentage points and a relative rate of approximately 1.995 times the baseline.

Three quantities describe this region and are easy to confuse, so they are reported side by side. The **churn rate** (52.94%) is the share of this region's own customers who churned. The **population share** (21.03%) is how much of the dataset the region holds. The **churn contribution** (41.95%) is how much of all observed churn the region accounts for. A region can carry a high churn rate while representing a small part of the population, which is exactly the case here: roughly a fifth of customers account for roughly two fifths of all observed churn.

### How this region was chosen, and what that costs

The 0–6 month region was **selected post-hoc, after inspection of the tenure distribution**. It was not specified in advance. The first stage described the two populations, the churned population was seen to concentrate toward low tenure, and only then was this region defined to measure what that concentration suggested.

A region selected because it looked interesting will tend to look interesting when measured. The finding above is therefore a lead worth following rather than a validated result, and specifically:

* It is an **exploratory slice**, defined in this investigation for descriptive measurement.
* It is **not a validated business threshold**.
* It is **not a proven churn threshold**.
* It is **not a production rule**.
* It **does not establish causality**.

The choice of 6 months does not establish an analytical boundary. Nothing in the data marks a change at that point; it is where this exploratory slice was drawn, and section 5 shows that churn inside the region is not uniform, which is itself evidence against reading the boundary as meaningful.

---

## 5. Month-Level Refinement

Because the 0–6 month region was an arbitrary aggregate, the observed churn rate was then computed for each individual tenure month, using only tenure values actually present in the data. No months were grouped, and no month absent from the data was invented.

The full month-level result covers 73 observed tenure months and lives in the analysis module rather than in this document. The verified observations that matter for the early-tenure region are:

| Tenure | Customers | Churned | Churn rate |
| -----: | --------: | ------: | ---------: |
|      0 |        11 |       0 |      0.00% |
|      1 |       613 |     380 |     61.99% |
|      2 |       238 |     123 |     51.68% |
|      3 |       200 |      94 |     47.00% |
|      4 |       176 |      83 |     47.16% |
|      5 |       133 |      64 |     48.12% |
|      6 |       110 |      40 |     36.36% |

These rows reconcile with section 4, which is a useful consistency check across the two analyses:

```text
11 + 613 + 238 + 200 + 176 + 133 + 110 = 1,481 customers
 0 + 380 + 123 +  94 +  83 +  64 +  40 =   784 churned
```

Across the broader month-level result, observed churn **generally declines as tenure increases**, while individual months show substantial variation. Later-tenure examples include:

```text
18 months → 24.74%
23 months → 15.29%
45 months →  9.84%
52 months → 10.00%
60 months →  7.89%
64 months →  5.00%
72 months →  1.66%
```

The decline is a general tendency rather than a step or a strictly monotonic sequence. Months 3, 4, and 5 illustrate this directly: their observed rates of 47.00%, 47.16%, and 48.12% rise slightly across consecutive months inside an overall declining pattern. The same happens later in the range, where month 52 (10.00%) sits marginally above month 45 (9.84%) despite the longer tenure.

Month 0 deserves a specific caution. Its observed churn rate of 0.00% rests on only 11 customers, which is a far smaller population than any neighbouring month, so it should not be read as a substantive finding about newly joined customers.

The key interpretation is:

> The elevated churn observed in the early-tenure region is strongest at one month and generally declines across subsequent tenure months, while month-to-month variation remains.

These are descriptive observations only. In particular, this refinement does **not** establish that month 1 is a churn threshold, that month 6 is a threshold, that later tenure causes lower churn, or that early tenure causes churn.

---

## 6. Analytical Interpretation

The evidence from this investigation forms a hierarchy, from the most robust observation to the most qualified:

1. **The retained and churned populations have substantially different tenure distributions.** This is the broadest and best-supported observation, measured over the whole population with no region selected in advance.
2. **The 0–6 month region has a substantially higher observed churn rate than the overall baseline** (52.94% against 26.54%). This is a descriptive pattern measured on a post-hoc slice, so it is weaker evidence than the point above.
3. **Month-level analysis shows the early-tenure elevation is not uniform.** It is strongest at month 1 (61.99%) and generally declines thereafter, which means the region-level rate is an average over genuinely different months rather than a property shared evenly across them.
4. **Individual month rates must be interpreted alongside their customer counts.** Sample sizes vary substantially by tenure month, so an isolated monthly rate can be extreme on very little evidence. This is why the month-level result reports counts next to every rate.
5. **The overall relationship is a descriptive association, not causal evidence.** Nothing here identifies a mechanism, a direction of effect, or a threshold.

Read together, these points support one exploratory finding: recorded tenure carries a strong observed association with recorded churn, concentrated in but not confined to the low-tenure region. That is a hypothesis to carry forward, not a conclusion about why customers leave.

---

## 7. Methodological Limitations

The following limitations apply to this investigation and constrain how its results may be used.

**Post-hoc selection.** The 0–6 month region was selected after inspecting the tenure distribution. It must not be treated as a pre-specified hypothesis or as a validated threshold, and its measured effect size should be read with the knowledge that the region was chosen because it looked notable.

**No causal inference.** The analysis does not establish that tenure causes churn. Observed differences between the populations are associations in recorded data.

**No statistical significance testing.** The month-level investigation is descriptive. No significance test, confidence interval, or p-value was computed, so no claim about whether any observed difference would generalise beyond this dataset is made or implied.

**Month-level sample size.** Customer counts vary substantially by tenure month, from 11 customers at month 0 to 613 at month 1. Isolated high or low monthly rates should not be interpreted without considering the number of customers represented.

**No production rule.** No tenure threshold, tenure band, risk category, or scoring rule has been created from this analysis. The analysis modules return descriptive summaries only.

---

## 8. Engineering Implication

The engineering conclusion for the project is:

> Tenure appears sufficiently informative to retain as a candidate predictor for subsequent baseline modeling. The observed relationship should be evaluated through model-based analysis rather than converted into a hand-crafted churn rule.

This marks the transition from EDA toward the modeling milestone. The value of this investigation is that it identifies `tenure` as worth carrying into a model and, just as importantly, records why the tempting shortcut was declined: the month-level result shows that any hand-drawn tenure boundary would encode an arbitrary cut through a gradually changing pattern. A model can weigh tenure directly and be evaluated on held-out data, which is the appropriate way to test whether the association observed here has predictive value.

---

## 9. Reproducibility

The analysis is implemented in three modules, each answering one question:

```text
src/churner/analysis/analyze_tenure_churn.py
src/churner/analysis/analyze_early_tenure_churn.py
src/churner/analysis/analyze_tenure_by_month.py
```

These modules hold the reproducible detailed computation, including the complete 73-row month-level result that this document deliberately summarises rather than reproduces.

The month-level analysis has its dedicated tests:

```text
tests/analysis/test_analyze_tenure_by_month.py
```

The current complete test result is:

```text
80 passed in 0.45s
```

The focused month-level test result is:

```text
27 passed in 0.34s
```

The suite can be run from the project root with `pytest`.

The PDF report at `reports/churn_eda_report.pdf` was not modified as part of this investigation. The month-level analysis was executed directly from the analysis module against the raw dataset.

---

## 10. Next Investigation

The tenure investigation is **complete**. Its results are recorded above, and `tenure` is retained as a candidate predictor for the modeling milestone rather than converted into a rule.

The next EDA investigation will examine a further customer attribute against the established **26.54% overall churn baseline**, following the same pattern used here: describe the attribute's relationship to observed churn, compare segment-level churn rates against the baseline, and record what the result does and does not establish.

No specific attribute or expected result is committed to in advance. Any pattern found will be treated as an observed association and will remain a hypothesis until supported by further analysis.
No
