# Data Dictionary — IBM Telco Customer Churn

This document describes every column in the IBM Telco Customer Churn dataset (`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`). Each record represents a single telecommunications customer, their account and service profile, and whether they left the company during the observation period.

The dataset contains 21 columns and 7,043 records.

| Column | Description | Business Meaning | Data Type (Conceptual) | Feature Category | Potential Preprocessing Notes |
|--------|-------------|------------------|------------------------|------------------|-------------------------------|
| `customerID` | Unique alphanumeric identifier assigned to each customer. | Distinguishes individual customer accounts; used for record tracking and joins. | String | Identifier | Unique per row; carries no predictive information on its own. |
| `gender` | Customer's gender. | Basic demographic attribute of the account holder. | Categorical (`Male`, `Female`) | Binary Categorical | Two distinct textual values; no missing entries observed. |
| `SeniorCitizen` | Indicates whether the customer is a senior citizen. | Flags an age-related demographic segment relevant to service usage patterns. | Categorical (encoded as `0`/`1`) | Binary Categorical | Stored numerically as `0`/`1` rather than text, unlike other binary fields. |
| `Partner` | Indicates whether the customer has a partner. | Reflects household relationship status, which can relate to account stability. | Categorical (`Yes`, `No`) | Binary Categorical | Two textual values; consistent with other yes/no fields. |
| `Dependents` | Indicates whether the customer has dependents. | Reflects household composition and financial responsibilities. | Categorical (`Yes`, `No`) | Binary Categorical | Two textual values; no missing entries observed. |
| `tenure` | Number of months the customer has stayed with the company. | Measures customer longevity and length of the relationship. | Numerical (integer, months) | Numerical | Ranges from 0 upward; value of 0 corresponds to newly joined customers. |
| `PhoneService` | Indicates whether the customer has phone service. | Identifies subscription to the core telephone product. | Categorical (`Yes`, `No`) | Binary Categorical | Two textual values; interacts with `MultipleLines`. |
| `MultipleLines` | Indicates whether the customer has multiple phone lines. | Reflects extent of phone service usage within the account. | Categorical (`Yes`, `No`, `No phone service`) | Multi-class Categorical | Includes a third value, `No phone service`, dependent on `PhoneService`. |
| `InternetService` | Type of internet service subscribed to. | Identifies the internet product tier held by the customer. | Categorical (`DSL`, `Fiber optic`, `No`) | Multi-class Categorical | Three distinct values; `No` indicates absence of internet service. |
| `OnlineSecurity` | Indicates whether the customer has online security add-on. | Reflects adoption of a value-added internet service. | Categorical (`Yes`, `No`, `No internet service`) | Multi-class Categorical | Third value, `No internet service`, dependent on `InternetService`. |
| `OnlineBackup` | Indicates whether the customer has online backup add-on. | Reflects adoption of a value-added internet service. | Categorical (`Yes`, `No`, `No internet service`) | Multi-class Categorical | Third value, `No internet service`, dependent on `InternetService`. |
| `DeviceProtection` | Indicates whether the customer has device protection add-on. | Reflects adoption of a value-added internet service. | Categorical (`Yes`, `No`, `No internet service`) | Multi-class Categorical | Third value, `No internet service`, dependent on `InternetService`. |
| `TechSupport` | Indicates whether the customer has technical support add-on. | Reflects adoption of a value-added internet service. | Categorical (`Yes`, `No`, `No internet service`) | Multi-class Categorical | Third value, `No internet service`, dependent on `InternetService`. |
| `StreamingTV` | Indicates whether the customer has streaming television service. | Reflects adoption of an entertainment-oriented internet service. | Categorical (`Yes`, `No`, `No internet service`) | Multi-class Categorical | Third value, `No internet service`, dependent on `InternetService`. |
| `StreamingMovies` | Indicates whether the customer has streaming movies service. | Reflects adoption of an entertainment-oriented internet service. | Categorical (`Yes`, `No`, `No internet service`) | Multi-class Categorical | Third value, `No internet service`, dependent on `InternetService`. |
| `Contract` | The customer's contract term. | Describes the commitment period, which relates to retention. | Categorical (`Month-to-month`, `One year`, `Two year`) | Multi-class Categorical | Three ordered-by-duration values represented as text. |
| `PaperlessBilling` | Indicates whether the customer uses paperless billing. | Reflects billing preference and channel of communication. | Categorical (`Yes`, `No`) | Binary Categorical | Two textual values; no missing entries observed. |
| `PaymentMethod` | The method used by the customer to pay bills. | Describes the payment channel, which can relate to account behavior. | Categorical (`Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)`) | Multi-class Categorical | Four distinct values; some labels include the `(automatic)` qualifier. |
| `MonthlyCharges` | The amount charged to the customer each month. | Represents the recurring revenue generated by the account. | Numerical (continuous, currency) | Numerical | Continuous values; scale differs from `TotalCharges`. |
| `TotalCharges` | The total amount charged to the customer over the tenure. | Represents cumulative revenue across the customer's lifetime. | Numerical (continuous, currency) | Numerical | Stored as text in the raw file; contains blank entries for customers with `tenure` of 0. |
| `Churn` | Indicates whether the customer left during the last period. | The outcome the project aims to predict: customer attrition. | Categorical (`Yes`, `No`) | Target | Label column; two textual values with class imbalance toward `No`. |
---

## Data Quality Investigation - TotalCharges

### Initial Hypothesis

`TotalCharges` is semantically numerical but is represented as text in the
raw dataset because one or more values prevent numeric interpretation.

### Investigation

The `TotalCharges` column was inspected for:

- pandas-detected missing values
- blank or whitespace-only values
- non-blank values that cannot be converted to numeric
- contextual information for affected records

A separate investigation also compared `TotalCharges` with the simple
relationship `tenure × MonthlyCharges` to determine whether it could serve as
a data-quality validation rule.

### Evidence

The dataset contains 7,043 records.

For `TotalCharges`:

- Raw dtype: `str`
- pandas-detected missing values: `0`
- blank/whitespace values: `11`
- non-blank, non-numeric values: `0`
- numerically convertible values: `7,032`

All 11 blank/whitespace records have `tenure = 0`.

### Engineering Interpretation

`TotalCharges` is conceptually a continuous numerical quantity despite being
represented as text in the raw dataset.

The 11 blank/whitespace representations appear to be a specific edge case
associated with zero-tenure customers. The available evidence does not justify
classifying these records as invalid or replacing the blank representation
with zero.

### Rejected Validation Rule

The relationship:

`TotalCharges = tenure × MonthlyCharges`

was investigated as a potential sanity check.

Among the 7,032 numerically convertible records:

- 614 had zero difference
- 6,418 had a non-zero difference
- mean absolute difference was `45.09`
- observed differences ranged from `-370.85` to `+373.25`

Therefore, the relationship is not suitable as a hard data-quality rule.

It may be useful for exploratory investigation, but it should not be used to
declare a `TotalCharges` value valid or invalid.

### Final Decision

The data-quality layer will detect and report:

- missing `TotalCharges`
- blank/whitespace representations
- non-numeric non-blank values
- contextual information for affected records

It will not modify the raw data, replace blank values with zero, or enforce
the `tenure × MonthlyCharges` relationship.

Transformation of `TotalCharges` will be addressed later during the
preprocessing stage.